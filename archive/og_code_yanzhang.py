import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only GPU 1
os.environ["TOKENIZERS_PARALLELISM"] = "false" #To stop tokenizer Warning, is set to false anyway
from lora_estimation import load_lora_bert_surrogate, train_cerm_pairwise, predict_with_model, compute_group_auc_difference, train_surrogate, tokenize_batch, df_map
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sklearn.metrics import roc_auc_score
import random
import logging
import sys
import torch
import time
import pandas as pd
import numpy as np
import gc

print("Torch is using device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

# === Load audit dataset ===

df = pd.read_csv("jigsaw_group.csv")
df["id"] = df["id"].astype("category").cat.codes 
condition = (df['group'] == 'white') | (df['group'] == 'black')
dataset_D = df[condition].copy() #Jigsaw data reduced dataset
dataset_D = dataset_D.sample(n=10000) #Subsample of D to work with slow CERM
dataset_D['weights'] = np.ones(len(dataset_D)) / len(dataset_D)

# === Black Box API ===

# Load SBIC training data and create black-box model h* 
df_agg = pd.read_csv('SBIC_group.csv')
train_df, test_df = train_test_split(df_agg[['text', 'true_label', 'group']], test_size=0.1, random_state=42)

# for simplicity first logistic regression
api_model = make_pipeline(
    TfidfVectorizer(max_features=50000, stop_words='english'),
    LogisticRegression(solver='liblinear')
)

biased_df = train_df.copy()
mask = (biased_df['group'] == 'black') & (np.random.rand(len(biased_df)) < 0.9)
biased_df['true_label'] = biased_df['true_label'].astype(int)
biased_df.loc[mask, 'true_label'] = 1 - biased_df.loc[mask, 'true_label']
api_model.fit(biased_df['text'], biased_df['true_label'])

def black_box_api(comments):
    return api_model.predict_proba(comments)[:, 1]

####### just for the black box api we have here

# === AUC Difference witch Black Box API ===

def compute_blackbox_auc_difference(texts, labels, groups, group1='white', group2='black'):
    scores = black_box_api(texts)
    df = pd.DataFrame({'score': scores, 'true_label': labels, 'group': groups})

    try:
        auc1 = roc_auc_score(df[df['group'] == group1]['true_label'],
                             df[df['group'] == group1]['score'])
    except ValueError:
        auc1 = 0

    try:
        auc2 = roc_auc_score(df[df['group'] == group2]['true_label'],
                             df[df['group'] == group2]['score'])
    except ValueError:
        auc2 = 0

    return auc1 - auc2

delta_auc_blackbox = compute_blackbox_auc_difference(
        texts=dataset_D['text'],
        labels=dataset_D['true_label'].astype(int),
        groups=dataset_D['group']
    )

# === Initialize audit states ===

def stratified_sampling(size, dataset):

    labels = dataset.true_label.to_numpy()
    groups = dataset.group.to_numpy()
    size_part = int(size/4)

    # Random initialization of constraint set T
    white_idx_True = np.where((groups == 'white') & (labels == True))[0]
    white_idx_False = np.where((groups == 'white') & (labels == False))[0]
    black_idx_True = np.where((groups == 'black') & (labels == True))[0]
    black_idx_False = np.where((groups == 'black') & (labels == False))[0]
    
    labeled_white_idx_True = np.random.choice(white_idx_True, size=size_part, replace=False)
    labeled_white_idx_False = np.random.choice(white_idx_False, size=size_part, replace=False)
    labeled_black_idx_True = np.random.choice(black_idx_True, size=size_part, replace=False)
    labeled_black_idx_False = np.random.choice(black_idx_False, size=size_part, replace=False)

    labeled_idx = np.concatenate([labeled_white_idx_True, labeled_white_idx_False,
                                        labeled_black_idx_True, labeled_black_idx_False])

    print(f'Labeled Idx:{len(labeled_idx)}, Size:{size}')    
    if len(labeled_idx) < size:
        extra = len(labeled_idx)-size
        idx_extra = np.where((groups == 'white') & (labels == True)|(groups == 'white') & (labels == False)|(groups == 'black') & (labels == True)(groups == 'black') & (labels == False))[0]
        labeled_idx_extra = np.random.choice(idx_extra, size=extra, replace=False)
        labeled_idx = np.concatenate([labeled_idx, labeled_idx_extra])
    
    dataset = dataset_D[dataset_D.index.isin(labeled_idx)]

    return df[df.index.isin(labeled_idx)]

def initialize_S(size):
    dataset_S = stratified_sampling(size, dataset_D)
    # QUERY black-box with T to include bb_score
    texts = dataset_S['text'].tolist()  # extract text column
    bb_output = black_box_api(texts)
    dataset_S = dataset_S.copy()
    dataset_S['bb_score'] = bb_output
    return dataset_S


def eval_h(base_model, df_D, df_D_mapped, inputs_D, df_T_mapped, constraint_pred, epochs_opt, batch_size, lambda_penalty, tokenizer, Maximize):
    h = train_cerm_pairwise(base_model, df_D, df_D_mapped, df_T_mapped, constraint_pred, epochs_opt, batch_size, lambda_penalty, tokenizer, maximize=Maximize)

    with torch.no_grad():
        pred_h = compute_group_auc_difference(h, inputs_D, df_D)
        
    if isinstance(pred_h[1], torch.Tensor):
        eval_scores_h = torch.tensor(pred_h[1].to_numpy()).detach().cpu()
    else:
        eval_scores_h = torch.tensor(pred_h[1].to_numpy()).detach().cpu()
    delta = pred_h[0]

    ### clean übergeben
    h.to('cpu')
    del h
    gc.collect()
    torch.cuda.empty_cache()

    return delta, eval_scores_h

def delta_progress(df_new, df_old):
    
    delta_auc = []
    for i in range(len(df_new)):
        df_progress = pd.concat([df_new.iloc[:i],df_old])
        delta = compute_blackbox_auc_difference(
            texts=df_progress['text'],
            labels=df_progress['true_label'].astype(int),
            groups=df_progress['group']
        )
        delta_auc.append(abs(delta-delta_auc_blackbox))
    return delta_auc

# === Compute disagreements ===

def compute_diff_auc(S, T):

    T = T.sort_values('weights')
    auc_func = delta_progress(T,S)

    delta_auc_blackbox_S= compute_blackbox_auc_difference(
        texts=S['text'],
        labels=S['true_label'].astype(int),
        groups=S['group']
    )
    diff_auc = delta_auc_blackbox -delta_auc_blackbox_S

    return diff_auc, auc_func

# === New Samling to compare stratified and random sampling ===


def new_sampling(size, dataset_old, sample_type):
    dataset = dataset_D[~dataset_D['id'].isin(dataset_old['id'])]
    dataset = dataset.sample(n=size) if sample_type == 'random' else stratified_sampling(size, df)
    pd.concat([df,dataset_old])
    
    delta = compute_blackbox_auc_difference(
            texts=dataset['text'],
            labels=dataset['true_label'].astype(int),
            groups=dataset['group']
        )
    
    delta_func = delta_progress(dataset, dataset_old)

    return dataset, delta, delta_func

# === Append new T to S ===

def make_S(dataset_S, dataset_T, iteration):

    if iteration == 0:
        dataset_S = dataset_T.copy()
        dataset_S['bb_score'] = black_box_api(dataset_S['text'].tolist()) 

    else:
        # Find only new entries in T (not yet in S)
        new_T = dataset_T[~dataset_T['text'].isin(dataset_S['text'])].copy()

        # Query black-box model and save scores
        if len(new_T['text'].tolist()) > 1:
            new_T['bb_score'] = black_box_api(new_T['text'].tolist()) 
            dataset_S = pd.concat([dataset_S, new_T], ignore_index=True)
        elif len(new_T['text'].tolist()) == 1: 
            new_T['bb_score'] = black_box_api(new_T['text']) 
            dataset_S = pd.concat([dataset_S, new_T], ignore_index=True)
        else:
            print('No new T')


    return dataset_S


def active_auc_auditing(model, size, max_iterations=5, epochs_sur = 3, epochs_opt=3, batch_size=20, lambda_penalty=5.0, epsilon=1e-2, change='size'):

    LOG_PATH = f"log/{change}.csv"
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'w') as f:
            f.write(f"Iteration,delta_random,delta_stratified,delta_auc\n\nT Size = {size}, Epochs = {epochs_opt}, Batch Size = {batch_size}, Lambda = {lambda_penalty}, epsilon = {epsilon}\n")
    else:
        with open(LOG_PATH, 'a') as f:
            f.write(f"\nT Size = {size}, Epochs = {epochs_opt}, Batch Size = {batch_size}, Lambda = {lambda_penalty}, epsilon = {epsilon}\n")

    if model == 'lora':
        tokenizer, base_model = load_lora_bert_surrogate()
        print("Train surrogate model")
    else:
        raise ValueError('Model does not exist, please choose another model')

    # Get Data D
    df_D, df_D_mapped = df_map(dataset_D, tokenizer, False)
    
    inputs_D = {
        "input_ids": torch.tensor(df_D_mapped['input_ids']).long(),
        "attention_mask": torch.tensor(df_D_mapped['attention_mask']).long(),
        "labels": torch.tensor(df_D_mapped['labels']).long(),
        "id":  df_D_mapped['id']
    }

    # Initialize Dataset S for first iteration
    dataset_S = initialize_S(size) # every step new T
    df_S, df_S_mapped = df_map(dataset_S, tokenizer, True)
    

    # Initialize Random and Stratified Datasets
    dataset_random = dataset_D.sample(n=size)
    dataset_stratified = stratified_sampling(size, dataset_D)

    # Train Surrogate Model
    surrogate_model = train_surrogate(base_model, df_S, df_S_mapped, epochs_sur, batch_size)
    surrogate_model = surrogate_model.to('cpu')
    torch.cuda.empty_cache()


    # === Random Initialization of thresholds τ_x ~ Exponential(λ) ===

    num_examples = len(dataset_D)
    delta = 0.05  # confidence level, can be tuned
    M = 100  # mistake bound or surrogate model complexity, tune as needed
    H_size = 1e6  # size of hypothesis class, e.g. for BERT-based class with LoRA

    lambda_param = np.log(H_size * 2**M / delta)  # scale parameter for Exponential
    thresholds = np.random.exponential(scale=1.0/lambda_param, size=num_examples)  # τ_x ∼ Exp(λ)

    for iteration in range(max_iterations):
        print(f"\n=== Iteration {iteration + 1} ===")
        start = time.time()

        # Drop 'bb_score column, as it is not known for further T

        weights = np.ones(len(dataset_D)) / len(dataset_D)
        dataset_T = dataset_S.drop(columns='bb_score')
        df_T, df_T_mapped = df_map(dataset_T, tokenizer, False)

        counter = 0
        
        while True:
            

            with torch.no_grad():
                constraint_pred = predict_with_model(list(dataset_T['text']), tokenizer, surrogate_model)
            
            # Train h1 and h2
            delta1, eval_scores_h1 = eval_h(base_model, df_D, df_D_mapped, inputs_D, df_T_mapped, constraint_pred, epochs_opt, batch_size, lambda_penalty, tokenizer, True)
            print(f'h1 DONE for ITERATION {iteration + 1}.{counter}')

            delta2, eval_scores_h2 = eval_h(base_model, df_D, df_D_mapped, inputs_D, df_T_mapped, constraint_pred, epochs_opt, batch_size, lambda_penalty, tokenizer, False)
            print(f'h2 DONE for ITERATION {iteration + 1}.{counter}')

            auc_diff = abs(delta1 - delta2)
            print(f'AUC(h1)-AUC(h2):{auc_diff:.4f}')
            if auc_diff  <= 2 * epsilon:
                print(f"Stopping: AUC difference {auc_diff:.4f} within epsilon tolerance.")
                break

            disagree_indices = [
                i for i in range(len(eval_scores_h1))
                if abs(eval_scores_h1[i] - eval_scores_h2[i]) > epsilon
            ]
            
            disagree_indices = np.array(disagree_indices)
            while sum(weights[disagree_indices]) < 1: # Irgendwas stimmt hier noch nicht!
                
                weights = np.array([
                    2 * weights[i] if i in disagree_indices else weights[i]
                    for i in range(len(weights))
                ])
            #weights[disagree_indices] *= 2.0
            #weights = weights / weights.sum()
            selected_indices = np.where(weights > thresholds)[0]

            dataset_D['weights'] = weights
            dataset_T = dataset_D.iloc[selected_indices].copy()
            #dataset_T = pd.concat([dataset_T, new_T], ignore_index=True)
            df_T, df_T_mapped = df_map(dataset_T, tokenizer, False)

            counter += 1

        # === Compare fairness difference (ΔAUC) between black-box h* and surrogate h^ ===
        diff_auc, delta_auc_progress = compute_diff_auc(df_S, df_T)

        dataset_S = make_S(dataset_S, dataset_T, iteration)
        df_S, df_S_mapped = df_map(dataset_S, tokenizer, True)
        surrogate_model = train_surrogate(surrogate_model.to("cuda"), df_S, df_S_mapped, epochs_sur, batch_size)
        surrogate_model = surrogate_model.to('cpu')
        torch.cuda.empty_cache()

        inputs_S = {
            "input_ids": torch.tensor(df_S_mapped['input_ids']).long(),
            "attention_mask": torch.tensor(df_S_mapped['attention_mask']).long(),
            "labels": torch.tensor(df_S_mapped['labels']).long(),
            "id":  df_S_mapped['id']
        }


        # Random sampling baseline with same T size
        dataset_random, delta_random, delta_random_progress = new_sampling(len(dataset_T), dataset_random, 'random')
        dataset_stratified, delta_stratified, delta_stratified_progress = new_sampling(len(dataset_T), dataset_stratified, 'stratified')

        print(f'ΔAUC(Active):{diff_auc:.4f}, ΔAUC(Stratified):{delta_stratified:.4f}, ΔAUC(Random):{delta_random:.4f}')
        print(f'S Size:{len(dataset_S)}, Size Random: {len(delta_random_progress)}, Size Stratifies: {len(delta_stratified_progress)}, Size Active: {len(delta_auc_progress)}')

        now = (time.time()-start)/60

        # Append log row to CSV
        with open(LOG_PATH, 'a') as f:
            for i in range(len(delta_random_progress)):
                f.write(f"{iteration+1},{delta_random_progress[i]:.4f},{delta_stratified_progress[i]:.4f},{delta_auc_progress[i]:.4f}\n")

        if diff_auc < epsilon:
            print("Stopping: surrogate fairness estimate converged to black-box.")
            break

    final_values = compute_group_auc_difference(surrogate_model, inputs_S, df_S)
    print("Final surrogate ΔAUC:", final_values[0])
    return final_values



def main():
    args = sys.argv[1:]
    if len(args) == 0:
        logging.info("Please provide arguments as follows: og_code_yanzhang.py --model <model> --size_T <size_T> --iterations <iterations> --epochs <epochs> --batch_size <batch_size> --lambda_penalty <lambda_penalty> --epsilon <epsilon> --change <change>")
        return
    if args[0] == '--model':
        model = args[1]
    else:
        logging.exception("Please provide model from <lora>, <something else> with the argument --model")
        return
    if args[2] == '--size_T':
        size_T = int(args[3])
    else:
        logging.exception("Please provide as integer the size of the initial guess for the dataset T --size_T")
        return
    if args[4] == '--iterations':
        iterations = int(args[5])
    else:
        logging.exception("Please provide the maximal iteration as an integer --iterations")
        return
    if args[6] == '--epochs_sur':
        epochs_sur = int(args[7])
    else:
        logging.exception("Please provide the epochs for the training --epochs")
        return
    if args[8] == '--epochs_opt':
        epochs_opt = int(args[9])
    else:
        logging.exception("Please provide the epochs for the training --epochs")
        return
    if args[10] == '--batch_size':
        batch_size = int(args[11])
    else:
        logging.exception("Please provide the batch size for the training --batch_size")
        return
    if args[12] == '--lambda_penalty':
        lambda_penalty = float(args[13])
    else:
        logging.exception("Please provide the lambda penalty for the loss function --lambda_penalty")
        return
    if args[14] == '--epsilon':
        epsilon = float(args[15])
    else:
        logging.exception("Please provide a number for the threshhold of |ΔAUC(h1)-ΔAUC(h2)| --epsilon")
        return
    if args[16] == '--change':
        change =  args[17]
    else:
        logging.exception("Please provide what changed in str format")
        return

    active_auc_auditing(model, size_T, iterations, epochs_sur, epochs_opt, batch_size,  lambda_penalty, epsilon, change)

if __name__ == '__main__':
    main()