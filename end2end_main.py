import os
import argparse
import pandas as pd
import torch
import json
from glob import glob
from losses import SimilarityLoss, FASG_Loss, tripletLoss
from model import End2EndModel
from model_utils import  build_optimizer_and_scheduler, train_end2endmodel
from dataset import SynDataset
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
import itertools
import ast
import random
from tqdm import tqdm
from datetime import datetime
from set_up_configs import set_up_end2end_configs
# from sentence_transformers import SentenceTransformer


LR_DROP = 10
N_EPOCH = 50
GRADUAL_UNFREEZE = True

def remove_surrogates(text):
    return re.sub(r'[\uD800-\uDFFF]', '', text)

def process_txt(txt_path):

    with open(txt_path, 'r') as file:
        content = file.read()
           
    dicts = []
    for entry in content.split('\n'):
        entry = entry.strip() 
        if entry:  
            try:
                parsed = ast.literal_eval(entry)  
                if isinstance(parsed, dict) and parsed: 
                    if isinstance(parsed['scores'][0], float):
                        dicts.append(parsed)
            except (SyntaxError, ValueError):
                pass
    return dicts



def main(configs):

    dicts = process_txt('/scratch/zc1592/small_data/data/synth_data_v2_24Nov.txt')
    random.seed(42)
    random.shuffle(dicts)
    # dicts = dicts[:1000]
    split_ratio = 0.8  
    split_index = int(split_ratio * len(dicts))
    train = dicts[:split_index]
    eval = dicts[split_index:]


    test = process_txt('/scratch/zc1592/small_data/data/testset_synth_data_v1.txt')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    base_model = AutoModel.from_pretrained(configs.model_name)
    tokenizer = AutoTokenizer.from_pretrained(configs.model_name)


    
    train_dataset = SynDataset(train, tokenizer, configs.n_features, configs.prompt_len)
    eval_dataset = SynDataset(eval, tokenizer, configs.n_features, configs.prompt_len)
    test_dataset = SynDataset(test, tokenizer, configs.n_features, configs.prompt_len)
    train_loader = DataLoader(train_dataset, batch_size = configs.batch_size)
    eval_loader = DataLoader(eval_dataset, batch_size = configs.batch_size)
    test_loader = DataLoader(test_dataset, batch_size = configs.batch_size)



    name = configs.model_name.split('/')[-1]
    
    current_time = datetime.now()
    current_time_str = current_time.strftime("%m-%d-%H-%M")
    exp_components = [
        N_EPOCH,
        configs.tmp,
        configs.loss, 
        configs.sae_model,
        configs.hook_id,
        current_time_str
    ]
    exp_components = list(map(str, exp_components))
    exp_id = '_'.join(exp_components)
    
    exp_export = f'/scratch/zc1592/small_data/experiments/end2end'
    exp_saving_path = os.path.join(exp_export, exp_id)
    os.makedirs(exp_saving_path, exist_ok=True)


    model = End2EndModel(
        base_model=base_model, 
        max_new_token = configs.max_new_token,
        tmp = configs.tmp,
        sae_model = configs.sae_model,
        release = configs.release,
        hook_id = configs.hook_id
    ).to(device)

    config_dict = vars(configs)
    with open(os.path.join(exp_saving_path, 'configs.json'), "w") as json_file:
        json.dump(config_dict, json_file, indent=4)

    score_writer = open(
        os.path.join(exp_saving_path, "eval_results.txt"), mode="w", encoding="utf-8"
    )

    sentence_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    loss_tokenizer = AutoTokenizer.from_pretrained(sentence_model_name)
    loss_encoder = AutoModel.from_pretrained(sentence_model_name).to(device)
    if configs.task == 'train':
        if configs.loss == 'triplet':
            criterion = tripletLoss(loss_tokenizer, loss_encoder)
        elif configs.loss == 'fasg':
            criterion = FASG_Loss(loss_tokenizer, loss_encoder)
        

        optimizer, scheduler = build_optimizer_and_scheduler(model, configs.lr, LR_DROP)

        train_end2endmodel(
            model, 
            train_loader, 
            eval_loader, 
            criterion, 
            optimizer, 
            scheduler, 
            device,
            num_epochs=N_EPOCH, 
            saving_dir=exp_saving_path, 
            max_unfreeze_layers=configs.n_unfrozen, 
            gradual_unfreeze=GRADUAL_UNFREEZE,
            score_writer = score_writer
        )
    elif configs.task == 'test':
        if configs.ckpt_path is not None and os.path.exists(configs.ckpt_path):
            ckpt = torch.load(configs.ckpt_path)
            model.load_state_dict(ckpt)
            model.eval()  
            output_list = []
            saving_dir = os.path.dirname(configs.ckpt_path)
            with torch.no_grad(): 
                for input_ids, attention_masks, scores in tqdm(test_loader, desc="Evaluating"):
                    input_ids = input_ids.to(device)
                    attention_masks = attention_masks.to(device)
                    scores = scores.to(device)

                    outputs = model(input_ids, attention_masks)  
                    if isinstance(outputs, tuple):
                        _, reg_preds = outputs
                        output_list.append(reg_preds.cpu())
                    else:
                        output_list.append(outputs.cpu())
            output_tensor = torch.cat(output_list, dim=0)
            torch.save(output_tensor, os.path.join(saving_dir,'output.pt'))
            rounded = torch.round(output_tensor, decimals = 2)
            for i in range(output_tensor.shape[0]):
                print(output_tensor[i,:])
  
                        
                   
if __name__ == '__main__':

    configs = set_up_end2end_configs()
    main(configs)


