import os
import argparse
import pandas as pd
import torch
import json
from glob import glob
from losses import MSEWithL1Loss, MaskedMSEWithL1Loss, CombinedLoss,  FocalLoss
from model import CustomTransformerWithHeads
from model_utils import train_model, build_optimizer_and_scheduler
from dataset import SynDataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer
import itertools
import ast
import random
from tqdm import tqdm
from datetime import datetime


LR_DROP = 10
N_EPOCH = 50
GRADUAL_UNFREEZE = True


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
    split_ratio = 0.8  
    split_index = int(split_ratio * len(dicts))
    train = dicts[:split_index]
    eval = dicts[split_index:]


    test = process_txt('/scratch/zc1592/small_data/data/testset_synth_data_v1.txt')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    base_model = AutoModel.from_pretrained(configs.model_name)
    tokenizer = AutoTokenizer.from_pretrained(configs.model_name)


    
    train_dataset = SynDataset(train, tokenizer, configs.n_features)
    eval_dataset = SynDataset(eval, tokenizer, configs.n_features)
    test_dataset = SynDataset(test, tokenizer, configs.n_features)
    train_loader = DataLoader(train_dataset, batch_size = configs.batch_size)
    eval_loader = DataLoader(eval_dataset, batch_size = configs.batch_size)
    test_loader = DataLoader(test_dataset, batch_size = configs.batch_size)

    name = configs.model_name.split('/')[-1]
    
    current_time = datetime.now()
    current_time_str = current_time.strftime("%m-%d-%H-%M")
    exp_components = [
        name,
        N_EPOCH,
        configs.sparsity_weight,
        configs.reg_weight,
        configs.cls_weight,
        configs.focal_alpha,
        configs.focal_gamma
    ]
    exp_components = list(map(str, exp_components))
    exp_id = '_'.join(exp_components)
    
    exp_export = f'/scratch/zc1592/small_data/experiments/{configs.loss_type}'
    exp_saving_path = os.path.join(exp_export, exp_id)
    os.makedirs(exp_saving_path, exist_ok=True)


    model = CustomTransformerWithHeads(
        base_model=base_model, 
        n_features=configs.n_features, 
        loss_type = configs.loss_type
    ).to(device)

    config_dict = vars(configs)
    with open(os.path.join(exp_saving_path, 'configs.json'), "w") as json_file:
        json.dump(config_dict, json_file, indent=4)

    score_writer = open(
        os.path.join(exp_saving_path, "eval_results.txt"), mode="w", encoding="utf-8"
    )
    if configs.task == 'train':
        if configs.loss_type == 'focal':
            criterion = FocalLoss(configs)
        elif configs.loss_type == 'reg_only':
            criterion = MaskedMSEWithL1Loss(configs)
        elif configs.loss_type == 'reg_cls':
            criterion  = CombinedLoss(configs)
        elif configs.loss_type == 'kl':
            criterion = KLdivergenceLoss(configs)
        optimizer, scheduler = build_optimizer_and_scheduler(model, configs.lr, LR_DROP)

        train_model(model, 
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
            ckpt = torch.load(configs.ckpt_path, map_location=torch.device('cpu'))
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
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_name', 
        type=str, 
        default='bert-base-uncased', 
        help="Model to use for training"
    )
    parser.add_argument(
        '--n_unfrozen', 
        type=int, 
        default=12, 
        help="Number of layers to unfreeze"
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        default=0.0005, 
        help="initial learning rate"
    )
    parser.add_argument(
        '--l1_lambda', 
        type=float, 
        default=0.001, 
        help="sparsity penalty"
    )
    parser.add_argument(
        '--n_features', 
        type=int, 
        default=768, 
        help="Number of features considered."
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=32, 
        help="Batch size."
    )
    parser.add_argument(
        '--mse_weight', 
        type=int, 
        default=10, 
        help="scale up mse loss in msewithl1loss"
    )
    parser.add_argument(
        '--task', 
        type=str, 
        default='train'
    )
    parser.add_argument(
        '--ckpt_path', 
        type=str, 
        default='/scratch/zc1592/small_data/experiments/focal/bert-base-uncased_0.01_0.5_0.5_1.0_1/model_epoch_28.pt'
    )
    parser.add_argument(
        "--loss_type", 
        type=str, 
        default='focal',
        help="loss function to use (focal, mseWithL1Loss, maskedMseWithL1Loss)"
    )
    parser.add_argument(
        '--sparsity_weight', 
        type=float, 
        default=0.001, 
        help="sparsity penalty"
    )
    parser.add_argument(
        '--reg_weight', 
        type=float, 
        default=1.0, 
        help="regression loss weight"
    )
    parser.add_argument(
        '--cls_weight', 
        type=float, 
        default=1.0, 
        help="classification loss weight"
    )
    parser.add_argument(
        '--focal_alpha', 
        type=float, 
        default=0.9, 
    )
    parser.add_argument(
        '--focal_gamma', 
        type=int, 
        default=2, 
    )
    parser.add_argument(
        '--huber_delta', 
        type=float, 
        default=0.1, 
    )


    configs = parser.parse_args()
    main(configs)


