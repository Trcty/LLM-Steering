import os
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import pandas as pd
from torch.amp import autocast, GradScaler
import glob


def evaluate_regression(model, eval_loader, criterion, device):
    model.eval()  
    total_eval_loss = 0
    with torch.no_grad(): 
        for input_ids, attention_masks, scores in tqdm(eval_loader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            scores = scores.to(device)

            outputs = model(input_ids, attention_masks)  
            loss = criterion(outputs, scores) 
            total_eval_loss += loss.item()  

    avg_eval_loss = total_eval_loss / len(eval_loader)  
    return avg_eval_loss

def evaluate_model(model, eval_loader, criterion, device):
    model.eval()  
    total_eval_loss = 0
    reg_loss = 0
    cls_loss = 0
    sparse_loss = 0
    output_list = []
    with torch.no_grad(): 
        for input_ids, attention_masks, scores in tqdm(eval_loader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            scores = scores.to(device)
            outputs = model(input_ids, attention_masks)
            if isinstance(outputs, tuple):
                class_preds, reg_preds = outputs
                loss_dict = criterion(class_preds, reg_preds, scores)
                output_list.append(reg_preds.cpu())

            else:
                loss_dict = criterion(outputs, scores) 
                output_list.append(outputs.cpu())
            loss = loss_dict['total_loss']
            total_eval_loss += loss.item()  
            reg_loss +=loss_dict['reg_loss'].item()
            cls_loss += loss_dict.get('cls_loss', torch.tensor(0.0)).item()
            sparse_loss += loss_dict.get('sparse_loss', torch.tensor(0.0)).item()

    output_tensor = torch.cat(output_list, dim=0)
    rounded = torch.round(output_tensor, decimals = 1)
    non_zero_counts = (rounded != 0).sum(dim=1)
    average_non_zero_elements = non_zero_counts.float().mean()

    avg_eval_loss = total_eval_loss / len(eval_loader) 
    avg_reg_loss = reg_loss / len(eval_loader)
    avg_cls_loss = cls_loss / len(eval_loader)
    avg_sparse_loss = sparse_loss / len(eval_loader) 
    return avg_eval_loss,  avg_reg_loss, avg_cls_loss, avg_sparse_loss,  average_non_zero_elements


def evaluate_end2end_model(model, eval_loader, criterion, device):
    model.eval()  
    total_eval_loss = 0
    steered_texts = []
    unsteered_texts = []
    original_texts = []
    sim_steered_list = []
    sim_unsteered_list = []
    with torch.no_grad(): 
        for input_ids, attention_masks, _, prompt, original_text in tqdm(eval_loader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            steered_text, unsteered_text = model(input_ids, attention_masks, prompt)
            steered_texts.extend(steered_text)
            unsteered_texts.extend(unsteered_text)
            original_texts.extend(original_text)
            loss , sim_steered, sim_unsteered = criterion(original_text, steered_text, unsteered_text)
            sim_steered_list.append(sim_steered)
            sim_unsteered_list.append(sim_unsteered)
            total_eval_loss += loss.item()  
            
    avg_sim_steered = torch.cat(sim_steered_list).mean().item()
    avg_sim_unsteered = torch.cat(sim_steered_list).mean().item()
    avg_eval_loss = total_eval_loss / len(eval_loader) 
    
    return {
        'loss': avg_eval_loss,
        'avg_sim_steered': avg_sim_steered,
        'avg_sim_unsteered': avg_sim_unsteered,
        'steered_text': steered_texts,
        'unsteered_text': unsteered_texts,
        'original_text': original_texts
        }


def build_optimizer_and_scheduler(model, lr, lr_drop, weight_decay = 0.01):
    no_decay = [
        "bias",
        "layer_norm",
        "LayerNorm",
    ]  # no decay for parameters of layer norm and bias
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    scheduler = StepLR(optimizer, lr_drop, gamma = 0.8)
    return optimizer, scheduler

def filter_checkpoints(model_dir, suffix="pt", max_to_keep=1):
    model_paths = glob.glob(os.path.join(model_dir, "*.{}".format(suffix)))
    if len(model_paths) > max_to_keep:
        model_file_dict = dict()
        suffix_len = len(suffix) + 1
        for model_path in model_paths:
            step = int(os.path.basename(model_path).split("_")[-1][0:-suffix_len])
            model_file_dict[step] = model_path
        sorted_tuples = sorted(model_file_dict.items())
        unused_tuples = sorted_tuples[0:-max_to_keep]
        for _, model_path in unused_tuples:
            os.remove(model_path)


def train_model(
    model, 
    train_loader, 
    eval_loader,
    criterion, 
    optimizer, 
    scheduler, 
    device, 
    saving_dir,
    num_epochs=10 , 
    max_unfreeze_layers=4, 
    gradual_unfreeze = True,
    score_writer = None
):

    best_metric = float('inf')
    train_losses = []
    eval_losses = []
    eval_reg_losses = [] 
    eval_cls_losses = [] 
    eval_sparse_losses = [] 
    eval_avg_non_zero_counts = []
    scaler = GradScaler()
    save_every = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        reg_loss = 0
        cls_loss = 0
        sparse_loss = 0

       
        if gradual_unfreeze:
            if epoch == 0:
                layers_to_unfreeze = 0  
            else:
                layers_to_unfreeze = min(epoch, max_unfreeze_layers)
            model.unfreeze_layers(layers_to_unfreeze)

        for input_ids, attention_masks, scores in tqdm(train_loader, desc="Training"):
    
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            scores = scores.to(device)
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                outputs = model(input_ids, attention_masks)
                if isinstance(outputs, tuple):
                    class_preds, reg_preds = outputs
                    loss_dict = criterion(class_preds, reg_preds, scores)
                else:
                    loss_dict = criterion(outputs, scores)
                loss = loss_dict['total_loss']

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            reg_loss +=loss_dict['reg_loss'].item()
            cls_loss += loss_dict.get('cls_loss', torch.tensor(0.0)).item()
            sparse_loss += loss_dict.get('sparse_loss', torch.tensor(0.0)).item()

        avg_loss = total_loss / len(train_loader)
        avg_reg_loss = reg_loss / len(train_loader)
        avg_cls_loss = cls_loss / len(train_loader)
        avg_sparse_loss = sparse_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.10f}")
        scheduler.step()

        
        (
            avg_eval_loss, 
            eval_reg_loss, 
            eval_cls_loss, 
            eval_sparse_loss, 
            avg_non_zero_counts
        ) = evaluate_model(model, eval_loader, criterion, device)
        if score_writer is not None:
            score_writer.write(f'Epoch {epoch+1}\n')
            score_writer.write(f'train total loss: {avg_loss:.10f}, train reg_loss: {avg_reg_loss:.10f}, train cls_loss: {avg_cls_loss:.10f}, train sparse_loss: {avg_sparse_loss:.10f}\n')
            score_writer.write(f'eval total loss: {avg_eval_loss:.10f}, eval reg_loss: {eval_reg_loss:.10f}, eval cls_loss: {eval_cls_loss:.10f}, eval sparse_loss: {eval_sparse_loss:.10f}\n')
            score_writer.write(f'average non-zero elemnts in eval {avg_non_zero_counts}\n\n')

            score_writer.flush()

        eval_losses.append(avg_eval_loss)
        eval_reg_losses.append(eval_reg_loss)
        eval_cls_losses.append(eval_cls_loss)
        eval_sparse_losses.append(eval_sparse_loss)
        eval_avg_non_zero_counts.append(avg_non_zero_counts)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Evaluation Loss: {avg_eval_loss:.10f}")


        saving_path = os.path.join(saving_dir, f"model_epoch_{epoch + 1}.pt")
        if avg_eval_loss < best_metric:
            best_metric = avg_eval_loss
            torch.save(model.state_dict(), saving_path)
            filter_checkpoints(saving_dir)
        
        if (epoch + 1) % save_every == 0:
            loss_df = pd.DataFrame({
                'train_loss': train_losses, 
                'eval_loss': eval_losses,
                'eval_reg_loss': eval_reg_losses,
                'eval_cls_loss': eval_cls_losses,
                'eval_sparse_loss': eval_sparse_losses,
                'eval_avg_nonzero_elements': eval_avg_non_zero_counts
                })
            loss_df.to_csv(os.path.join(saving_dir, 'losses.csv'), index = False) 
        
    
  

    loss_df = pd.DataFrame({
                'train_loss': train_losses, 
                'eval_loss': eval_losses,
                'eval_reg_loss': eval_reg_losses,
                'eval_cls_loss': eval_cls_losses,
                'eval_sparse_loss': eval_sparse_losses,
                'eval_avg_nonzero_elements': eval_avg_non_zero_counts
                })
    loss_df.to_csv(os.path.join(saving_dir, 'losses.csv'), index = False) 

    return train_losses, eval_losses



def train_end2endmodel(
    model, 
    train_loader, 
    eval_loader,
    criterion, 
    optimizer, 
    scheduler, 
    device, 
    saving_dir,
    num_epochs=10 , 
    max_unfreeze_layers=4, 
    gradual_unfreeze = True,
    score_writer = None
):

    best_metric = float('inf')
    train_losses = []
    eval_losses = []
    avg_sim_steered_list = []
    avg_sim_unsteered_list = []
    save_every = 1
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        # steered_texts = []
        # unsteered_texts = []
        steered_sims = []
        unsteered_sims = []
        # original_texts = []


       
        if gradual_unfreeze:
            if epoch == 0:
                layers_to_unfreeze = 0  
            else:
                layers_to_unfreeze = min(epoch, max_unfreeze_layers)
            model.unfreeze_layers(layers_to_unfreeze)

        for input_ids, attention_masks, _, prompt, original_text in tqdm(train_loader, desc=f"Training epoch {epoch+1}"):
    
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            # prompt = prompt.to(device)
            optimizer.zero_grad()
            
            steered_text, unsteered_text = model(input_ids, attention_masks, prompt)
            
            loss , sim_steered, sim_unsteered= criterion(original_text, steered_text, unsteered_text)
            
            # steered_texts.extend(steered_text)
            # unsteered_texts.extend(unsteered_text)
            # original_texts.extend(original_texts)
            steered_sims.append(sim_steered)
            unsteered_sims.append(sim_unsteered)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)

            optimizer.step()
            total_loss += loss.item()

  

        avg_sim_steered = torch.cat(steered_sims).mean().item()
        avg_sim_unsteered = torch.cat(unsteered_sims).mean().item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.10f}")
        scheduler.step()

        
        result_dict = evaluate_end2end_model(model, eval_loader, criterion, device)
        avg_eval_loss = result_dict['loss']
        avg_eval_sim_steered = result_dict['avg_sim_steered']
        avg_eval_sim_unsteered = result_dict['avg_sim_unsteered']
        avg_sim_steered_list.append(avg_eval_sim_steered)
        avg_sim_unsteered_list.append(avg_eval_sim_unsteered)
        if score_writer is not None:
            score_writer.write(f'Epoch {epoch+1}\n')
            score_writer.write(f'train loss: {avg_loss:.10f}, avg sim steered: {avg_sim_steered}, avg sim unsteered: {avg_sim_unsteered}\n')
            score_writer.write(f'eval loss: {avg_eval_loss:.10f}, avg sim steered: {avg_eval_sim_steered}, avg sim unsteered: {avg_eval_sim_unsteered}\n')

            score_writer.flush()

        eval_losses.append(avg_eval_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Evaluation Loss: {avg_eval_loss:.10f}")


        saving_path = os.path.join(saving_dir, f"model_epoch_{epoch + 1}.pt")
        if avg_eval_loss < best_metric:
            best_metric = avg_eval_loss
            torch.save(model.state_dict(), saving_path)
            filter_checkpoints(saving_dir)
        
        if (epoch + 1) % save_every == 0 or (epoch + 1) == num_epochs:
            loss_df = pd.DataFrame({
                'train_loss': train_losses, 
                'eval_loss': eval_losses
                })
            sims_df = pd.DataFrame({
                'avg_sim_steered': avg_sim_steered_list,
                'avg_sim_unsteered': avg_sim_unsteered_list
            })
            texts_df = pd.DataFrame({
                'steered_text': result_dict['steered_text'],
                'unsteered_text': result_dict['unsteered_text'],
                'original_text': result_dict['original_text']
            })
            loss_df.to_csv(os.path.join(saving_dir, 'losses.csv'), index = False) 
            sims_df.to_csv(os.path.join(saving_dir, 'sims.csv'), index = False) 
            texts_df.to_csv(os.path.join(saving_dir, 'texts.csv'), index = False) 
        
    
         

    return train_losses, eval_losses











