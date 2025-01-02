import torch
import torch.nn as nn
import torch.nn.functional as F


class FASG_Loss(nn.Module):
    def __init__(self, tokenizer, encoder):
        super(FASG_Loss, self).__init__()
        # self.sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
        # self.device = "cuda"
        self.tokenizer = tokenizer
        self.model = encoder
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def compute_embeddings(self, texts):
        encoded_input = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        encoded_input = {k: v.cuda() for k, v in encoded_input.items()}
        model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return F.normalize(sentence_embeddings, p=2, dim=1)  
    
    def forward(self, original_texts, steered_texts, unsteered_texts):
        original_embeddings = self.compute_embeddings(original_texts)
        steered_embeddings = self.compute_embeddings(steered_texts)
        unsteered_embeddings = self.compute_embeddings(unsteered_texts)

        sim_steered = (original_embeddings * steered_embeddings).sum(dim=1)
        sim_unsteered = (original_embeddings * unsteered_embeddings).sum(dim=1)

        loss = -torch.log(torch.exp(sim_steered) / (torch.exp(sim_steered) + torch.exp(sim_unsteered))).mean()
        return loss, sim_steered, sim_unsteered


    # def forward(self, reference_text,steered_text,baseline_text,temp= 0.6):
    #     tokenized_inputs = self.sentence_transformer.tokenize(
    #         [reference_text, steered_text, baseline_text])
    #     # move all tensors to gpu
    #     for key in tokenized_inputs.keys():
    #         tokenized_inputs[key] = tokenized_inputs[key].to(self.device)
    #     embeddings = self.sentence_transformer(tokenized_inputs)['sentence_embedding']
    #     embeddings = embeddings * temp
        
    #     # Compute cosine similarities
    #     sim_1 = st_util.cos_sim(embeddings[0], embeddings[1])
    #     sim_2 = st_util.cos_sim(embeddings[0], embeddings[2])

    #     # Compute softmax triplet loss
    #     loss = -torch.log(torch.exp(sim_1) / (torch.exp(sim_1) + torch.exp(sim_2)))
    #     return loss

class SimilarityLoss(nn.Module):
    def __init__(self, tokenizer, encoder):
        super(SimilarityLoss, self).__init__()
        self.tokenizer = tokenizer
        self.model = encoder

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def compute_embeddings(self, texts):
        encoded_input = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        encoded_input = {k: v.cuda() for k, v in encoded_input.items()}
        model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return F.normalize(sentence_embeddings, p=2, dim=1)  

    def forward(self, original_texts, steered_texts, unsteered_texts):
        original_embeddings = self.compute_embeddings(original_texts)
        steered_embeddings = self.compute_embeddings(steered_texts)
        unsteered_embeddings = self.compute_embeddings(unsteered_texts)
        sim_steered = (original_embeddings * steered_embeddings).sum(dim=1)
        sim_unsteered = (original_embeddings * unsteered_embeddings).sum(dim=1)

        margin = 0.2
        loss = torch.clamp(sim_unsteered + margin - sim_steered, min=0.0).mean()
        return loss, sim_steered, sim_unsteered

class tripletLoss(nn.Module):
    def __init__(self, tokenizer, encoder):
        super(tripletLoss, self).__init__()
        self.tokenizer = tokenizer
        self.model = encoder
        self.triplet_loss = nn.TripletMarginLoss()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def compute_embeddings(self, texts):
        encoded_input = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        encoded_input = {k: v.cuda() for k, v in encoded_input.items()}
        model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return F.normalize(sentence_embeddings, p=2, dim=1)  

    def forward(self, original_texts, steered_texts, unsteered_texts):
        original_embeddings = self.compute_embeddings(original_texts)
        steered_embeddings = self.compute_embeddings(steered_texts)
        unsteered_embeddings = self.compute_embeddings(unsteered_texts)
        sim_steered = (original_embeddings * steered_embeddings).sum(dim=1)
        sim_unsteered = (original_embeddings * unsteered_embeddings).sum(dim=1)
        loss = self.triplet_loss(original_embeddings, steered_embeddings, unsteered_embeddings)
        
        return loss, sim_steered, sim_unsteered

class MSEWithL1Loss(nn.Module):
    def __init__(self, configs):
        super(MSEWithL1Loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.lambda_sparsity = configs.sparsity_weight

    def forward(self, predictions, targets):
        mse_loss = self.mse_loss(predictions, targets)
        l1_norm = torch.sum(torch.abs(predictions))
        sparse_loss = self.lambda_sparsity * l1_norm
        loss = mse_loss + sparse_loss
        
        return {
            'total_loss': loss, 
            'reg_loss': mse_loss, 
            'sparse_loss': sparse_loss }

class MaskedMSEWithL1Loss(nn.Module):
    def __init__(self, configs):
        super(MaskedMSEWithL1Loss, self).__init__()
        self.reg_weight = configs.reg_weight
        self.sparsity_weight = configs.sparsity_weight

    def forward(self, reg_preds, targets):
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets)

        mask = torch.ne(targets, 0).float()
        masked_regression_loss = F.mse_loss(reg_preds * mask, targets * mask, reduction='mean')
        sparsity_reg = torch.mean(torch.abs(reg_preds))
        final_reg_loss = self.reg_weight * masked_regression_loss
        final_sparsity_loss = self.sparsity_weight * sparsity_reg
       
        total_loss = final_reg_loss + final_sparsity_loss
        
        return {
            'total_loss': total_loss,
            'reg_loss': final_reg_loss,
            'sparsity_loss': final_sparsity_loss
        }

class CombinedLoss(nn.Module):
    def __init__(self, configs):
        super(CombinedLoss, self).__init__()
        self.reg_weight = configs.reg_weight
        self.cls_weight = configs.cls_weight
        self.sparsity_weight = configs.sparsity_weight
        self.pos_weight = torch.tensor([90]).cuda()

    def forward(self, class_preds, reg_preds, targets):
        binary_targets = (targets != 0).float()
        
        classification_loss = F.binary_cross_entropy_with_logits(
            class_preds, binary_targets, pos_weight=self.pos_weight
        )
        
        mask = binary_targets
        masked_regression_loss = F.mse_loss(reg_preds * mask, targets * mask, reduction='mean')
        
        sparsity_reg = torch.mean(torch.abs(reg_preds))
        
        final_cls_loss = self.cls_weight * classification_loss
        final_reg_loss = self.reg_weight * masked_regression_loss
        final_sparsity_loss = self.sparsity_weight * sparsity_reg
        
        total_loss = final_cls_loss + final_reg_loss + final_sparsity_loss
        
        return {
            'total_loss': total_loss,
            'cls_loss': final_cls_loss,
            'reg_loss': final_reg_loss,
            'sparsity_loss': final_sparsity_loss
        }

class FocalLoss(nn.Module):
    def __init__(self, configs):
        super(FocalLoss, self).__init__()
        self.alpha = configs.focal_alpha
        self.gamma = configs.focal_gamma
        self.huber_delta = configs.huber_delta
        self.class_weight = configs.cls_weight
        self.reg_weight = configs.reg_weight
        self.sparsity_weight = configs.sparsity_weight

    def focal_loss(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean()

    # def focal_loss(self, inputs, targets):
    #     BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    #     pt = torch.exp(-BCE_loss)
    #     F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
    #     return F_loss.mean()

    def forward(self, class_preds, reg_preds, targets):
        binary_targets = (targets != 0).float()
   
        classification_loss = self.focal_loss(class_preds, binary_targets)
        
    
        mask = binary_targets
        huber_loss = F.huber_loss(reg_preds * mask, targets * mask, 
                                      delta=self.huber_delta, reduction='mean')
        

        sparsity_reg = torch.mean(torch.abs(reg_preds))
        final_cls_loss =  self.class_weight * classification_loss
        final_reg_loss = self.reg_weight * huber_loss
        final_sparse_loss = self.sparsity_weight * sparsity_reg
        
        total_loss = final_cls_loss + final_reg_loss + final_sparse_loss
        
        return {
            'total_loss': total_loss, 
            'cls_loss': final_cls_loss,
            'reg_loss': final_reg_loss,
            'sparse_loss': final_sparse_loss}

