import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class SynDataset(Dataset):
    def __init__(self, dict_list, tokenizer, n_features, prompt_length,
                 max_length=512):

        self.dict_list = dict_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_features = n_features
        self.prompt_length = prompt_length

    def __len__(self):
        return len(self.dict_list)

    def process_dict(self, dict):
        text = dict['text'].encode('utf-8', errors='ignore').decode('utf-8')
        feature_ids = dict['feature_ids']
        scores = dict['scores']
        output_list = [0.0] * self.n_features
        for feature_id, score in zip(feature_ids, scores):
            if feature_id < self.n_features:  
                output_list[feature_id] = score
        
        return text, output_list

    

    def __getitem__(self, idx):
        dict = self.dict_list[idx]
        text, label = self.process_dict(dict)
        words = text.split()
        prompt = ' '.join(words[:self.prompt_length])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return  (
            encoding['input_ids'].flatten(), 
            encoding['attention_mask'].flatten(), 
            torch.tensor(label,dtype=torch.float), 
            prompt,
            text
        )
        

