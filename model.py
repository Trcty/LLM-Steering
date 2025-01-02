import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from sae_lens import SAE, HookedSAETransformer
from huggingface_hub import login


class CustomTransformerWithHeads(nn.Module):
    def __init__(self, base_model, n_features=768, loss_type = 'reg_only'):
        super(CustomTransformerWithHeads, self).__init__()
        self.base_model = base_model
        self.loss_type = loss_type
        base_model_hidden_size = self.base_model.config.hidden_size

        self.classification_head = nn.Linear(base_model_hidden_size, n_features)
        self.regression_head = nn.Linear(base_model_hidden_size, n_features)

        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze_layers(self, num_layers):
        if num_layers > 0:
            for layer in self.base_model.encoder.layer[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        cls_embeddings = last_hidden_states[:, 0, :]  # [CLS] token embeddings
        regression_output = self.regression_head(cls_embeddings) 

        if self.loss_type != 'reg_only':
            classification_output = self.classification_head(cls_embeddings)
            return classification_output, regression_output
        else:
            return regression_output





class End2EndModel(nn.Module):
    def __init__(self, base_model, max_new_token, tmp ,sae_model, release, hook_id):
        super(End2EndModel, self).__init__()
        self.base_model = base_model
        base_model_hidden_size = self.base_model.config.hidden_size
        self.max_new_token = max_new_token
        self.tmp = tmp

        login('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')

        for param in self.base_model.parameters():
            param.requires_grad = False

        self.sae_model = HookedSAETransformer.from_pretrained(sae_model)
        self.sae, self.cfg_dict, self.sparsity = SAE.from_pretrained(
            release=release,
            sae_id=hook_id
        )
        self.regression_head = nn.Linear(base_model_hidden_size, self.cfg_dict['d_sae'])
        self.freeze_params()

    def freeze_params(self):
        for param in self.base_model.parameters():
            param.requires_grad = False
        for param in self.sae_model.parameters():
            param.requires_grad = False
        for param in self.sae.parameters():
            param.requires_grad = False

    def unfreeze_layers(self, num_layers):
        if num_layers > 0:
            for layer in self.base_model.encoder.layer[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def steering(
        self, activations, hook, steering_strength=1.0, steering_vector=None
    ):
    
        return activations + (steering_strength * steering_vector).unsqueeze(1)
    
    def steering_features(self, value, hook, steering_vector):
        encoded_activation = self.sae.encode(value)
        steering_vector = steering_vector.unsqueeze(1)
        steered_vector = steering_vector*encoded_activation 
        decoded_vector = self.sae.decode(steered_vector)
        return decoded_vector
    
    def generate_with_steering(
        self, 
        model,
        sae,
        prompt,
        steering_vector,
        steering_strength=1.0,
    ):
        input_ids = model.to_tokens(prompt)
       
        steering_hook = partial(
            self.steering_features,
            steering_vector=steering_vector
        )
        
        with model.hooks(fwd_hooks=[(sae.cfg.hook_name, steering_hook)]):
            output = model.generate(
                input_ids,
                max_new_tokens=self.max_new_token,
                temperature=self.tmp,
                top_p=0.9,
                stop_at_eos= True,
                prepend_bos=sae.cfg.prepend_bos,
                verbose = False,
                # return_type = 'str'
            )
            # print(output)
 
          

        return model.tokenizer.batch_decode(output,skip_special_tokens=True)
    
    def generate_without_steering(
        self, 
        model,
        sae,
        prompt
    ):
        input_ids = model.to_tokens(prompt)

        output = model.generate(
            input_ids,
            max_new_tokens=self.max_new_token,
            temperature=self.tmp,
            top_p=0.9,
            stop_at_eos= True,
            prepend_bos=sae.cfg.prepend_bos,
            verbose = False,
            # return_type = 'str'
        )
        # print(output)
       
        return model.tokenizer.batch_decode(output,skip_special_tokens=True)

    def forward(self, input_ids, attention_mask, initial_prompt):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        cls_embeddings = last_hidden_states[:, 0, :]  
        regression_output = self.regression_head(cls_embeddings)

        steered_text = self.generate_with_steering(
            self.sae_model,
            self.sae,
            initial_prompt,
            regression_output
        )
        unsteered_text = self.generate_without_steering(
            self.sae_model,
            self.sae,
            initial_prompt
        )
        # print(f'steered text: {steered_text}\n')
        # print(f'unsteered text: {unsteered_text}\n')





        return steered_text, unsteered_text
