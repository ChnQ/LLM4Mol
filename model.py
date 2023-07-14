"""FGWApproximator class.""" 
import torch
import random
import numpy as np
from tqdm import tqdm, trange
from torch_geometric.nn import GCNConv, GINConv
import MLP

import torch
import torch.nn as nn
import numpy as np
from transformers import PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
from utils import init_random_state


class BertClassifier(PreTrainedModel):
    def __init__(self, model, n_labels, dropout=0.0, seed=0, cla_bias=True, feat_shrink=''):
        super().__init__(model.config)
        self.bert_encoder = model
        self.dropout = nn.Dropout(dropout)
        self.feat_shrink = feat_shrink
        hidden_dim = model.config.hidden_size

        # todo 确认 label_smoothing
        self.loss_func = nn.CrossEntropyLoss(
            label_smoothing=0.3, reduction='mean')

        if feat_shrink:
            self.feat_shrink_layer = nn.Linear(
                model.config.hidden_size, int(feat_shrink), bias=cla_bias)
            hidden_dim = int(feat_shrink)
        self.classifier = nn.Linear(hidden_dim, n_labels, bias=cla_bias)
        # init_random_state(seed)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                return_dict=None,
                preds=None):

        outputs = self.bert_encoder(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    return_dict=return_dict,
                                    output_hidden_states=True)
        # outputs[0]=last hidden state
        emb = self.dropout(outputs['hidden_states'][-1])
        # Use CLS Emb as sentence emb.
        cls_token_emb = emb.permute(1, 0, 2)[0]
        if self.feat_shrink:
            cls_token_emb = self.feat_shrink_layer(cls_token_emb)
        logits = self.classifier(cls_token_emb)

        if labels.shape[-1] == 1:
            labels = labels.squeeze()
        loss = self.loss_func(logits, labels)

        return TokenClassifierOutput(loss=loss, logits=logits)
    
    # def save_feats(self, iter, feat_save_path='/data00/qc/mol_record/test.npy'):
    #     with torch.no_grad():

    #         print('testing!!!!!!!!!!!!!!')


# 修改 MLP
class BertClassifierV2(PreTrainedModel):
    def __init__(self, model, n_labels, dropout=0.0, seed=0, cla_bias=True, feat_shrink=''):
        super().__init__(model.config)
        self.bert_encoder = model
        self.dropout = nn.Dropout(dropout)
        self.feat_shrink = feat_shrink
        hidden_dim = model.config.hidden_size
        print('hidden_size', model.config.hidden_size)

        # todo 确认 label_smoothing
        self.loss_func = nn.CrossEntropyLoss()

        if feat_shrink:
            self.feat_shrink_layer = nn.Linear(
                model.config.hidden_size, int(feat_shrink), bias=cla_bias)
            hidden_dim = int(feat_shrink)
    
        self.mlp_classifier = MLP.MLP(
            input_dim=hidden_dim,
            output_dim=n_labels,
            hidden_dim=256,  # todo：需要调
            num_hidden=2,
            output_activation='linear',
            dtype=torch.float32
        )
        init_random_state(seed)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                return_dict=None,
                preds=None):

        outputs = self.bert_encoder(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    return_dict=return_dict,
                                    output_hidden_states=True)
        # outputs[0]=last hidden state
        emb = self.dropout(outputs['hidden_states'][-1])
        # Use CLS Emb as sentence emb.
        cls_token_emb = emb.permute(1, 0, 2)[0]
        if self.feat_shrink:
            cls_token_emb = self.feat_shrink_layer(cls_token_emb)
        logits = self.mlp_classifier(cls_token_emb)

        if labels.shape[-1] == 1:
            labels = labels.squeeze()
        loss = self.loss_func(logits, labels)

        return TokenClassifierOutput(loss=loss, logits=logits)


class MorganClassifier(torch.nn.Module):
    def __init__(self, input_dim, num_class, mlp_params, device='cpu'):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(MorganClassifier, self).__init__()
        

        self.lm = None
        self.mlp_classifier = MLP.MLP(
            # input_dim=num_temp_graphs,
            input_dim=input_dim,
            output_dim=num_class,
            hidden_dim=mlp_params['hidden_dim'],
            num_hidden=mlp_params['num_hidden_layers'],
            output_activation=mlp_params['output_activation'],
            device=device,
        )
        # self.mlp_classifier = nn.Linear(2048, num_class, bias=True, dtype=torch.float64)

    
    def get_optimizer(self, lr=0.01):
        
        return torch.optim.Adam(
            params=self.mlp_classifier.parameters(),
            lr=lr,
            betas=[0.9, 0.99]
        )

    def set_model_to_train(self):
        self.mlp_classifier.train()

    def set_model_to_eval(self):
        self.mlp_classifier.eval()
    
    def forward(self, data):

        # 1. Classify by MLP
        
        logits = self.mlp_classifier(data)
        return logits
    

class LLM4Mol(torch.nn.Module):
    def __init__(self, num_class, lm_params, mlp_params, device='cpu'):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(LLM4Mol, self).__init__()
        

        self.lm = None
        self.mlp_classifier = MLP.MLP(
            # input_dim=num_temp_graphs,
            input_dim=16+16,
            output_dim=num_class,
            hidden_dim=mlp_params['hidden_dim'],
            num_hidden=mlp_params['num_hidden_layers'],
            output_activation=mlp_params['output_activation'],
            device=device,
        )
    
    def get_optimizer(self, lr=0.01):
        self.param_list = list(self.lm.parameters()) + list(self.mlp_classifier.parameters())
        
        return torch.optim.Adam(
            params=self.param_list,
            lr=lr,
            betas=[0.9, 0.99]
        )

    def set_model_to_train(self):
        self.lm.train()
        self.mlp_classifier.train()

    def set_model_to_eval(self):
        self.lm.eval()
        self.mlp_classifier.eval()
    
    def forward(self, data):
        # 1. Pass LM
        features = self.lm(data)
        
        # 2. Classify by MLP
        
        logits = self.mlp_classifier(features)

        return logits