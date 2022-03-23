import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

SIM_DIM = 384
NLI_DIM = 768
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SIM_SCALAR = 50

class AgreemNet(nn.Module):
    def __init__(self, hdim_1=1024, hdim_2=512, dropout=0.7, num_classes=3, num_heads=5):
        super(AgreemNet, self).__init__()
        self.dropout = dropout
        self.num_classes = num_classes
        self.attention_heads = [torch.nn.MultiheadAttention(embed_dim=SIM_DIM, vdim=NLI_DIM, num_heads=1) for _ in range(num_heads)]
        self.reduce_head = torch.nn.Linear(NLI_DIM, SIM_DIM)
        
        self.fc1 = torch.nn.Linear((SIM_DIM * (num_heads + 1)) + num_heads, hdim_1)
        self.fc2 = torch.nn.Linear(hdim_1, hdim_1)
        self.fc3 = torch.nn.Linear(hdim_1, hdim_2)
        self.fc4 = torch.nn.Linear(hdim_2, num_classes)

    def forward(self, H_sims, H_nlis, B_sims, B_nlis):
        batch_size = B_sims.shape[0]
        sent_len = B_sims.shape[1]
        
        # Attention layer
        attn_outs = [attention(
            H_sims.view(1, batch_size, SIM_DIM),
            B_sims.view(sent_len, batch_size, SIM_DIM),
            B_nlis.view(sent_len, batch_size, NLI_DIM),
        )[0] for attention in self.attention_heads]
        attn_outs = torch.stack(attn_outs).squeeze() # convert list to tensor
        flattened_attn_outs = torch.cat(attn_outs.split(1), dim=-1).squeeze()

        # Linear transform head_nli to be same dimension as attn_out
        reduced_head = self.reduce_head(H_nlis)
        cosines = F.cosine_similarity(reduced_head, attn_outs, dim=2).transpose(0, 1)
        xx = torch.cat((flattened_attn_outs, reduced_head, cosines), dim=1)
        
        xx = self.fc1(xx)
        xx = F.dropout(xx, p=self.dropout)
        xx = F.relu(xx)
        xx = self.fc2(xx)
        xx = F.dropout(xx, p=self.dropout)
        xx = F.relu(xx)
        xx = self.fc3(xx)
        xx = F.dropout(xx, p=self.dropout)
        xx = F.relu(xx)
        logits = self.fc4(xx)
        
        return logits

class TopKNet(nn.Module):
    def __init__(self, kk=5, hdim_1=1024, hdim_2=512, dropout=0.7, num_classes=3):
        super(TopKNet, self).__init__()
        self.kk = kk
        self.num_classes = num_classes
        self.dropout = dropout
        self.fc1 = torch.nn.Linear((kk + 1) * NLI_DIM, hdim_1)
        self.fc2 = torch.nn.Linear(hdim_1, hdim_1)
        self.fc3 = torch.nn.Linear(hdim_1, hdim_1)
        self.fc4 = torch.nn.Linear(hdim_1, hdim_2)
        self.fc5 = torch.nn.Linear(hdim_2, num_classes)

    def forward(self, sim_stance_emb, nli_stance_emb, sim_body_emb, nli_body_emb):
        '''
        TODO
        '''
        batch_size = sim_stance_emb.shape[0]
        assert batch_size == nli_stance_emb.shape[0]
        assert batch_size == sim_body_emb.shape[0]
        assert batch_size == nli_body_emb.shape[0]
        
        # Select the k nli embeddings with highest similarity
        sims = torch.bmm(
            sim_stance_emb.unsqueeze(1),
            torch.transpose(sim_body_emb,1,2)
        ).squeeze()

        top_k = torch.topk(sims,k=self.kk,dim=1)

        nli_body_emb_top_k = torch.transpose(
            nli_body_emb[np.arange(batch_size),top_k.indices.T],
        dim0=0,dim1=1)

        nli_body_emb_top_k_flat = nli_body_emb_top_k.flatten(start_dim=1)
        xx = torch.hstack([nli_stance_emb, nli_body_emb_top_k_flat])
        
        xx = self.fc1(xx)
        xx = F.dropout(xx, p=self.dropout)
        xx = F.relu(xx)
        xx = self.fc2(xx)
        xx = F.dropout(xx, p=self.dropout)
        xx = F.relu(xx)
        xx = self.fc3(xx)
        xx = F.dropout(xx, p=self.dropout)
        xx = F.relu(xx)
        xx = self.fc4(xx)
        xx = F.dropout(xx, p=self.dropout)
        xx = F.relu(xx)
        logits = self.fc5(xx)

        return logits

class RelatedNet(nn.Module):
    def __init__(self, kk=5, hdim_1=1024, hdim_2=512, num_classes=2):
        super(RelatedNet, self).__init__()
        self.kk = kk
        self.num_classes = num_classes
        self.fc1 = torch.nn.Linear((kk + 1) * SIM_DIM, hdim_1)
        self.fc2 = torch.nn.Linear(hdim_1, hdim_1)
        self.fc3 = torch.nn.Linear(hdim_1, hdim_1)
        self.fc4 = torch.nn.Linear(hdim_1, hdim_2)
        self.fc5 = torch.nn.Linear(hdim_2, num_classes)

    def forward(self, sim_stance_emb, nli_stance_emb, sim_body_emb, nli_body_emb):
        '''
        TODO
        '''
        batch_size = sim_stance_emb.shape[0]
        assert batch_size == nli_stance_emb.shape[0]
        assert batch_size == sim_body_emb.shape[0]
        assert batch_size == nli_body_emb.shape[0]
        
        # Select the k nli embeddings with highest similarity
        sims = torch.bmm(
            sim_stance_emb.unsqueeze(1),
            torch.transpose(sim_body_emb,1,2)
        ).squeeze()

        top_k = torch.topk(sims,k=self.kk,dim=1)

        sim_body_emb_top_k = torch.transpose(
            sim_body_emb[np.arange(batch_size),top_k.indices.T],
        dim0=0,dim1=1)

        sim_body_emb_top_k_flat = sim_body_emb_top_k.flatten(start_dim=1)
        
        xx = torch.hstack([sim_stance_emb, sim_body_emb_top_k_flat])
        
        xx = self.fc1(xx)
        xx = F.dropout(xx, p=0.5)
        xx = F.relu(xx)
        xx = self.fc2(xx)
        xx = F.dropout(xx, p=0.5)
        xx = F.relu(xx)
        xx = self.fc3(xx)
        xx = F.dropout(xx, p=0.5)
        xx = F.relu(xx)
        xx = self.fc4(xx)
        xx = F.dropout(xx, p=0.5)
        xx = F.relu(xx)
        logits = self.fc5(xx)

        return logits
