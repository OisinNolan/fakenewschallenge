import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from functools import reduce

SIM_DIM = 384
NLI_DIM = 768
NUM_CLASSES = 4 # TODO
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SIM_SCALAR = 50
TOP_K_SENT = 5

FC_DIMS = [(NLI_DIM * 6) + 5, 1024, 512, NUM_CLASSES]
HIDDEN_DIMS = [1024, 512]

class AgreemNet(nn.Module):
    def __init__(self):
        super(AgreemNet, self).__init__()
        self.sim_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.nli_encoder = SentenceTransformer('sentence-transformers/nli-distilroberta-base-v2')
        #self.attention = torch.nn.MultiheadAttention(embed_dim=SIM_DIM, vdim=NLI_DIM, num_heads=1)
        #self.reduce_head = torch.nn.Linear(NLI_DIM, SIM_DIM)
        
        self.fc1 = torch.nn.Linear(FC_DIMS[0], FC_DIMS[1])
        self.fc2 = torch.nn.Linear(FC_DIMS[1], FC_DIMS[2])
        self.fc3 = torch.nn.Linear(FC_DIMS[2], FC_DIMS[3])

    def encode_(self, encoder, embed_dim, B_flat):
        '''
        Encodes sentence strings but skips padding sentences, encoding them as zeros.
        '''

        embeddings = np.array(list(map(
            lambda sent: encoder.encode(sent) if sent != '[PAD]' else np.zeros((embed_dim), dtype=np.float32), 
            B_flat
        )))

        return torch.from_numpy(embeddings)


    def forward(self, H, B):
        '''
        H: List of headline strings
        B: List of padded lists of body sentence strings
        '''
        
        batch_size = len(B)
        # Only include first 5 sentences
        sent_len = len(B[0])
        # flatten list of lists of lists -> list of lists
        B_flat = reduce(lambda x, y: x+y, B)

        # Compute embeddings
        head_sim = torch.from_numpy(self.sim_encoder.encode(H)).to(DEVICE) * SIM_SCALAR
        body_sims = self.encode_(self.sim_encoder, SIM_DIM, B_flat).view(batch_size, sent_len, SIM_DIM).to(DEVICE) * SIM_SCALAR
        head_nli = torch.from_numpy(self.nli_encoder.encode(H)).to(DEVICE)
        body_nlis = self.encode_(self.nli_encoder, NLI_DIM, B_flat).view(batch_size, sent_len, NLI_DIM).to(DEVICE)

        # flatten list of lists of lists -> list of lists
        # B_flat = np.array(B)[:,:sent_len].flatten()
        

        # Attention layer
        # attn_out, attn_weights = self.attention(
        #     head_sim.view(1, batch_size, SIM_DIM),
        #     body_sims.view(sent_len, batch_size, SIM_DIM),
        #     body_nlis.view(sent_len, batch_size, NLI_DIM)
        # )

        # Linear transform head_nli to be same dimension as attn_out
        # reduced_head = self.reduce_head(head_nli)
        # xx = torch.cat((attn_out.squeeze(), reduced_head, F.cosine_similarity(attn_out.squeeze(), reduced_head).view(batch_size, 1)), dim=1)

        cosines = torch.zeros((batch_size, sent_len)).to(DEVICE)
        for i in range(batch_size):
            cosines[i] = F.cosine_similarity(head_nli[i].unsqueeze(0).repeat(5, 1), body_nlis[i])

        xx = torch.cat((head_nli, body_nlis.flatten(1), cosines), dim=1)
        
        xx = self.fc1(xx)
        xx = F.relu(xx)
        xx = self.fc2(xx)
        xx = F.relu(xx)
        xx = self.fc3(xx)

        output = xx#F.softmax(xx, dim=1)
        return output

class AgreemFlat(nn.Module):
    def __init__(self, kk=5):
        super(AgreemFlat, self).__init__()
        self.kk = kk
        self.fc1 = torch.nn.Linear((kk + 1) * NLI_DIM, HIDDEN_DIMS[0])
        self.fc2 = torch.nn.Linear(HIDDEN_DIMS[0], NUM_CLASSES)

    def forward(self, sim_stance_emb, nli_stance_emb, sim_body_emb, nli_body_emb):
        '''
        TODO
        '''
        batch_size = sim_stance_emb.shape[0]
        assert batch_size == nli_stance_emb.shape[0]
        assert batch_size == sim_body_emb.shape[0]
        assert batch_size == nli_body_emb.shape[0]
        
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
        xx = F.relu(xx)
        xx = self.fc2(xx)
        output = xx#F.softmax(xx, dim=1)

        return output

# class SimDeep

# class AgreeDeep(nn.Module): 

# class AgreeAttn(nn.Module): 