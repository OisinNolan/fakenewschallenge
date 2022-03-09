import torch
from torch import nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from functools import reduce

SIM_DIM = 384
NLI_DIM = 768
NUM_CLASSES = 3

class AgreemNet(nn.Module):
    def __init__(self):
        super(AgreemNet, self).__init__()
        self.sim_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.nli_encoder = SentenceTransformer('sentence-transformers/nli-distilroberta-base-v2')
        self.attention = torch.nn.MultiheadAttention(embed_dim=SIM_DIM, vdim=NLI_DIM, num_heads=1)
        self.nli_head_weight = torch.nn.Linear(NLI_DIM, SIM_DIM)
        self.classifier = torch.nn.Linear(SIM_DIM * 2, NUM_CLASSES)

    def forward(self, H, B):
        '''
        H: (N,)
        B: (N, M,)
        '''
        batch_size = len(B)
        sent_len = len(B[0])
        # flatten list of lists of lists -> list of lists
        B_flat = reduce(lambda x, y: x+y, B)

        # Compute embeddings
        head_sim = torch.from_numpy(self.sim_encoder.encode(H))
        body_sims = torch.from_numpy(self.sim_encoder.encode(B_flat)).view(batch_size, sent_len, SIM_DIM)
        head_nli = torch.from_numpy(self.nli_encoder.encode(H))
        body_nlis = torch.from_numpy(self.nli_encoder.encode(B_flat)).view(batch_size, sent_len, NLI_DIM)

        # Attention layer
        attn_out, attn_weights = self.attention(head_sim.view(1, batch_size, SIM_DIM), body_sims.view(sent_len, batch_size, SIM_DIM), body_nlis.view(sent_len, batch_size, NLI_DIM))
        # attn_out = attn_out.view(1, SIM_DIM)

        # Linear transform head_nli to be same dimension as attn_out
        reduced_head = self.nli_head_weight(head_nli)
        catted = torch.cat((attn_out.squeeze(), reduced_head), dim=1)
        logits = self.classifier(catted)
        return F.softmax(logits, dim=0)