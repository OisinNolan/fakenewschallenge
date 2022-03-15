import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from functools import reduce

SIM_DIM = 384
NLI_DIM = 768
NUM_CLASSES = 3
TOPK = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SIM_SCALAR = 50

class AgreemNet(nn.Module):
    def __init__(self):
        super(AgreemNet, self).__init__()
        self.sim_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.nli_encoder = SentenceTransformer('sentence-transformers/nli-distilroberta-base-v2')
        self.attention = torch.nn.MultiheadAttention(embed_dim=SIM_DIM, vdim=NLI_DIM, num_heads=1)
        self.reduce_head = torch.nn.Linear(NLI_DIM, SIM_DIM)
        self.classifier_fully_connected = torch.nn.Linear(1 + (SIM_DIM * 2) + (SIM_DIM) + (SIM_DIM * TOPK), NUM_CLASSES)

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
        sent_len = len(B[0])
        # flatten list of lists of lists -> list of lists
        B_flat = reduce(lambda x, y: x+y, B)

        # Compute embeddings
        head_sim = torch.from_numpy(self.sim_encoder.encode(H)).to(DEVICE) * SIM_SCALAR
        body_sims = self.encode_(self.sim_encoder, SIM_DIM, B_flat).view(batch_size, sent_len, SIM_DIM).to(DEVICE) * SIM_SCALAR
        head_nli = torch.from_numpy(self.nli_encoder.encode(H)).to(DEVICE)
        body_nlis = self.encode_(self.nli_encoder, NLI_DIM, B_flat).view(batch_size, sent_len, NLI_DIM).to(DEVICE)

        # Attention layer
        attn_out, attn_weights = self.attention(
            head_sim.view(1, batch_size, SIM_DIM),
            body_sims.view(sent_len, batch_size, SIM_DIM),
            body_nlis.view(sent_len, batch_size, NLI_DIM),
            key_padding_mask=torch.from_numpy(np.array(B) == '[PAD]')
        )

        # Get top k similar (attended) sentences 
        _, idx = torch.topk(attn_weights, k=TOPK)
        topk_sims = torch.zeros((batch_size, TOPK, SIM_DIM))
        for i in range(idx.shape[0]):
            topk_sims[i] = torch.index_select(body_sims[i], 0, idx.squeeze()[i])

        # Linear transform head_nli to be same dimension as attn_out
        reduced_head = self.reduce_head(head_nli)
        catted = torch.cat((head_sim, topk_sims.flatten(1, 2), attn_out.squeeze(), reduced_head, F.cosine_similarity(attn_out.squeeze(), reduced_head).view(batch_size, 1)), dim=1)
        logits = self.classifier_fully_connected(catted)
        return F.softmax(logits, dim=1)