import torch
from torch import nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

SIM_DIM = 384
NLI_DIM = 768
NUM_CLASSES = 3

class AgreemNet(nn.Module):
    def __init__(self):
        super(AgreemNet, self).__init__()
        self.sim_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.nli_encoder = SentenceTransformer('sentence-transformers/nli-distilroberta-base-v2')
        self.attention = torch.nn.MultiheadAttention(embed_dim=SIM_DIM, vdim=NLI_DIM, num_heads=1)
        self.nli_head_weight = torch.nn.Parameter(torch.zeros(NLI_DIM, SIM_DIM))
        self.fully_connected = torch.nn.Parameter(torch.zeros(1 + (SIM_DIM * 2), NUM_CLASSES))

    def forward(self, head, bodies):
        '''
        H: (N,)
        B: (N, M,)
        '''
        # How can we batchify this when M varies? Should we use some sort of padding?
        # Or would padding slow us down
        # It seems that encoding all the bodies in the batch at once could help speed up
        # If we have a list of how many sentences in each body we could reconstruct
        #       a flattened set of bodies. 
        # Seems like we can also provide key_padding_mask to MultiHeadAttention
        # If we first add padding then we can flatten and unflatten without a for-loop
        N = len(bodies)

        # Compute embeddings
        head_sim = torch.from_numpy(self.sim_encoder.encode(head))
        body_sims = torch.from_numpy(self.sim_encoder.encode(bodies))
        head_nli = torch.from_numpy(self.nli_encoder.encode(head))
        body_nlis = torch.from_numpy(self.nli_encoder.encode(bodies))

        # Attention layer
        attn_out, attn_weights = self.attention(head_sim.view(1, 1, SIM_DIM), body_sims.view(N, 1, SIM_DIM), body_nlis.view(N, 1, NLI_DIM))
        attn_out = attn_out.view(1, SIM_DIM)
        
        # Linear transform head_nli to be same dimension as attn_out
        reduced_head = torch.matmul(head_nli, self.nli_head_weight).view(1, SIM_DIM)
        catted = torch.cat((attn_out.view(SIM_DIM), reduced_head.view(SIM_DIM), F.cosine_similarity(attn_out, reduced_head)))
        logits = torch.matmul(catted, self.fully_connected)
        return F.softmax(logits, dim=0)