from sentence_transformers import SentenceTransformer
from torch import embedding
from dataset import FakeNewsDataset
from csv import reader
import pickle
from util import pad_tokenize, dont_encode_pad
from tqdm import tqdm
import numpy as np

SIM_DIM = 384
NLI_DIM = 768

# Format of CSV files
STANCE_TEXT = 0
STANCE_ID = 1
STANCE_LABEL = 2
BODY_ID = 0
BODY_TEXT = 1

def process_file(filename, sim_encoder: SentenceTransformer, nli_encoder: SentenceTransformer, type="body"):
    PIK = f"{filename}.{type}.dat"
    print(f"GENERATING {PIK}.")

    # open file in read mode
    with open(filename, 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        next(csv_reader, None) # skip header
        # Iterate over each row in the csv using reader object
        for ii, row in tqdm(enumerate(csv_reader)):
            if (type == "stance"):
                body_id = row[STANCE_ID]
                sim_embedding = sim_encoder.encode(row[STANCE_TEXT])
                nli_embedding = nli_encoder.encode(row[STANCE_TEXT])
                stance = row[STANCE_LABEL]

                to_dump = [body_id, sim_embedding, nli_embedding, stance]
            
            elif (type == "body"):
                body_id = row[BODY_ID]
                sentences = pad_tokenize([row[BODY_TEXT]])[0]
                sim_embeddings = dont_encode_pad(sim_encoder, SIM_DIM, sentences)
                nli_embeddings = dont_encode_pad(nli_encoder, NLI_DIM, sentences)

                to_dump = [body_id, sim_embeddings, nli_embeddings]

            with open(PIK, "ab") as f:
                pickle.dump(to_dump, f)
    print("-"*10, "\n")

def main():
    sim_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    nli_encoder = SentenceTransformer('sentence-transformers/nli-distilroberta-base-v2')

    # process_file("data/train_bodies.csv", encoder, dim, type="body")
    # process_file("data/train_stances.csv", encoder, dim, type="stance")

    # process_file("data/train_bodies.csv", encoder, dim, type="body")
    # process_file("data/train_stances.csv", encoder, dim, type="stance")

    process_file(
        filename="data/custom/train_bodies.csv",
        sim_encoder=sim_encoder,
        nli_encoder=nli_encoder,
        type="body",
    )
    
    process_file(
        filename="data/custom/train_stances.csv",
        sim_encoder=sim_encoder,
        nli_encoder=nli_encoder,
        type="stance",
    )

if __name__ == "__main__":
    main()