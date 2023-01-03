import numpy as np
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from torch.nn import CosineSimilarity
from torch.nn.functional import one_hot
# Train function
def baseline(song_text: str, query: str) -> float:
    # returns the cosine similariry
    doc_vec = transformer.encode(song_text)
    query_vec = transformer.encode(query)
    return CosineSimilarity()(doc_vec, query_vec)