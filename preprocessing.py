from pathlib import Path
from time import monotonic
from typing import List, Tuple, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn
from nltk.tokenize import sent_tokenize, word_tokenize
from pandas import DataFrame
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from tqdm import tqdm
from multiprocessing import Pool

sentence_transformer = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')  # TODO try different pretrained model
sentence_transformer.max_seq_length = 512

def song_data_path() -> Path:
    return Path("spotify_abba_songdata.csv")


def _load_data(path: Path) -> DataFrame:
    full = pd.read_csv(path)
    return full[full.artist == "ABBA"][["artist", "song", "link", "text"]]


def create_vector_representation(data: List[str]) -> npt.NDArray[Any]:
    return sentence_transformer.encode(data)


def train_test_split(data: DataFrame):
    return sklearn.model_selection.train_test_split(data, test_size=0.2, random_state=42)


def split_into_sentences(text: str) -> List[str]:
    return sent_tokenize(' '.join(text.splitlines()))  # TODO improve language support


def split_sentence_into_n_word_strings(sentence: str, n: int) -> List[str]:
    # returns a list with sentences containing 5 words
    # TODO maybe remove punctuation also
    words: List[str] = word_tokenize(sentence)  # TODO improve language support
    # Better not, since we cannot check if the substring is in the text anymore
    # if remove_stop_words:
    #     stop_words = set(stopwords.words('english'))  # TODO improve language support
    #     words = [w for w in words if not w.lower() in stop_words]
    tuples_of_5 = []
    for i in range(len(words) - n):
        tuples_of_5.append(" ".join(words[i:i + n]))
    return tuples_of_5


def find_random_negative(query_text: str, data_set: DataFrame, max_retries=42):
    # One negative per positive query example should be sufficient.
    # BALANCED AS ALL THINGS SHOULD BE
    index = np.random.choice(data_set.index.tolist(), 1)
    retry = 0
    while query_text in data_set.text[index]:
        index = np.random.choice(data_set.index.tolist(), 1)
        retry += 1
        if retry > max_retries:
            raise ValueError(f"Tried {retry} times to find an document not containing {query_text=}")

    return int(index)


def create_queries(data: DataFrame, query_sizes: List[int]) -> List[
    Tuple[npt.NDArray[Any], npt.NDArray[Any], bool]]:
    """
        Creates a list of queries for the given data.

        :param data:
            Either test **or** training data having a column named text.
        :param query_sizes:
            defines in which sizes the queries should be
        :returns:
            A List of Tuples the query vector representation, the document vector representation
            and a bool vector size 1 if the document is relevant or not.
    """
    queries: List[Tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]] = []
    query_text_n = [(q, n, document_index) for document_index, sentences in
                    tqdm([(i, split_into_sentences(text)) for i, text in zip(data.index, data.text)], desc="Prep data.")
                    for sentence in sentences
                    for n in query_sizes
                    for q in split_sentence_into_n_word_strings(sentence, n)]
    query_texts, _, _ = map(list, zip(*query_text_n))
    pre = monotonic()
    query_vectors = sentence_transformer.encode(query_texts)
    print(f"Encoded all queries in {monotonic() - pre}")

    for (query, n, document_index), query_vector in tqdm(zip(query_text_n, query_vectors), desc='Create queries'):
        rel_doc_vec = data.loc[document_index, ["text_vector"]].tolist()[0]
        queries.append((query_vector, rel_doc_vec, np.array([1])))
        try:
            irrelevant_document_index = find_random_negative(query, data)
            irr_doc_vec = data.loc[irrelevant_document_index, ["text_vector"]].tolist()[0]
            queries.append((query_vector, irr_doc_vec, np.array([0])))
        except ValueError as e:
            print(e)
    return queries


def preprocess_data(batch_size: int, file_path: Path = song_data_path()) -> Tuple[DataLoader, DataLoader]:
    data = _load_data(file_path)
    data['text_vector'] = [sentence_transformer.encode(" ".join(t.splitlines())) for t in
                           tqdm(data.text, desc="Converting text to single vector")]
    train_data, test_data = train_test_split(data)
    train_set = create_queries(train_data, query_sizes=[4, 7, 10])
    test_set = create_queries(data=test_data, query_sizes=[4, 7, 10])

    return (
        DataLoader(train_set, batch_size=batch_size),
        DataLoader(test_set, batch_size=batch_size)
    )
