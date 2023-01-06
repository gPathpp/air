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


def song_data_path() -> Path:
    return Path("spotify_millsongdata.csv")


def _load_data() -> DataFrame:
    return pd.read_csv(song_data_path())


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
    index = np.random.randint(0, len(data_set))
    retry = 0
    while query_text in data_set.text.iloc[index]:
        index = np.random.randint(0, len(data_set))
        if retry > max_retries:
            raise ValueError(f"Tried {retry} times to find an document not containing {query_text=}")

    return index


def create_queries(data: DataFrame, load_from_file: bool, file_path: Path, query_sizes: List[int]) -> DataFrame:
    """
        Creates a list of queries for the given data.


        :param data:
            Either test **or** training data having a column named text.
        :param load_from_file:
            If true tries to load the queries from file
        :param file_path:
            loads/saves queries into this file
        :returns:
            A Dataframe with len_of_text, query text, the query vector representation, a document index
            and a bool value if the document is relevant or not.
    """
    if load_from_file and file_path.is_file():
        return pd.read_csv(file_path)
    queries: List[Tuple[int, str, npt.NDArray[Any], int, bool]] = []
    query_text_n = [(q, n, document_index) for document_index, sentences in
                    tqdm(enumerate([split_into_sentences(text) for text in data.text])) for sentence in sentences
                    for n
                    in query_sizes for q in
                    split_sentence_into_n_word_strings(sentence, n)]
    query_texts, _, _ = map(list, zip(*query_text_n))
    pre = monotonic()
    query_vectors = sentence_transformer.encode(query_texts)
    print(f"Encoded all queries in {monotonic() - pre}")

    for (query, n, document_index), vector in zip(query_text_n, query_vectors):
        queries.append((n, query, vector, document_index, True))
        try:
            irrelevant_document_index = find_random_negative(query, data)
            queries.append((n, query, vector, irrelevant_document_index, False))
        except ValueError as e:
            print(e)
    query_dataframe = DataFrame(data=queries, columns=['len_of_text', 'text', 'vector', 'document', 'relevance'])
    query_dataframe.to_csv(file_path)
    return query_dataframe


def preprocess_data(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    data = _load_data()
    if 'vector' not in data.columns:
        texts = [split_into_sentences(text) for text in tqdm(data.text, desc="Splitting text into sentences")]
        data['vector'] = [create_vector_representation(s) for s in tqdm(texts, desc="Converting text to vectors")]
        data.to_csv(song_data_path())
    if 'text_vector' not in data.columns:
        data['text_vector'] = [sentence_transformer.encode(" ".join(t.splitlines())) for t in
                               tqdm(data.text, desc="Converting text to single vector")]
        data.to_csv(song_data_path())
    train_data, test_data = train_test_split(data)
    train_queries = create_queries(data=train_data, load_from_file=True,
                                   file_path=Path('train_queries_len_4710.csv'), query_sizes=[4, 7, 10])
    test_queries = create_queries(data=test_data, load_from_file=True, file_path=Path('test_queries_len_4710.csv'),
                                  query_sizes=[4, 7, 10])
    return DataLoader(train_queries, batch_size=batch_size), DataLoader(test_queries, batch_size=batch_size)
