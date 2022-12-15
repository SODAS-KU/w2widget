from collections import defaultdict
from functools import partial
from typing import Callable, List

from gensim.models.keyedvectors import KeyedVectors


def tokenize_with_ws(text: str, tokenizer: Callable[[str], List[str]]) -> List[str]:
    "Tokenize a string with whitespaces with the specified tokenizer"
    if isinstance(text, str):
        return [
            x for y in [tokenizer(token) + [" "] for token in text.split()] for x in y
        ]


def tokenizer_with_ws(
    tokenizer: Callable[[str], List[str]]
) -> Callable[[str], List[str]]:
    "Returns a function which tokenizes a string with whitespaces based on the specified tokenizer"
    return partial(tokenize_with_ws, tokenizer=tokenizer)


def generate_word2vec_format(index_to_key, vectors):

    key_vecs = "\n".join(
        [w + " " + " ".join(v) for w, v in zip(index_to_key, vectors.astype(str))]
    )

    return f"""{vectors.shape[0]} {vectors.shape[1]}
{key_vecs}"""


class WordVector(KeyedVectors):
    def __init__(self, vectors, index_to_key):
        self.vectors = vectors
        self.vector_size = vectors.shape[1]
        self.index_to_key = index_to_key
        self.key_to_index = {word: n for n, word in enumerate(index_to_key)}
        self.norms = None
        pass


def create_topic_dict(dictionary: dict[str, list]):
    """
    Function take a dictionary with topics as keys and keywords as values
    and makes into appropriate dict. format for w2widget.
    """

    topic_dict = defaultdict(dict)

    for key in dictionary.keys():
        topic_words = dictionary[key]
        topic_dict[key]["topic_words"] = topic_words
        topic_dict[key]["search_words"] = topic_words
        topic_dict[key]["negative_words"] = []
        topic_dict[key]["skip_words"] = []

    return dict(topic_dict)
