from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from openTSNE import TSNE
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm.autonotebook import tqdm


def calculate_inverse_frequency(docs: List[List[str]]) -> Dict:
    """Calculates from inverse frequencies for words from a list of tokenized documents

    Args:
        docs (List[List[str]]): List of tokenized documents.

    Returns:
        Dict: Dictionary with words as keys and inverse frequencies as values
    """

    def word_if(word_count: int, total_count: int):
        return 1e-3 / (1e-3 + word_count / total_count)

    all_words = [x for y in docs for x in y]

    counter = Counter(all_words)

    total_count = len(all_words)

    return {k: word_if(v, total_count) for k, v in counter.items()}


class Doc2Vec:
    """Class for calculating doc2vec from a gensim wv_model"""

    def __init__(self, wv_model, word_weights: Dict):
        self.wv_model = wv_model
        self.word_weights = word_weights

        self.doc2vec_dict = None
        self.doc2vec_array = None

        self.scaler = None

        self.pca = None
        self.pca_embedding_array = None

        self.tsne = None
        self.TSNE_embedding_array = None

        self.scaled_vectors = None

    def add_doc2vec(self, docs: List[List[str]]):
        """Calculates and adds document-vectors based on the wv_model and provided documents based on the word weights provided

        Args:
            docs (_type_): _description_
        """
        # add doc:vector dictionary
        self.doc2vec_dict = {
            key: self.calculate_doc2vec(doc)
            for key, doc in tqdm(
                enumerate(docs),
                total=len(docs),
                smoothing=0,
                desc="Calculating document vectors",
            )
            if [token for token in doc if token in self.wv_model]
        }

        # Add array
        self.doc2vec_array = np.array(list(self.doc2vec_dict.values()))

        # Fit the scaler
        self.scaler = StandardScaler().fit(self.doc2vec_array)

    def calculate_doc2vec(self, doc: List[str]):
        # Get the word vectors from the document
        word_vectors = np.array(
            [self.wv_model.get_vector(token) for token in doc if token in self.wv_model]
        )

        # Get the weights for the words
        word_weights = np.array(
            [self.word_weights[token] for token in doc if token in self.wv_model]
        )

        # Multiply the vectors
        weighted_vectors = word_vectors * word_weights[:, np.newaxis]

        # Return the product
        return np.mean(weighted_vectors, axis=0)

    def reduce_dimensions(
        self,
        pca_dims: int = 50,
        n_components: int = 2,
        verbose: int = 0,
        random_state: int = 420,
    ):
        self.get_scaled_vectors()

        self.pca = PCA(n_components=pca_dims)
        self.pca.fit(self.scaled_vectors)

        self.pca_embedding_array = self.pca.transform(self.scaled_vectors)

        self.tsne = TSNE(
            n_components=n_components,
            learning_rate="auto",
            random_state=random_state,
            verbose=verbose,
        ).fit(self.pca_embedding_array)

        self.TSNE_embedding_array = self.tsne.transform(self.pca_embedding_array)

    def get_scaled_vectors(self):
        self.scaled_vectors = self.scaler.transform(self.doc2vec_array)
        return self.scaled_vectors

    def most_similar(
        self, query: List[str], top_n: int = 10
    ) -> List[Tuple[int, float]]:

        # Calculate distances
        distances = distance.cdist(
            np.array([self.calculate_doc2vec(query)]), self.doc2vec_array, "cosine"
        )[0]

        # Sort them from most to least similar
        sorted_index = list(reversed(np.argsort(distances)))

        # Return the top_n most similar
        return list(zip(sorted_index, distances[sorted_index]))[:top_n]

    def get_TSNE_reduced_doc(self, doc):
        return self.tsne.transform(
            self.pca.transform(
                self.scaler.transform(np.array([self.calculate_doc2vec(doc)]))
            )
        )
