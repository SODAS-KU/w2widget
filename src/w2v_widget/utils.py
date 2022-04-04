from typing import Dict
from tqdm.autonotebook import tqdm
import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class Doc2Vec:
    def __init__(self, wv_model, word_if:Dict):
        self.wv_model = wv_model
        self.word_if = word_if
    
    def calculate_doc2vec(self, doc):
        return np.mean([self.wv_model.get_vector(token) for token in doc if token in self.wv_model], axis=0)

    def add_doc2vec(self, docs):
        self.doc2vec_dict = {key:self.calculate_doc2vec(doc) for key, doc in tqdm(enumerate(docs), total=len(docs)) if [token for token in doc if token in self.wv_model]}
        self.doc2vec_array = np.array(list(self.doc2vec_dict.values()))
    
    def get_normed_vectors(self):
        self.normed_vectors = normalize(self.doc2vec_array, axis=1)
        return self.normed_vectors

    def most_similar(self, query, n=10):
        distances = distance.cdist(
            np.array([self.calculate_doc2vec(query)]),
            self.doc2vec_array,
            "cosine")[0]

        sorted_index = list(reversed(np.argsort(distances)))

        return [(x,y) for x,y in zip(sorted_index, distances[sorted_index])][:n]
    
    def reduce_dimensions(self, pca_dims=50, n_components=2, verbose=1):
        self.pca = PCA(n_components=pca_dims)
        self.pca.fit(self.normed_vectors)

        self.pca_embedding = self.pca.transform(self.normed_vectors)

        self.TSNE_embedding = TSNE(
            n_components=n_components, 
            learning_rate='auto',
            init='random', 
            random_state=420,
            verbose=verbose                 
        ).fit(self.pca_embedding)

        self.TSNE_embedding_array = self.TSNE_embedding.transform(self.pca_embedding)

    def get_TSNE_reduced_doc(self, doc):
        return self.TSNE_embedding.transform(self.calculate_doc2vec(doc))