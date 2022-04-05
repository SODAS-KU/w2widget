from typing import Dict
from tqdm.autonotebook import tqdm
import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from openTSNE import TSNE

from collections import Counter
from typing import List, Dict

def calculate_inverse_frequency(docs:List[List[str]]) -> Dict:
    
    def word_if(word_count:int, total_count:int):
        return 1e-3/(1e-3+word_count/total_count)
    
    all_words = [x for y in docs for x in y]

    c = Counter(all_words)

    total_count = len(all_words)
    
    return {k:word_if(v, total_count) for k,v in c.items()}

class Doc2Vec:
    def __init__(self, wv_model, word_weights:Dict):
        self.wv_model = wv_model
        self.word_weights = word_weights
    
    def calculate_doc2vec(self, doc):
        word_vectors = np.array([self.wv_model.get_vector(token) for token in doc if token in self.wv_model])
        word_weights = np.array([self.word_weights[token] for token in doc if token in self.wv_model])
        weighted_vectors = word_vectors*word_weights[:,np.newaxis]
        
        return np.mean(weighted_vectors, axis=0)

    def add_doc2vec(self, docs):
        self.doc2vec_dict = {key:self.calculate_doc2vec(doc) for key, doc in tqdm(enumerate(docs), total=len(docs)) if [token for token in doc if token in self.wv_model]}
        self.doc2vec_array = np.array(list(self.doc2vec_dict.values()))
        self.scaler = StandardScaler().fit(self.doc2vec_array)
        
    def get_scaled_vectors(self):
        self.scaled_vectors = self.scaler.transform(self.doc2vec_array)
        return self.scaled_vectors

    def most_similar(self, query, n=10):
        distances = distance.cdist(
            np.array([self.calculate_doc2vec(query)]),
            self.doc2vec_array,
            "cosine")[0]

        sorted_index = list(reversed(np.argsort(distances)))

        return [(x,y) for x,y in zip(sorted_index, distances[sorted_index])][:n]
    
    def reduce_dimensions(self, pca_dims=50, n_components=2, verbose=0):
        self.get_scaled_vectors()
        
        self.pca = PCA(n_components=pca_dims)
        self.pca.fit(self.scaled_vectors)

        self.pca_embedding = self.pca.transform(self.scaled_vectors)

        self.tsne = TSNE(
            n_components=n_components, 
            learning_rate='auto',
            random_state=420,
            verbose=verbose                 
        ).fit(self.pca_embedding)

        self.TSNE_embedding_array = self.tsne.transform(self.pca_embedding)

    def get_TSNE_reduced_doc(self, doc):
        return self.tsne.transform(
            self.pca.transform(
                self.scaler.transform(
                    np.array([self.calculate_doc2vec(doc)])
                )
            )
        )