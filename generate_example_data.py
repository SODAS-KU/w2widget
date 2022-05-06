"""
Messy file for generating data for widget example
"""

import pickle
from functools import partial
from typing import List

from gensim.models import Word2Vec
from nltk.corpus import inaugural, reuters, twitter_samples
from nltk.tokenize import WordPunctTokenizer
from openTSNE import TSNE
from sklearn.decomposition import PCA

from w2widget.doc2vec import Doc2Vec, calculate_inverse_frequency

import json
# import nltk
# nltk.download('twitter_samples')

### Twitter samples

# twitter_samples._readme = 'README.txt'
# print(twitter_samples.readme())
# twitter_samples.abspaths()

# tweets_path = twitter_samples.abspaths()[-1]
# # with open(tweets_path, 'r') as f:
# tweets = []
# with twitter_samples.open(tweets_path) as f:
#     for line in f:
#         tweets.append(json.loads(line.strip()))

# docs = [tweet["text"] for tweet in tweets if "retweeted_status" not in tweet]

### Reuters
docs = []

for path in reuters.fileids():
    with reuters.open(path) as f:
        docs.append(f.read())

### Inaugural speeches
# docs = []

# for doc in inaugural.abspaths():
#     # with open(doc, 'r') as f:
#     f = inaugural.open(doc)
#     docs.append(f.read())
#     f.close()

## Text preprocessing

tokenizer = WordPunctTokenizer()


def tokenize_with_ws(text: str, tokenizer) -> List[str]:
    return [x for y in [tokenizer(x) + [" "] for x in text.split()] for x in y]


tokenizer.tokenize_with_ws = partial(tokenize_with_ws, tokenizer=tokenizer.tokenize)

document_tokens = [
    [token.lower() for token in tokenizer.tokenize_with_ws(doc) if token.isalnum()]
    for doc in docs
]

tokens_with_ws = [tokenizer.tokenize_with_ws(doc) for doc in docs]

print("Saving tokens with white spaces")

with open("data/tokens_with_ws.pkl", "wb") as f:
    pickle.dump(tokens_with_ws, f)

## Train word2vec model

w2v = Word2Vec()

wv_model = Word2Vec(
    document_tokens,
    vector_size=200,
    window=10,
    workers=4,
    seed=42,
    epochs=10,
    min_count=2,
).wv


## Reduce dimensions

normed_vectors = wv_model.get_normed_vectors()

pca = PCA(n_components=50)
pca_embedding = pca.fit_transform(normed_vectors)

TSNE_embedding = TSNE(
    n_components=2, learning_rate="auto", random_state=420, verbose=1
).fit(pca_embedding)

wv_tsne_word_embedding = TSNE_embedding.transform(pca_embedding)

print("Saving wv_model")
with open("data/wv_model.pkl", "wb") as f:
    pickle.dump(wv_model, f)

print("Saving wv_tsne embeddings")
with open("data/wv_tsne_word_embedding.pkl", "wb") as f:
    pickle.dump(wv_tsne_word_embedding, f)

## doc2vec

word_weights = calculate_inverse_frequency(document_tokens)

dv_model = Doc2Vec(wv_model, word_weights)

dv_model.add_doc2vec(document_tokens)

dv_model.reduce_dimensions()

dv_tsne_embedding = dv_model.TSNE_embedding_array

print("Saving dv_model")
with open("data/dv_model.pkl", "wb") as f:
    pickle.dump(dv_model, f)

print("Saving dv_tsne embeddings")
with open("data/dv_tsne_embedding.pkl", "wb") as f:
    pickle.dump(dv_tsne_embedding, f)
