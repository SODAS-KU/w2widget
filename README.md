# w2widget
Widget for exploring and sampling words from text data through word2vec models in order to construct topic dictionaries.

## Package content

The `w2widget` package contains two modules:
- `doc2vec.py`
- `widget.py`

## Examples

In the `widget_example.ipynb` you can play with the widget from pretrained data from [Reuters](http://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html) dataset.

If you want to see an example of the data-workflow generating the necessary input, check out `workflow_example.ipynb`.

### Doc2Vec

This module helps with calculating and handling doc2vec. The approach applied is that every document's vector is calculated by taking a weighted (ie. based on inverse frequencies) average of the document's word vectors.

```python
from w2widget.doc2vec import calculate_inverse_frequency, Doc2Vec

# Calculate word weigts from inverse frequency
word_weights = calculate_inverse_frequency(document_tokens)

# Initiate the model
dv_model = Doc2Vec(wv_model, word_weights)

# Add documents and calculated the document vectors
dv_model.add_doc2vec(document_tokens)

# reduce the dimensions
dv_model.reduce_dimensions()

# Store the embeddings
two_dim_doc_embedding = dv_model.TSNE_embedding_array
```

### Widget

This widget module displays the results from:
- A gensim word2vec model,
- it's 2-dimensional embedding (ie. TSNE).
- The custom implemented doc2vec model,
- it's 2-dimensional embedding (ie. TSNE).
- A list of tokenized documents with whitespaces and
- optionally a list of initial search words

```python
from w2widget.widget import Widget

wv_widget = Widget(
    wv_model,
    two_dim_word_embedding,
    tokens_with_ws
    dv_model=None,
    two_dim_doc_embedding=None,
    initial_search_words=[],
)

wv_widget.display_widget()
```

You can save the topics to a `json` file from the widget, or access them from the dictionary stored in `wv_widget.topics`.