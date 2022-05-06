# w2v_widget
Widget for exploring and sampling words from text data through word2vec models in order to construct topic dictionaries.

## Package content

The package contains two modules:
- `doc2vec.py`
- `widget.py`

### Doc2Vec

```python
from w2widget.doc2vec import calculate_inverse_frequency, Doc2Vec

```

### Widget

```python
from w2widget.widget import Widget

```


## Notes
We need [openTSNE](https://opentsne.readthedocs.io/en/latest/installation.html) to get doc2vec embedding of topic documents