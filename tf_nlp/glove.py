import numpy as np
import tensorflow as tf

from .models import Embedding
from .vocab  import Vocab

def load_glove(path, max_words=None):
    glove_vectors = []

    with open(path) as f:
        for i, line in enumerate(f):
            if len(line) < 1: break
            if max_words is not None and i >= max_words: break
            line = line.split(' ')
            word, vector = line[0], np.array([float(x) for x in line[1:]], dtype=np.float32)
            glove_vectors.append((word, vector))
    vocab = Vocab([word for word, _ in glove_vectors])
    embedding_size = len(glove_vectors[0][1])
    embedding_matrix = np.empty((len(vocab), embedding_size), dtype=np.float32)
    embedding_matrix[vocab.eos] = np.random.normal(size=(embedding_size,))
    embedding_matrix[vocab.unk] = np.random.normal(size=(embedding_size,))
    for word, vector in glove_vectors:
        embedding_matrix[vocab[word]] = vector
    return vocab, Embedding(len(vocab), embedding_size, tf.constant(embedding_matrix))
