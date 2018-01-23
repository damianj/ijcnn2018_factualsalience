from gensim.models import KeyedVectors
from numpy import asarray, float32
import pickle


def process_embeddings(file, w2v_format=True, is_binary=True):
    embeddings = {}

    if w2v_format:
        word2vec = KeyedVectors.load_word2vec_format(file, binary=is_binary, datatype=float32)

        for word in word2vec.wv.vocab:
            embeddings[word] = word2vec.wv[word].reshape(1, -1)
    else:
        with open(file, mode="r") as glove_txt:
            for line in glove_txt:
                try:
                    values = line.split()
                    word = values[0]
                    coefficients = asarray(values[1:], dtype='float32')
                    embeddings[word] = coefficients.reshape(1, -1)
                except ValueError:
                    pass

    return embeddings


with open("glove_dict.pkl", mode="wb") as pkl_out:
    pickle.dump(process_embeddings("/home/damian/Dropbox/Projects & Academia/Pre-trained embedding files/GloVe_pretrained_embeddings.txt", False, False), pkl_out)
