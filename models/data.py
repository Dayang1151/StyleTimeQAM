import re
import gensim
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

class LCQMC_Dataset(Dataset):
    def __init__(self, LCQMC_file, vocab_file, max_char_len):
        p, h, self.label = load_sentences(LCQMC_file)
        word2idx, _, _ = load_vocab(vocab_file)
        self.p_list, self.p_lengths, self.h_list, self.h_lengths = word_index(p, h, word2idx, max_char_len)
        self.p_list = torch.from_numpy(self.p_list).type(torch.long)
        self.h_list = torch.from_numpy(self.h_list).type(torch.long)
        self.max_length = max_char_len
        
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.p_list[idx], self.p_lengths[idx], self.h_list[idx], self.h_lengths[idx], self.label[idx]
    

def load_sentences(file, data_size=None):
    df = pd.read_csv(file,encoding='GBK')
    p = map(get_word_list, df['text_a'].values[0:data_size])
    h = map(get_word_list, df['text_b'].values[0:data_size])
    label = df['label'].values[0:data_size]
    return p, h, label

# word->index
def word_index(p_sentences, h_sentences, word2idx, max_char_len):
    p_list, p_length, h_list, h_length = [], [], [], []
    for p_sentence, h_sentence in zip(p_sentences, h_sentences):
        p = [word2idx[word] for word in p_sentence if word in word2idx.keys()]
        h = [word2idx[word] for word in h_sentence if word in word2idx.keys()]
        p_list.append(p)
        p_length.append(min(len(p), max_char_len))
        h_list.append(h)
        h_length.append(min(len(h), max_char_len))
    p_list = pad_sequences(p_list, maxlen = max_char_len)
    h_list = pad_sequences(h_list, maxlen = max_char_len)
    return p_list, p_length, h_list, h_length

def load_vocab(vocab_file):
    vocab = [line.strip() for line in open(vocab_file, encoding='utf-8').readlines()]
    word2idx = {word: index for index, word in enumerate(vocab)}
    idx2word = {index: word for index, word in enumerate(vocab)}
    return word2idx, idx2word, vocab



def get_word_list(query):
    regEx = re.compile('[\\W]+')
    sentences = regEx.split(query.lower())
    str_list = []
    for sentence in sentences:
        str_list.append(sentence)
    return [w for w in str_list if len(w.strip()) > 0]



def load_embeddings(embdding_path):
    # glove_file = datapath("C:/Users/16603/Desktop/QA问题/TextMatch-master/data/glove.840B.300d.txt") 
    # tmp_file = get_tmpfile("C:/Users/16603/Desktop/QA问题/TextMatch-master/data/w2v.txt") 
    # _ = glove2word2vec(glove_file, tmp_file)  
    model = KeyedVectors.load_word2vec_format(embdding_path) 

    # model = gensim.models.KeyedVectors.load_word2vec_format(embdding_path, binary=False)
    embedding_matrix = np.zeros((len(model.index_to_key) + 1, model.vector_size))
    for idx, word in enumerate(model.index_to_key):
        embedding_matrix[idx + 1] = model[word]
    return embedding_matrix

def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post',
                  truncating='post', value=0.):

    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x
