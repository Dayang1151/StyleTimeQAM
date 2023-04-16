import re
import gensim
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


class LEA_Dataset(Dataset):
    def __init__(self, LEA_file, vocab_file, max_char_len):
        # p, h, self.label = load_sentences(LEA_file)
        sentence,self.user,self.label,self.timestamp,self.match = load_sentences(LEA_file)
        word2idx, _, _ = load_vocab(vocab_file)
        self.sentence_list = torch.from_numpy(word_index(sentence,word2idx, max_char_len)).to(torch.long)
        # self.sentence_list = torch.from_numpy(self.sentence_list).to(torch.long)
        self.max_length = max_char_len

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.sentence_list[idx], self.user[idx], self.label[idx], self.timestamp[idx], self.match[idx]

    def getdata(self):
        return self.sentence_list, self.user, self.label, self.timestamp, self.match


def load_match(d):
    arr_2 = []
    for i in range(len(d)):
        arr_1 = []
        for j in range(len(d[i][1:-1])):
            if (d[i][1:-1][j] == '0') or (d[i][1:-1][j] == '1'):
                arr_1.append(int(d[i][1:-1][j]))
        arr_2.append(arr_1)
    return np.array(arr_2)

# 加载word_index训练数据
def load_sentences(file, data_size=None):
    df = pd.read_csv(file, encoding='GBK')
    sentence = map(get_word_list, df['sentence'].values[0:data_size])
    user = df['user_ID'].values[0:data_size]
    label = df['label'].values[0:data_size]
    timestamp = df['timestamp'].values[0:data_size]
    match = df['match'].values[0:data_size]
    match = load_match(match)
    return sentence,user,label,timestamp,match

file = 'new_test.csv'
# a,b,c,d,e = load_sentences(file)
# print(b,c,d,e)

# word->index
def word_index(sentence,word2idx, max_char_len):
    sentence_list = []
    for s_sentence in sentence:
        s = [word2idx[word] for word in s_sentence if word in word2idx.keys()]
        sentence_list.append(s)
    sentence_list = pad_sequences(sentence_list, maxlen=max_char_len)
    return sentence_list

# # word->index
# def word_index(p_sentences, h_sentences, word2idx, max_char_len):
#     sentences_list = []
#     for p_sentence, h_sentence in zip(p_sentences, h_sentences):
#         p = [word2idx[word] for word in p_sentence if word in word2idx.keys()]
#         h = [word2idx[word] for word in h_sentence if word in word2idx.keys()]
#         p_list.append(p)
#         p_length.append(min(len(p), max_char_len))
#         h_list.append(h)
#         h_length.append(min(len(h), max_char_len))
#     p_list = pad_sequences(p_list, maxlen=max_char_len)
#     h_list = pad_sequences(h_list, maxlen=max_char_len)
#     return p_list, p_length, h_list, h_length

# 加载字典
def load_vocab(vocab_file):
    vocab = [line.strip() for line in open(vocab_file, encoding='GBK').readlines()]
    # print(vocab)
    word2idx = {word: index for index, word in enumerate(vocab)}
    # print(word2idx)
    idx2word = {index: word for index, word in enumerate(vocab)}
    # print(idx2word)
    return word2idx, idx2word, vocab


def get_word_list(query):
    regEx = re.compile('[\\W]+')  # 我们可以使用正则表达式来切分句子，切分的规则是除单词，数字外的任意字符串
    sentences = regEx.split(query.lower())
    str_list = []
    for sentence in sentences:
        str_list.append(sentence)
    return [w for w in str_list if len(w.strip()) > 0]


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post',
                  truncating='post', value=0.):
    """ pad_sequences
    把序列长度转变为一样长的，如果设置了maxlen则长度统一为maxlen，如果没有设置则默认取
    最大的长度。填充和截取包括两种方法，post与pre，post指从尾部开始处理，pre指从头部
    开始处理，默认都是从尾部开始。
    Arguments:
        sequences: 序列
        maxlen: int 最大长度
        dtype: 转变后的数据类型
        padding: 填充方法'pre' or 'post'
        truncating: 截取方法'pre' or 'post'
        value: float 填充的值
    Returns:
        x: numpy array 填充后的序列维度为 (number_of_sequences, maxlen)
    """
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

