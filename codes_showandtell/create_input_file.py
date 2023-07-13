import os
from config import *
import jieba
from collections import Counter
import json

jieba.load_userdict('./data/user_dict.txt')
word_freq = Counter()
with open('./data/2019-triton-radial/filename+diagnostic.txt', 'r', encoding='utf-8') as f:
    data = f.readlines()
    f.close()

for recode in data:
    check_num = recode.strip().split()[0]
    diagnostic = recode.strip().split()[1]
    seq_cut = jieba.cut(diagnostic)
    word_freq.update(seq_cut)


words = [w for w in word_freq.keys() if word_freq[w] > min_word_feq]
word_map = {k: v + 1 for v, k in enumerate(words)}
word_map['<unk>'] = len(word_map) + 1
word_map['<start>'] = len(word_map) + 1
word_map['<end>'] = len(word_map) + 1
word_map['<pad>'] = 0

data = json.dumps(word_map, indent=1, ensure_ascii=False)
# Save word map to a JSON
with open('./data/2019-triton-radial/vocab.json', 'w', encoding='utf-8') as f:
    f.write(data)
    f.close()














