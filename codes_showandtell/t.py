# import os
#
# check_files_2019_1 = os.listdir(r'F:\mydata\2019\report_image_version5')
# check_files_2019_2 = os.listdir(r'F:\mydata\2019\report_image_version5-2')
#
# interaction = [check_file for check_file in check_files_2019_1 if check_file in check_files_2019_2]
#
# for check_file in interaction:
#     img_file_1 = os.listdir(os.path.join(r'F:\mydata\2019\report_image_version5', check_file))
#     img_file_2 = os.listdir(os.path.join(r'F:\mydata\2019\report_image_version5-2', check_file))
#
#     if img_file_1 != img_file_2:
#         print(check_file)

# import pandas as pd
# import os
# import shutil
# from tqdm import tqdm
#
# df = pd.read_excel('./data/patho_info_2019.xlsx')
# check_file_2019 = [r'F:\mydata\2019\2019-1', r'F:\mydata\2019\2019-2', r'F:\mydata\2019\2019-3']
#
# check_files_dia_true = []
# for row in range(df.shape[0]):
#     check_files_dia_true.append(df.loc[row][1])
#
# check_files = tqdm(os.listdir(check_file_2019[2]))
# for check_file in check_files:
#     if check_file in check_files_dia_true:
#         # print(check_file)
#         # image_files = os.listdir(os.path.join(check_file_2019[0], check_file))
#         source_path = os.path.join(check_file_2019[2], check_file)
#         target_path = os.path.join(r'F:\mydata\2019\2019', check_file)
#         if not os.path.exists(target_path):
#             shutil.copytree(source_path, target_path)

# import pandas as pd
# import os
# import shutil
# from tqdm import tqdm
# import numpy as np
#
# base_path = r'F:\mydata\2019\2019'
# base_save_path = r'F:\mydata\2019\2019-triton-radial'
# bar = tqdm(os.listdir(base_path))
#
# for check_file in bar:
#     image_files = os.listdir(os.path.join(base_path, check_file))
#     image_file_conform = [image_file for image_file in image_files
#                           if (len(image_file.split('-')) == 5 and
#                                image_file.split('-')[2] == 'triton' and
#                               image_file.split('-')[3] == 'Radial' and
#                               int(image_file.split('-')[1]) >= 30)]
#     if len(image_file_conform) != 0:
#         if len(image_file_conform) % 2 != 0:
#             print(check_file, 'num_error!')
#         else:
#             scores = np.array([image_file.split('-')[1] for image_file in image_file_conform])
#             image1 = image_file_conform[np.argsort(scores)[0]]
#             image2 = image_file_conform[np.argsort(scores)[1]]
#             if not os.path.exists(os.path.join(base_save_path, check_file)):
#                 os.makedirs(os.path.join(base_save_path, check_file))
#             shutil.copy(os.path.join(base_path, check_file, image1),
#                         os.path.join(base_save_path, check_file, image1))
#             shutil.copy(os.path.join(base_path, check_file, image2),
#                         os.path.join(base_save_path, check_file, image2))


"""取诊断意见"""
# import pandas as pd
# import os
# import shutil
# from tqdm import tqdm
# import numpy as np
# from string import punctuation
# import re
# with open('./data/2019-triton-radial/filename.txt', 'r') as f:
#     filenames = f.readlines()
#     f.close()
#
# punc = punctuation + u'.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，&～、|\s:：um 1234567890'
# df = pd.read_excel('./data/patho_info_2019.xlsx')
# diag = []
# for row in range(df.shape[0]):
#     if df.loc[row][1]+'\n' in filenames:
#         diag_ind = [5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]
#         diagnostic = ''
#         single_record = dict()
#         l_bracket, r_bracket = [], []
#         for id in diag_ind:
#             if type(df.loc[row][id]) == str:
#                 diagnostic += df.loc[row][id]
#         # 去括号
#         diagnostic = list(diagnostic)
#         for i, word in enumerate(diagnostic):
#             if word == '（':
#                 l_bracket.append(i)
#             if word == '）':
#                 r_bracket.append(i)
#         for idx in range(len(l_bracket)-1, -1, -1):
#             del diagnostic[l_bracket[idx]:r_bracket[idx]+1]
#         diagnostic = ''.join(diagnostic)
#         diagnostic = re.sub(r"[{}]+".format(punc), "", diagnostic)
#         if diagnostic != '':
#             single_record['check_num'] = df.loc[row][1]
#             single_record['diagnostic'] = diagnostic
#             diag.append(single_record)
#         else:
#             print(df.loc[row][1])
#
# with open('./data/2019-triton-radial/filename+diagnostic.txt', 'w', encoding='utf-8') as f:
#     for record in diag:
#         f.writelines(record['check_num'] + '\t' + record['diagnostic'] + '\n')
#     f.close()
#
# print(len(diag))

"""划分数据集"""
# import random
#
# with open('./data/2019-triton-radial/filename+diagnostic.txt', 'r', encoding='utf-8') as f:
#     data = f.readlines()
#     f.close()
#
# random.shuffle(data)
#
# with open('./data/2019-triton-radial/traindata.txt', 'w', encoding='utf-8') as f:
#     f.writelines(data[:3632])
#     f.close()
#
# with open('./data/2019-triton-radial/valdata.txt', 'w', encoding='utf-8') as f:
#     f.writelines(data[3632:4842])
#     f.close()
# with open('./data/2019-triton-radial/testdata.txt', 'w', encoding='utf-8') as f:
#     f.writelines(data[4842:])
#     f.close()

"""word2id"""
# import json
#
# with open('./data/2019-triton-radial/vocab.json', 'r', encoding='utf-8') as j:
#     word_map = json.load(j)
#     j.close()
#
# id2word = {}
# idx = 0
#
# for k, v in word_map.items():
#     id2word[v] = k
#
# data = json.dumps(id2word, indent=1, ensure_ascii=False)
#
# with open('./data/2019-triton-radial/word2idx.json', 'w', encoding='utf-8') as j:
#     j.write(data)
#     j.close()


"""对训练数据进行分词"""
import jieba
from config import *
import json

# jieba.load_userdict('./data/user_dict.txt')
#
#
# def encode_caption(word_map, c):
#     return [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
#         word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))
#
#
# with open('./data/2019-triton-radial/vocab.json', 'r', encoding='utf-8') as f:
#         word_map = json.load(f)
#
# with open('./data/2019-triton-radial/valdata.txt', 'r', encoding='utf-8') as f:
#     data = f.readlines()
#     f.close()
#
# recodes = []
# for record in data:
#     check_num = record.strip().split()[0]
#     diagnostic = record.strip().split()[1]
#     seq_cut = list(jieba.cut(diagnostic))
#     # enc_diagnostic = encode_caption(word_map, seq_cut)
#     single_recode = {}
#     single_recode['check_num'] = check_num
#     single_recode['caption'] = seq_cut
#     recodes.append(single_recode)
#
# with open('./data/2019-triton-radial/valdata_enc.txt', 'w', encoding='utf-8') as f:
#     for record in recodes:
#         f.writelines(record['check_num'] + '\t')
#         for word_enc in record['caption']:
#             f.writelines(str(word_enc) + '\t')
#         f.writelines('\n')
#     f.close()


import os

def takeSecond(elem):
    return elem[1]

check_files = os.listdir(r'D:\BaiduNetdiskDownload\2019')
for check_file in check_files[:100]:
    image_names = os.listdir(os.path.join(r'D:\BaiduNetdiskDownload\2019', check_file))
    scores_pair = [(s, int(s.split('-')[1])) for s in image_names]
    scores = [int(s.split('-')[1]) for s in image_names]
    scores_pair.sort(key=takeSecond, reverse=True)
    print(image_names)
    print([s[0] for s in scores_pair])













