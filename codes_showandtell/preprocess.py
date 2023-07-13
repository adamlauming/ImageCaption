# import jieba
# import pandas as pd
#
# df_original = pd.read_excel('./data/2019-all/patho_info_2019.xlsx', sheet_name='patho_info_2019').values.tolist()
# cate_list = [5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]
# label_list = [5, 6, 7, 10, 11, 12, 13, 14, 16, 19, 20]
# # label = [裂孔，劈裂，水肿，前膜，玻璃体后脱离，内界膜，光感受器层，IS/OS层，神经上皮层脱离，RPE层脱离，
# #           脉络膜]
# recodes = []
# for row in df_original[:]:
#     single_recode = dict()
#     # 诊断意见
#     diagnostic = ''
#     for i in cate_list:
#         if type(row[i]) == str:
#             diagnostic += row[i]
#     single_recode['check_num'] = row[1]
#     # 去掉括号
#     diagnostic = list(diagnostic)
#     l_bracket = []
#     r_bracket = []
#     # 去掉每条诊断意见中的括号
#     for i, word in enumerate(diagnostic):
#         if word == '（':
#             l_bracket.append(i)
#         if word == '）':
#             r_bracket.append(i)
#     assert len(l_bracket) == len(r_bracket)
#     if len(l_bracket) != 0:
#         if len(l_bracket) != len(r_bracket):
#             print(row[0])
#         else:
#             for idx in range(len(l_bracket)-1, -1, -1):
#                 del diagnostic[l_bracket[idx]:r_bracket[idx]+1]
#     sentences = ''.join(diagnostic)
#     single_recode['diagnostic'] = sentences
#     # label
#     single_recode['label'] = ['0'] * 11
#     for id, i in enumerate(label_list):
#         if type(row[i]) == str:
#                if i == 16: #  神经上皮层脱离
#                    if '贴附' in row[i] and '脱离' not in row[i]:
#                        single_recode['label'][id] = '0'
#                    else:
#                        single_recode['label'][id] = '1'
#                elif i == 19:
#                    if '脱离' in row[i]:
#                        single_recode['label'][id] = '1'
#                    else:
#                        single_recode['label'][id] = '1'
#                else:
#                    single_recode['label'][id] = '1'
#
#     recodes.append(single_recode)
#
# with open('./data/2019-all/corpus.txt', 'w', encoding='utf-8') as f:
#     for single_recode in recodes:
#         f.write(single_recode['check_num'] + '\t' + single_recode['diagnostic'] + '\t' + ' '.join(single_recode['label']) + '\n')
#     f.close()


"""制作字典"""
# import jieba
# import re
# from collections import Counter
# from string import punctuation
# import json
#
# jieba.load_userdict('./data/user_dict.txt')
# jieba.del_word('处见')
# jieba.del_word('反射光')
# jieba.del_word('呈囊样')
# jieba.del_word('见囊样')
# jieba.del_word('片中')
# jieba.del_word('一大')
# jieba.del_word('一小')
# jieba.del_word('下大团')
# jieba.del_word('团且')
# jieba.del_word('一团')
# jieba.del_word('表面膜')
# jieba.del_word('缘浅')
# jieba.del_word('前及')
# jieba.del_word('合并')
# jieba.del_word('未愈')
# jieba.del_word('呈强')
# jieba.del_word('一长')
# jieba.del_word('并伴')
# jieba.del_word('侧反射')
# jieba.del_word('团其下')
# jieba.del_word('未贴')
# jieba.del_word('一中')
# jieba.del_word('近视')
#
# with open('./data/2019-all/corpus.txt', 'r', encoding='utf-8') as f:
#     data = f.readlines()
#     f.close()
#
# word_freq = Counter()
# punc = punctuation + u'um 1234567890'
#
# for recode in data:
#     diagnostic = recode.strip().split()[1]
#     diagnostic = re.sub(r"[{}]+".format(punc), "", diagnostic)
#     sentences = diagnostic.split('。')[:-1]
#     for seq in sentences:
#         word_cut = list()
#         seq_cut = jieba.cut(seq)
#         # for word in seq_cut:
#         #     word_cut.append(word)
#         # if '光感受器' in word_cut:
#         #     print(recode.strip().split()[0], word_cut)
#         word_freq.update(seq_cut)
#
# words = [word for word, cnt in word_freq.items()]
#
# word_map = {k:v+1 for v, k in enumerate(words)}
# word_map['<unk>'] = len(word_map) + 1
# word_map['<start>'] = len(word_map) + 1
# word_map['<end>'] = len(word_map) + 1
# word_map['<pad>'] = 0
#
# word_map = json.dumps(word_map, indent=1, ensure_ascii=False)
#
# # Save word map to a JSON
# with open('./data/2019-all/WORDMAP.json', 'w', encoding='utf-8') as j:
#     j.write(word_map)
#     j.close()
#
# word_freq_dict = dict(word_freq)
# word_freq_sorted = sorted(word_freq_dict.items(), key=lambda word_freq_dict:word_freq_dict[1], reverse=True)
# with open('./data/2019-all/word_freq.txt', 'w', encoding='utf-8') as f:
#     for item in word_freq_sorted:
#         f.writelines(str(item) + '\n')
#     f.close()

"""制作数据集"""
# import os
# from collections import Counter
# import random
# image_files_checknum = os.listdir(r'D:\BaiduNetdiskDownload\2019')
# recodes = []
# check_nums = list()
#
#
# def takeSecond(elem):
#     return elem[1]
#
#
# with open('./data/2019-all/corpus.txt', 'r', encoding='utf-8') as f:
#     for line in f:
#         check_num = line.strip().split()[0]
#         if check_num in image_files_checknum:
#             sel_imgs = list()
#             image_names = os.listdir(os.path.join(r'D:\BaiduNetdiskDownload\2019', check_num))
#             for image_name in image_names:
#                 if len(image_name.split('-')) == 5:
#                     score = image_name.split('-')[1]
#                     if int(score) > 20:
#                         sel_imgs.append(image_name)
#                 elif len(image_name.split('-')) == 3:
#                     score = image_name.split('-')[1]
#                     if int(score) >= 3:
#                         sel_imgs.append(image_name)
#                 elif len(image_name.split('-')) == 2:
#                     score = image_name.split('-')[1][:-4]
#                     if int(score) >= 3:
#                         sel_imgs.append(image_name)
#                 else:
#                     print(image_name, check_num)
#             if len(sel_imgs) != 0:
#                 if len(sel_imgs) == 4:
#                     recodes.append(line.strip() + '\t' + ' '.join(sel_imgs) + '\n')
#                 elif len(sel_imgs) < 4:
#                     recodes.append(line.strip() + '\t' + ' '.join(sel_imgs) + ('\t' + sel_imgs[0]) * (4-len(sel_imgs)) + '\n')
#                 elif len(sel_imgs) > 4:
#                     scores_pair = [(s, int(s.split('-')[1][0])) for s in sel_imgs]
#                     scores = [int(s.split('-')[1][0]) for s in sel_imgs]
#                     scores_pair.sort(key=takeSecond, reverse=True)
#                     recodes.append(line.strip() + '\t' + ' '.join([s[0] for s in scores_pair[:4]]) + '\n')
#             else:
#                 print(check_num)
#         check_nums.append(check_num)
#     f.close()
#
# check_fre = Counter()
# check_fre.update(check_nums)
# # print(check_fre)
# with open('./data/2019-all/paired_corpus.txt', 'w', encoding='utf-8') as f:
#     f.writelines(recodes)
#     f.close()
#
# random.shuffle(recodes)
# with open('./data/2019-all/traindata.txt', 'w', encoding='utf-8') as f:
#     f.writelines(recodes[:12754])
#     f.close()
#
# with open('./data/2019-all/valdata.txt', 'w', encoding='utf-8') as f:
#     f.writelines(recodes[12754:17005])
#     f.close()
#
# with open('./data/2019-all/testdata.txt', 'w', encoding='utf-8') as f:
#     f.writelines(recodes[17005:])
#     f.close()
import random
with open('./data/2019-all/paired_corpus.txt', 'r', encoding='utf-8') as f:
    data = f.readlines()
    f.close()

random.shuffle(data)
with open('./data/2019-all/traindata.txt', 'w', encoding='utf-8') as f:
    f.writelines(data[:11703])
    f.close()

with open('./data/2019-all/valdata.txt', 'w', encoding='utf-8') as f:
    f.writelines(data[11703:15604])
    f.close()

with open('./data/2019-all/testdata.txt', 'w', encoding='utf-8') as f:
    f.writelines(data[15604:])
    f.close()
"""查看每次检查包含的图片数量"""
# import os
# from collections import Counter
# checkfiles = os.listdir(r'D:\BaiduNetdiskDownload\2019')
#
# # image_nums = list()
# image_nums = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0}
# for check_file in checkfiles:
#     image_names = os.listdir(os.path.join(r'D:\BaiduNetdiskDownload\2019', check_file))
#     image_num = len(image_names)
#
#     for key, val in image_nums.items():
#         if image_num == int(key):
#             val += 1
#             image_nums[key] = val
# print(image_nums)






