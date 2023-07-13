import torch
from torch.utils.data import Dataset
from config import *
import json
import jieba
from PIL import Image
import torchvision.transforms as transforms
import os
from string import punctuation
import re


def encode_caption(word_map, c):
    return [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))


# load jieba
jieba.load_userdict('./codes/data/user_dict.txt')
jieba.del_word('处见')
jieba.del_word('反射光')
jieba.del_word('呈囊样')
jieba.del_word('见囊样')
jieba.del_word('片中')
jieba.del_word('一大')
jieba.del_word('一小')
jieba.del_word('下大团')
jieba.del_word('团且')
jieba.del_word('一团')
jieba.del_word('表面膜')
jieba.del_word('缘浅')
jieba.del_word('前及')
jieba.del_word('合并')
jieba.del_word('未愈')
jieba.del_word('呈强')
jieba.del_word('一长')
jieba.del_word('并伴')
jieba.del_word('侧反射')
jieba.del_word('团其下')
jieba.del_word('未贴')
jieba.del_word('一中')
jieba.del_word('近视')


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """
    def __init__(self, split, transform=None):
        """
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        self.transform = transform
        self.punc = punctuation + u'um 1234567890'
        assert self.split in {'train', 'valid', 'test'}

        if split == 'train':
            with open(train_filename, 'r', encoding='utf-8') as f:
                self.check_files = f.readlines()[:]
        elif split == 'valid':
            with open(val_filename, 'r', encoding='utf-8') as f:
                self.check_files = f.readlines()[:]
        elif split == 'test':
            with open(test_filename, 'r', encoding='utf-8') as f:
                self.check_files = f.readlines()[:]
        with open(os.path.join(data_folder, 'WORDMAP_unstructed.json'), 'r', encoding='utf-8') as f:
            self.word_map = json.load(f)

    def __getitem__(self, index):
        # process image
        # jieba.load_userdict('./data/user_dict.txt')
        check_num = self.check_files[index].strip().split()[0]
        # print(check_num)
        # image_files = os.listdir(os.path.join(image_folder, check_num))
        image_files = self.check_files[index].strip().split()[13:17]
        if len(image_files) != 4:
            print(check_num)
        images = (self.transform(Image.open(os.path.join(image_folder, check_num, image_files[0]))),
                  self.transform(Image.open(os.path.join(image_folder, check_num, image_files[1]))),
                  self.transform(Image.open(os.path.join(image_folder, check_num, image_files[2]))),
                  self.transform(Image.open(os.path.join(image_folder, check_num, image_files[3]))),
                  )
        # if self.transform is not None:
        #     image1 = self.transform(image1)

        # process annotation
        # diagnostic = self.check_files[index].strip().split()[1]
        # diagnostic = list(jieba.cut(diagnostic))
        diagnostic = self.check_files[index].strip().split()[1]
        diagnostic = re.sub(r"[{}]+".format(self.punc), "", diagnostic)
        diagnostic = list(jieba.cut(diagnostic))
        enc_diagnostic = encode_caption(self.word_map, diagnostic)
        caption = torch.LongTensor(enc_diagnostic)
        caplen = torch.LongTensor([len(diagnostic) + 2])
        # label
        label = self.check_files[index].strip().split()[2:13]
        label = [int(i) for i in label]
        return images, check_num, caption, caplen, torch.Tensor(label)

    def __len__(self):
        return len(self.check_files)


if __name__ == '__main__':
    jieba.load_userdict('./codes/data/user_dict.txt')
    jieba.del_word('处见')
    jieba.del_word('反射光')
    jieba.del_word('呈囊样')
    jieba.del_word('见囊样')
    jieba.del_word('片中')
    jieba.del_word('一大')
    jieba.del_word('一小')
    jieba.del_word('下大团')
    jieba.del_word('团且')
    jieba.del_word('一团')
    jieba.del_word('表面膜')
    jieba.del_word('缘浅')
    jieba.del_word('前及')
    jieba.del_word('合并')
    jieba.del_word('未愈')
    jieba.del_word('呈强')
    jieba.del_word('一长')
    jieba.del_word('并伴')
    jieba.del_word('侧反射')
    jieba.del_word('团其下')
    jieba.del_word('未贴')
    jieba.del_word('一中')
    jieba.del_word('近视')
    transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    dataset = CaptionDataset('valid', transform)
    train_dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32,
                                                   shuffle=False, num_workers=0)

    for i, (_, check_num, caption, caplen) in enumerate(train_dataloader):
        # print(check_num)
        # break
        pass












