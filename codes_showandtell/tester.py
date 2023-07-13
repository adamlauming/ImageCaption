import json
import time
import jieba

import numpy as np

import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from nltk.translate.bleu_score import corpus_bleu
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from config import *
from dataset import CaptionDataset
from model import Encoder, DecoderWithAttention
from utils import *
from sklearn import metrics


def main():
    global word_map, vocab, id2_word
    jieba.load_userdict('./data/user_dict.txt')
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
    # load word_map
    word_map_file = os.path.join(data_folder, 'WORDMAP_3.json')
    with open(word_map_file, 'r', encoding='utf-8') as j:
        word_map = json.load(j)

    # load word2id
    with open(os.path.join(data_folder, 'word2idx_3.json'), 'r', encoding='utf-8') as j:
        id2_word = json.load(j)

    test_checkpoint = torch.load(best_checkpoint)
    decoder = test_checkpoint['decoder']
    encoder = test_checkpoint['encoder']

    print(decoder.vocab_size)
    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_loader = torch.utils.data.DataLoader(
        CaptionDataset('test', transform=transform),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    test(test_loader, encoder, decoder)


def test(test_loader, encoder, decoder):
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()
    top5accs = AverageMeter()
    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    labels = np.zeros((3900, 11))
    outputs = np.zeros((3900, 11))
    # Batches
    for i, (imgs, _, caps, caplens, label) in enumerate(test_loader):

        # Move to device, if available
        # imgs = imgs.to(device)
        label = label.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)
        with torch.no_grad():
            # Forward prop.
            if encoder is not None:
                imgs, classifier_out = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        label_loss = bce(classifier_out, label)
        if label.shape[0] == batch_size:
            labels[i * batch_size:i * batch_size + batch_size, :] = label.cpu().detach().float().numpy()
            outputs[i * batch_size:i * batch_size + batch_size, :] = classifier_out.cpu().detach().float().numpy()
        else:
            labels[i * batch_size:, :] = label.cpu().detach().float().numpy()
            outputs[i * batch_size:, :] = classifier_out.cpu().detach().float().numpy()
        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores_copy = scores.clone()
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Keep track of metrics
        top5 = accuracy(scores.data, targets.data, 5)
        top5accs.update(top5, sum(decode_lengths))

        if i % print_freq == 0:
            print('Validation: [{0}/{1}]\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(test_loader), top5=top5accs))

        # Store references (true captions), and hypothesis (prediction) for each image
        # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
        # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

        # References
        caps = caps[sort_ind]  # because images were sorted in the decoder
        img_captions = []
        for j in range(caps.shape[0]):
            img_caps = caps[j].tolist()
            img_caption = []
            for w in img_caps:
                if w not in {word_map['<start>'], word_map['<pad>']}:
                    img_caption.append(w)  # remove <start> and pads
            img_captions.append(img_caption)
            references.append(img_caption)

        # Hypotheses
        _, preds = torch.max(scores_copy, dim=2)
        preds = preds.tolist()
        temp_preds = list()
        for j, p in enumerate(preds):
            temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
        preds = temp_preds
        hypotheses.extend(preds)
        assert len(references) == len(hypotheses)

        """visualize caption and preds"""
        # assert len(img_captions) == len(preds)
        # img_captions_words = []
        # preds_words = []
        # for i in range(len(img_captions)):
        #     img_captions_word = ''
        #     preds_word = ''
        #     for j in img_captions[i]:
        #         # img_captions_word.append(id2_word[str(j)])
        #         img_captions_word += id2_word[str(j)]
        #     for l in preds[i]:
        #         # preds_word.append(id2_word[str(l)])
        #         preds_word += id2_word[str(l)]
        #     img_captions_words.append(img_captions_word)
        #     preds_words.append(preds_word)
        # print(img_captions_words)
        # print(preds_words)

    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(np.expand_dims(references, axis=1), hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

    """visualize caption and preds"""
    img_captions_words = []
    preds_words = []
    references = np.squeeze(references)

    for i in range(len(references)):
        img_captions_word = ''
        preds_word = ''
        for j in references[i]:
            # img_captions_word.append(id2_word[str(j)])
            img_captions_word += id2_word[str(j)]
        for l in hypotheses[i]:
            # preds_word.append(id2_word[str(l)])
            preds_word += id2_word[str(l)]
        img_captions_words.append(img_captions_word)
        preds_words.append(preds_word)
    with open('target.txt', 'w', encoding='utf-8') as f:
        for i in img_captions_words:
            f.write(i)
            f.write('\n')
        f.close()
    with open('pred.txt', 'w', encoding='utf-8') as f:
        for i in preds_words:
            f.write(i)
            f.write('\n')
        f.close()
    # print(img_captions_words)
    # print(preds_words)
    epoch_auc = metrics.roc_auc_score(labels, outputs)
    print(
        '\n * TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}, AUC - {epoch_auc}\n'.format(
            top5=top5accs,
            bleu=bleu4,
            epoch_auc=epoch_auc))

    # return bleu4


if __name__ == '__main__':
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'  # 屏蔽通知信息
    main()



#  BLEU-4 - 0.5124699533349092
#  BLEU-2 - 0.6800287786638363
#  BLEU-3 - 0.5903991963039011








