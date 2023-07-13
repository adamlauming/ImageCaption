import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from dataset import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import jieba
from sklearn import metrics
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge

# Parameters
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Load model
checkpoint = torch.load(best_checkpoint)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

# Load word map (word2ix)
word_map_file = os.path.join(data_folder, 'WORDMAP_unstructed.json')
with open(word_map_file, 'r', encoding='utf-8') as j:
    word_map = json.load(j)

rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# load word2id
with open('./data/2019-all/word2idx_unstructed.json', 'r', encoding='utf-8') as j:
    id2_word = json.load(j)
# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def evaluate(beam_size):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # DataLoader
    # jieba.load_userdict('./data/user_dict.txt')

    loader = torch.utils.data.DataLoader(
        CaptionDataset('test', transform=transform),
        batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    # For each image
    for i, (image, check_num, caps, caplens, label) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):
        k = beam_size

        # Move to GPU device, if available
        # image = image.to(device)  # (1, 3, 256, 256)

        # Encode
        encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)

        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words // vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            # for ind, next_word in enumerate(next_word_inds):
            #     print(next_word)
            #     print(next_word.item())
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               (next_word.item() != word_map['<end>'])]
            # print(incomplete_inds)
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1
        if complete_seqs_scores:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]
        else:
            seq = ''

        # References
        # caps = caps[sort_ind]  # because images were sorted in the decoder
        img_captions = []
        for j in range(caps.shape[0]):
            img_caps = caps[j].tolist()
            img_caption = []
            for w in img_caps:
                if w not in {word_map['<start>'], word_map['<pad>'], word_map['<end>']}:
                    img_caption.append(w)  # remove <start> and pads
            img_captions.append(img_caption)
        references.append(img_captions)

        # Hypotheses
        hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

        assert len(references) == len(hypotheses)

    # Calculate BLEU-4 scores
    # bleu4 = corpus_bleu(np.expand_dims(references, axis=1), hypotheses)
    bleu4 = corpus_bleu(references, hypotheses)

    # Calculate CIDEr
    gts = dict()
    res = dict()
    for i in range(len(hypotheses)):
        gts[str(i)] = [' '.join(str(w) for w in hypotheses[i])]
        res[str(i)] = [' '.join(str(w) for w in references[i][0])]
    cide_score = Cider()
    cider_score, cider_scores = cide_score.compute_score(res, gts)

    print('cider_score', cider_score)
    # Calculate bleu
    bleu_score = Bleu()
    bleu_scorer, bleu_scorers = bleu_score.compute_score(res, gts)
    print('bleu_scorer', bleu_scorer)
    rouge_score = Rouge()
    rouge_scorer, rouge_scorers = rouge_score.compute_score(res, gts)
    print('rouge_scorer', rouge_scorer)

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

    check_num = list()
    with open(test_filename, 'r', encoding='utf-8') as f:
        for line in f:
            check_num.append(line.strip().split()[0])
    with open('target.txt', 'w', encoding='utf-8') as f:
        for idx, i in enumerate(img_captions_words):
            f.write(check_num[idx] + '\t' + i + '\n')
        f.close()
    with open('pred.txt', 'w', encoding='utf-8') as f:
        for idx, i in enumerate(preds_words):
            f.write(check_num[idx] + '\t' + i + '\n')
        f.close()
    print(bleu4)
    return bleu4


if __name__ == '__main__':
    # load jieba
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
    beam_size = 1
    print("\nBLEU-4 score @ beam size of %d is %.4f.\n" % (beam_size, evaluate(beam_size)))
