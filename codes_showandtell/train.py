import json
import time
import jieba
import numpy as np
import torch

import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from nltk.translate.bleu_score import corpus_bleu
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from dataset import CaptionDataset
from model import Encoder, DecoderWithAttention
from utils import *
from config import *
import warnings
from sklearn import metrics


def main():
    """
    Training and validation.
    """
    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, word_map  #, fine_tune_encoder

    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_unstructed.json')
    with open(word_map_file, 'r', encoding='utf-8') as j:
        word_map = json.load(j)

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
    # Initialize / load checkpoint
    torch.set_num_threads(2)
    if checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout)
        # decoder = nn.DataParallel(decoder)
        # decoder = torch.nn.DataParallel(decoder.cuda(), device_ids=[0, 1, 2, 3])
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = Encoder()
        # fine_tune_encoder = True
        encoder.fine_tune(fine_tune_encoder)
        # encoder = nn.DataParallel(encoder)
        # encoder = torch.nn.DataParallel(encoder.cuda(), device_ids=[0, 1, 2, 3])
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            print('fine_tune_encoder')
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset('train', transform=transform),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset('valid', transform=transform),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train_loss = train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_bleu4, val_loss = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)
        with open('logs.txt', 'a') as f:
            f.writelines(str(train_loss) + '\t' + str(val_loss) + '\t' + str(recent_bleu4) + '\n')
            f.close()


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.
    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()
    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, _, caps, caplens, label) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        # imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)
        # print('caps', caps)
        # print('caplens', caplens)
        # Forward prop.
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # print('scores: ', scores)
        # print('caps_sorted: ', caps_sorted)
        # print('decode_lengths: ', decode_lengths)
        # print('sort_ind', sort_ind)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate loss
        loss_word = criterion(scores.data, targets.data)

        # Add doubly stochastic attention regularization
        loss = loss_word + alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean() 

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores.data, targets.data, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))
    return losses.val


def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # Batches
    for i, (imgs, _, caps, caplens, label) in enumerate(val_loader):

        # Move to device, if available
        # imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        with torch.no_grad():
            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            start_token = torch.LongTensor([[word_map['<start>']]] * imgs.shape[0]).to(device)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder.sample(imgs, caps, caplens, start_token)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            word_loss = criterion(scores.data, targets.data)

            # Add doubly stochastic attention regularization
            loss = word_loss + alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Keep track of metrics
        losses.update(loss.item(), sum(decode_lengths))
        top5 = accuracy(scores.data, targets.data, 5)
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        if i % print_freq == 0:
            print('Validation: [{0}/{1}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                            loss=losses, top5=top5accs))

        # Store references (true captions), and hypothesis (prediction) for each image
        # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
        # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

        # References
        caps = caps[sort_ind]  # because images were sorted in the decoder
        # print(caps.shape)
        for j in range(caps.shape[0]):
            img_caps = caps[j].tolist()
            img_captions = []
            for w in img_caps:
                if w not in {word_map['<start>'], word_map['<pad>']}:
                    img_captions.append(w)  # remove <start> and pads

            # img_captions = list(
            #     map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
            #         img_caps))  # remove <start> and pads
            references.append(img_captions)

        # print(np.expand_dims(references, axis=1))
        # Hypotheses
        _, preds = torch.max(scores_copy, dim=2)
        preds = preds.tolist()
        temp_preds = list()
        for j, p in enumerate(preds):
            temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
        preds = temp_preds
        hypotheses.extend(preds)
        # print(hypotheses)
        assert len(references) == len(hypotheses)

    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(np.expand_dims(references, axis=1), hypotheses)
    print(
        '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
            loss=losses,
            top5=top5accs,
            bleu=bleu4))

    return bleu4, losses.val


if __name__ == '__main__':
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'  # 屏蔽通知信息
    # warnings.filterwarnings(action='once')
    main()

