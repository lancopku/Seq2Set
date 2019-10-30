import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
import models
import data.dataloader as dataloader
import data.utils as utils
import data.dict as dict
from optims import Optim

from optims import Optim
import lr_scheduler as L

import os
import argparse
import time
import math
import json
import collections
import codecs
import numpy as np
from sklearn import metrics

#config
parser = argparse.ArgumentParser(description='train.py')
parser.add_argument('-config', default='config.yaml', type=str,
                    help="config file")
parser.add_argument('-gpus', default=[], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")
parser.add_argument('-restore', default='./data/data/log/f1/best_micro_f1_checkpoint.pt', type=str,
                    help="restore checkpoint")
parser.add_argument('-seed', type=int, default=1234,
                    help="Random seed")
parser.add_argument('-model', default='seq2seq', type=str,
                    help="Model selection")
parser.add_argument('-score', default='', type=str,
                    help="score_fn")
parser.add_argument('-pretrain', action='store_true',
                    help="load pretrain embedding")
parser.add_argument('-limit', type=int, default=0,
                    help="data limit")
parser.add_argument('-log', default='predict', type=str,
                    help="log directory")
parser.add_argument('-unk', action='store_true',
                    help="replace unk")
parser.add_argument('-memory', action='store_true',
                    help="memory efficiency")
parser.add_argument('-beam_size', type=int, default=1,
                    help="beam search size")
parser.add_argument('-label_dict_file', default='./data/data/topic_sorted.json', type=str,
                    help="label_dict")

opt = parser.parse_args()
config = utils.read_config(opt.config)
torch.manual_seed(opt.seed)


def eval_metrics(reference, candidate, label_dict, log_path):
    ref_dir = log_path + 'reference/'
    cand_dir = log_path + 'candidate/'
    if not os.path.exists(ref_dir):
        os.mkdir(ref_dir)
    if not os.path.exists(cand_dir):
        os.mkdir(cand_dir)
    ref_file = ref_dir+'reference'
    cand_file = cand_dir+'candidate'

    for i in range(len(reference)):
        with codecs.open(ref_file+str(i), 'w', 'utf-8') as f:
            f.write("".join(reference[i])+'\n')
        with codecs.open(cand_file+str(i), 'w', 'utf-8') as f:
            f.write("".join(candidate[i])+'\n')
        
    def make_label(l, label_dict):
        length = len(label_dict)
        result = np.zeros(length)
        indices = [label_dict.get(label.strip().upper(), 0) for label in l]
        result[indices] = 1
        return result

    def prepare_label(y_list, y_pre_list, label_dict):
        reference = np.array([make_label(y, label_dict) for y in y_list])
        candidate = np.array([make_label(y_pre, label_dict) for y_pre in y_pre_list])
        return reference[:, :], candidate[:, :]

    def get_one_error(y, candidate, label_dict):
        idx = [label_dict.get(c[0].strip().upper(), 0) for c in candidate]
        result = [(y[i, idx[i]] == 1) for i in range(len(idx))]
        return (1 - np.array(result).mean())


    def get_metrics(y, y_pre):
        subset_0_1_loss = metrics.zero_one_loss(y, y_pre)
        hamming_loss = metrics.hamming_loss(y, y_pre)
        macro_f1 = metrics.f1_score(y, y_pre, average='macro')
        macro_precision = metrics.precision_score(y, y_pre, average='macro')
        macro_recall = metrics.recall_score(y, y_pre, average='macro')
        micro_f1 = metrics.f1_score(y, y_pre, average='micro')
        micro_precision = metrics.precision_score(y, y_pre, average='micro')
        micro_recall = metrics.recall_score(y, y_pre, average='micro')
        return subset_0_1_loss, hamming_loss, macro_f1, macro_precision, macro_recall, micro_f1, micro_precision, micro_recall


    y, y_pre = prepare_label(reference, candidate, label_dict)
    #one_error = get_one_error(y, candidate, label_dict)
    subset_0_1_loss, hamming_loss, macro_f1, macro_precision, macro_recall, micro_f1, micro_precision, micro_recall = get_metrics(y, y_pre)
    return {'subset_0_1_loss': subset_0_1_loss,
            'hamming_loss': hamming_loss, 
            'macro_f1': macro_f1,
            'macro_precision': macro_precision, 
            'macro_recall': macro_recall,
            'micro_f1': micro_f1,
            'micro_precision': micro_precision, 
            'micro_recall': micro_recall}

#checkpoint
if opt.restore:
    print('loading checkpoint...\n')
    checkpoints = torch.load(opt.restore)
    config = checkpoints['config']
    config.mask = True

# cuda
#use_cuda = torch.cuda.is_available() and len(opt.gpus) > 0
use_cuda = True
if use_cuda:
    #torch.cuda.set_device(opt.gpus[0])
    torch.cuda.manual_seed(opt.seed)

#data
print('loading data...\n')
start_time = time.time()
datas = torch.load(config.data)
print('loading time cost: %.3f' % (time.time()-start_time))

testset = datas['valid']

src_vocab, tgt_vocab = datas['dicts']['src'], datas['dicts']['tgt']
config.src_vocab = src_vocab.size()
config.tgt_vocab = tgt_vocab.size()
testloader = dataloader.get_loader(testset, batch_size=config.batch_size, shuffle=False, num_workers=2)

if opt.pretrain:
    pretrain_embed = torch.load(config.emb_file)
else:
    pretrain_embed = None
# model
print('building model...\n')
model = getattr(models, opt.model)(config, src_vocab.size(), tgt_vocab.size(), use_cuda,
                       pretrain=pretrain_embed, score_fn=opt.score)

if opt.restore:
    model.load_state_dict(checkpoints['model'])
if use_cuda:
    model.cuda()
if len(opt.gpus) > 1:
    model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)

# optimizer
if opt.restore:
    optim = checkpoints['optim']
else:
    optim = Optim(config.optim, config.learning_rate, config.max_grad_norm,
                  lr_decay=config.learning_rate_decay, start_decay_at=config.start_decay_at)
optim.set_parameters(model.parameters())
if config.schedule:
    scheduler = L.CosineAnnealingLR(optim.optimizer, T_max=config.epoch)

param_count = 0
for param in model.parameters():
    param_count += param.view(-1).size()[0]

if not os.path.exists(config.log):
    os.mkdir(config.log)
if opt.log == '':
    log_path = config.log + utils.format_time(time.localtime()) + '/'
else:
    log_path = config.log + opt.log + '/'
if not os.path.exists(log_path):
    os.mkdir(log_path)
logging = utils.logging(log_path+'model_config.txt')  
logging_csv = utils.logging_csv(log_path+'record.csv') 
for k, v in config.items():
    logging("%s:\t%s\n" % (str(k), str(v)))
logging("\n")
logging(repr(model)+"\n\n")  

logging('total number of parameters: %d\n\n' % param_count)

if opt.restore:
    updates = checkpoints['updates']
else:
    updates = 0

with open(opt.label_dict_file, 'r') as f:
    label_dict = json.load(f)

def clean_s(ss):
	data = [s.item() for s in ss]
	return data 

def eval(epoch):
    model.eval()
    reference, candidate, source, alignments = [], [], [], []
    for raw_src, src, src_len, raw_tgt, tgt, tgt_len in testloader:
        if len(opt.gpus) > 1:
            samples, alignment = model.module.sample(src, src_len)
        else:
            if config.beam_size > 1:
                samples, alignment = model.beam_sample(src, src_len, beam_size=config.beam_size)
            else:
                samples, alignment = model.sample(src, src_len)
        print(samples)
        candidate += [tgt_vocab.convertToLabels(clean_s(s), dict.EOS) for s in samples]
        source += raw_src
        reference += raw_tgt
        alignments += [align for align in alignment]
        
    if opt.unk:
        cands = []
        for s, c, align in zip(source, candidate, alignments):
            cand = []
            for word, idx in zip(c, align):
                if word == dict.UNK_WORD and idx < len(s):
                    try:
                        cand.append(s[idx])
                    except:
                        cand.append(word)
                        print("%d %d\n" % (len(s), idx))
                else:
                    cand.append(word)
            cands.append(cand)
        candidate = cands
    score = {}
    result = eval_metrics(reference, candidate, label_dict, log_path)
    logging_csv([result['subset_0_1_loss'], result['hamming_loss'], \
                result['macro_f1'], result['macro_precision'], result['macro_recall'],\
                result['micro_f1'], result['micro_precision'], result['micro_recall']])
    print('subset_0_1_loss: %.4f | hamming_loss: %.8f | macro_f1: %.4f | micro_f1: %.4f'
          % (result['subset_0_1_loss'], result['hamming_loss'], result['macro_f1'], result['micro_f1']))
    

if __name__ == '__main__':
    eval(0)
