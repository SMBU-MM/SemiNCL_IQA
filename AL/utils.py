import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from BaseCNN import BaseCNN
from Transformers import AdaptiveResize
import scipy.stats
import numpy as np
from itertools import combinations
import os, random, copy
import prettytable as pt

def prediction_diversity_driven(ckpt, config, img_file=None, img_path= None, save_path=None, num_heads=8):
    test_transform = transforms.Compose([
            AdaptiveResize(768),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])
    model = nn.DataParallel(BaseCNN(config).cuda())
    model.load_state_dict(torch.load(ckpt)['state_dict'])
    model.eval()
    img_names, moss = [], []
    with open(os.path.join(save_path,"preds.txt"), 'w') as wfile:
        with open(os.path.join(save_path,'std.txt'), 'w') as stdfile:
            with open(img_file, 'r') as rfile:
                lines = rfile.readlines()
                q_hats, q_ens, stds = np.zeros((len(lines), num_heads)), np.zeros((len(lines))), np.zeros((len(lines)))
                for c1, line in enumerate(lines):
                    line_list = line.replace('\n', '').split('\t')
                    img_name = line_list[0]
                    mos = float(line_list[1])
                    img = test_transform(Image.open(os.path.join(img_path, img_name)).convert("RGB")).unsqueeze(0).cuda()
                    y_bar, _, y_ens, _ = model(img)
      
                    wfile.write('{},{},{}\n'.format(img_name, mos, y_ens.cpu().data.numpy()[0]))
                    for c2, item in enumerate(y_bar):
                        q_hats[c1, c2] = item.cpu().data.numpy()[0]
                    q_ens[c1] = y_ens.cpu().data.numpy()[0]
                    stds[c1] = np.std(q_hats[c1,:])
                    stdfile.write('{}, {}\n'.format(img_name, stds[c1]))
                    moss.append(mos)
                    img_names.append(img_name)
    return q_ens, stds, np.array(moss), np.array(img_names)

def sampling_diversity_driven(config):
    ##############################################################################################################################
    # predictions of best-performing model on unlabeled dataset
    ##############################################################################################################################
    preds, stds, moss, img_names = prediction_diversity_driven(os.path.join(config.ckpt_best_path, "best.pt"), config, 
           img_file = config.unlabeled_test,
           img_path= config.unlabeled_set,
           save_path = config.trainset_path,
           num_heads = config.num_heads)
    # start sampling
    sort = np.argsort(stds)
    # order from small to large
    idxs = sort[-1*config.num_per_round:]
    img_sampling = img_names[idxs]
    mos_sampling = moss[idxs]
    pred_sampling = preds[idxs]
    srcc = scipy.stats.mstats.spearmanr(x=mos_sampling, y=pred_sampling)[0] 
    plcc = scipy.stats.mstats.pearsonr(x=mos_sampling, y=pred_sampling)[0]
    img_unlabeled, mos_unlabeled = [], []
    for img, mos in zip(img_names.tolist(), moss.tolist()):
        if img not in img_sampling.tolist():
            img_unlabeled.append(img)
            mos_unlabeled.append(mos)
    return {"img_sampling": img_sampling, "mos_sampling": mos_sampling, \
            "img_unlabeled": np.array(img_unlabeled), "mos_unlabeled": np.array(mos_unlabeled),\
            "pred_sampling": pred_sampling, "srcc_sampling": srcc, "plcc_sampling": plcc}

def pair_wise(img_sampling, mos_sampling, img_unlabeled, path='spaq', num_pairs=20000, train_txt=None):
    img_sampling = img_sampling.tolist()
    mos_sampling = mos_sampling.tolist()
    n = len(img_sampling)
    combs = combinations([i for i in range(n)], 2)
    comb_lists = []
    for item in combs:
        comb_lists.append(item)
    random.shuffle(comb_lists)
    comb_lists = comb_lists[:num_pairs] if len(comb_lists)>num_pairs else comb_lists
    unlabel_1 = copy.deepcopy(img_unlabeled)
    unlabel_2 = copy.deepcopy(img_unlabeled)
    random.shuffle(unlabel_2)
    with open(train_txt, 'r') as rfile:
        # replace the unlabeled data if sampling for training
        lines = rfile.readlines()
    with open(train_txt, 'w') as wfile:    
        for step, line in enumerate(lines):
            line_list = line.replace("\n", "").split("\t") 
            un_img1 = unlabel_1[step%len(img_unlabeled)]
            un_img2 = unlabel_2[step%len(img_unlabeled)]
            wstr = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(line_list[0], line_list[1], 
                                                             os.path.join(path, un_img1), 
                                                             os.path.join(path, un_img2),
                                                             line_list[4], line_list[5],
                                                             line_list[6], line_list[7])
            wfile.write(wstr)    
        # append the new sampling data 
        for step, (i, j) in enumerate(comb_lists):
            img1 = img_sampling[i]
            img2 = img_sampling[j]
            binary_label = 1 if mos_sampling[i]>mos_sampling[j] else 0
            un_img1 = unlabel_1[step%len(img_unlabeled)]
            un_img2 = unlabel_2[step%len(img_unlabeled)]
            wstr = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(os.path.join(path, img1), 
                                                             os.path.join(path, img2),
                                                             os.path.join(path, un_img1), 
                                                             os.path.join(path, un_img2),
                                                             binary_label,0,0,binary_label)
            wfile.write(wstr)

def print_tb(srcc, plcc, n):
    # evaluate after every epoch
    tb = pt.PrettyTable()
    tb.field_names = ["Model1", "SPAQ_TEST", "KONIQ_TEST", "SPAQ_VALID", "KONIQ_ACTIVE"]
    tb.add_row(['SRCC', srcc["spaq_test"]['model{}'.format(0)], srcc["koniq10k_test"]['model{}'.format(0)],\
                        srcc["spaq_valid"]['model{}'.format(0)], srcc["koniq10k_active"]['model{}'.format(0)]])
    tb.add_row(['PLCC', plcc["spaq_test"]['model{}'.format(0)], plcc["koniq10k_test"]['model{}'.format(0)], \
                        plcc["spaq_valid"]['model{}'.format(0)], plcc["koniq10k_active"]['model{}'.format(0)]])
    
    for i in range(n-1): # do not include head1 and ensemble
        tb.add_row(["Model{}".format(i+2),  "SPAQ_TEST", "KONIQ_TEST", "SPAQ_ACTIVE", "KONIQ_VALID"])
        tb.add_row(['SRCC', srcc["spaq_test"]['model{}'.format(i+1)], srcc["koniq10k_test"]['model{}'.format(i+1)], \
                            srcc["spaq_valid"]['model{}'.format(i+1)], srcc["koniq10k_active"]['model{}'.format(i+1)]])
        tb.add_row(['PLCC', plcc["spaq_test"]['model{}'.format(i+1)], plcc["koniq10k_test"]['model{}'.format(i+1)], \
                            plcc["spaq_valid"]['model{}'.format(i+1)], plcc["koniq10k_active"]['model{}'.format(i+1)]])

    tb.add_row(["Ensemble", "SPAQ_TEST", "KONIQ_TEST", "SPAQ_ACTIVE", "KONIQ_VALID"])
    tb.add_row(['SRCC', srcc["spaq_test"]['ensemble'], srcc["koniq10k_test"]['ensemble'], \
                        srcc["spaq_valid"]['ensemble'], srcc["koniq10k_active"]['ensemble']])
    tb.add_row(['PLCC', plcc["spaq_test"]['ensemble'], plcc["koniq10k_test"]['ensemble'], \
                        plcc["spaq_valid"]['ensemble'], plcc["koniq10k_active"]['ensemble']])
    return tb