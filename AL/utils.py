import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from BaseCNN import BaseCNN
from Transformers import AdaptiveResize
import scipy.stats
import numpy as np
from itertools import combinations
import os, random, copy, math
import prettytable as pt
from ImageDataset import QueryLoad
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist
from torchvision import models

random.seed(10)
np.random.seed(10)

def get_features(model, transform, config):
    sampled_feats = []        
    with open(os.path.join(config.trainset_path, 'all_sampled_round{}.txt'.format(config.round))) as rfile:
        lines = rfile.readlines()
        for c1, line in enumerate(lines):
            line_list = line.replace('\n', '').split('\t')
            img_name = line_list[0]
            img = transform(Image.open(os.path.join(config.trainset, img_name)).convert("RGB")).unsqueeze(0).cuda()
            feat = model(img)
            feat_cpu = feat.view(-1).cpu().data.numpy()
            sampled_feats.append(feat_cpu)   
    return sampled_feats
    
def distance(unlabeled_feats, sampled_feats):
    def euler_distance(feat, sampled_feats):
        distance = 0.0
        for item in sampled_feats:
            distance += np.mean((feat-item)*(feat-item))
        return math.sqrt(distance/len(sampled_feats))
    dists = []
    for feat in unlabeled_feats:
        dists.append(euler_distance(feat, sampled_feats))
    return np.array(dists)

def pred_clive_live(model, transform, config):
    preds = []
    moss = []       
    with open(os.path.join(config.trainset_path,'sampled_clive_round0.txt'), 'r') as rfile:
        lines = rfile.readlines()
        for c1, line in enumerate(lines):
            line_list = line.replace('\n', '').split('\t')
            img_name = line_list[0]
            img = transform(Image.open(os.path.join(config.trainset, img_name)).convert("RGB")).unsqueeze(0).cuda()
            _,_,y_ens,_ = model(img)
            preds.append(y_ens.view(-1).cpu().data.numpy()[0])
            moss.append(float(line_list[1]))   
    srcc = scipy.stats.mstats.spearmanr(x=preds, y=moss)[0],
    plcc = scipy.stats.mstats.spearmanr(x=preds, y=moss)[0]
    print('clive SRCC: {}, PLCC: {}'.format(srcc, plcc))

    preds = []
    moss = []       
    with open(os.path.join(config.trainset_path,'sampled_live_round0.txt'), 'r') as rfile:
        lines = rfile.readlines()
        for c1, line in enumerate(lines):
            line_list = line.replace('\n', '').split('\t')
            img_name = line_list[0]
            img = transform(Image.open(os.path.join(config.trainset, img_name)).convert("RGB")).unsqueeze(0).cuda()
            _,_,y_ens,_ = model(img)
            preds.append(y_ens.view(-1).cpu().data.numpy()[0])
            moss.append(float(line_list[1])) 
    srcc = scipy.stats.mstats.spearmanr(x=preds, y=moss)[0],
    plcc = scipy.stats.mstats.spearmanr(x=preds, y=moss)[0]
    print('LIVE SRCC: {}, PLCC: {}'.format(srcc, plcc))
    return 0

    
def predict(ckpt, config, img_file=None, img_path= None, save_path=None, num_heads=8):
    test_transform = transforms.Compose([
            AdaptiveResize(768),
            transforms.RandomCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])
    model = nn.DataParallel(BaseCNN(config).cuda())
    model.load_state_dict(torch.load(ckpt)['state_dict'])
    model.eval()
    # feature diversity、
    vgg = models.vgg19(pretrained=True).features.cuda().eval()
    pred_clive_live(model, test_transform, config)
    sampled_feats = get_features(vgg, test_transform, config)
    #dists = distance(unlabeled_feats, sampled_feats) 
    img_names, moss, mos_stds, unlabeled_feats = [], [], [], []
    with open(os.path.join(save_path,"preds.txt"), 'w') as wfile:
        with open(os.path.join(save_path,'std.txt'), 'w') as stdfile:
            with open(img_file, 'r') as rfile:
                lines = rfile.readlines()
                q_hats, q_ens, stds = np.zeros((len(lines), num_heads)), np.zeros((len(lines))), np.zeros((len(lines)))
                for c1, line in enumerate(lines):
                    line_list = line.replace('\n', '').split('\t')
                    img_name = line_list[0]
                    mos = float(line_list[1])
                    mos_std = float(line_list[2])
                    img = test_transform(Image.open(os.path.join(img_path, img_name)).convert("RGB")).unsqueeze(0).cuda()
                    y_bar, _, y_ens, _ = model(img)
                    feat = vgg(img)
                    feat_cpu = feat.view(-1).cpu().data.numpy()
                    unlabeled_feats.append(feat_cpu)  
                    wfile.write('{},{},{}\n'.format(img_name, mos, y_ens.cpu().data.numpy()[0]))
                    for c2, item in enumerate(y_bar):
                        q_hats[c1, c2] = item.cpu().data.numpy()[0]
                    q_ens[c1] = y_ens.cpu().data.numpy()[0]
                    stds[c1] = np.std(q_hats[c1,:])
                    stdfile.write('{}, {}\n'.format(img_name, stds[c1]))
                    moss.append(mos)
                    mos_stds.append(mos_std)
                    img_names.append(img_name)
                    if c1 % 100 == 0:
                        print("Predictions of unlabeled images at step:", c1)
    return q_ens, stds, np.array(moss), np.array(mos_stds), np.array(img_names), q_hats, unlabeled_feats, sampled_feats

def query_samples(config):
    ##############################################################################################################################
    # predictions of best-performing model on unlabeled dataset
    ##############################################################################################################################
    preds, stds, moss, mos_stds, img_names, q_hats, unlabeled_feats, sampled_feats = predict(os.path.join(config.ckpt_best_path, "best.pt"), 
           config, 
           img_file = config.unlabeled_test,
           img_path= config.trainset,
           save_path = config.trainset_path,
           num_heads = config.num_heads)

    # start greedy sampling
    stds_copy = copy.deepcopy(stds)
    img_names_copy = copy.deepcopy(img_names)
    img_sampled = []
    for step in range(config.num_per_round):
        dists = distance(unlabeled_feats, sampled_feats) 
        # sort from small to large
        sort = np.argsort(stds_copy + 6*dists)
        idx_sampled = sort[-1]
        img_sampled.append(img_names_copy[idx_sampled])
        stds_copy = np.delete(stds_copy, idx_sampled)
        img_names_copy = np.delete(img_names_copy, idx_sampled)
        sampled_feats.append(unlabeled_feats[idx_sampled])
        unlabeled_feats.pop(idx_sampled)
        print("Greedy sampling at step: {}".format(step))
    # index idxs_sampled
    idxs_sampled = []
    for img in img_sampled:
        idxs_sampled.append(img_names.tolist().index(img)) 
    # order from small to large
    idxs_sampled = np.array(idxs_sampled)
    img_sampled = img_names[idxs_sampled]
    mos_sampled = moss[idxs_sampled]
    mos_std_sampled = mos_stds[idxs_sampled]
    pred_sampled = preds[idxs_sampled]
    idxs_kadid = []
    idxs_koniq = []
    for step, img in enumerate(img_sampled.tolist()):
        idxs_kadid.append(True if 'kadid' in img else False)
        idxs_koniq.append(False if 'kadid' in img else True)
    idxs_kadid = np.array(idxs_kadid)
    idxs_koniq = np.array(idxs_koniq)

    srcc = [scipy.stats.mstats.spearmanr(x=mos_sampled[idxs_kadid], y=pred_sampled[idxs_kadid])[0],
            scipy.stats.mstats.spearmanr(x=mos_sampled[idxs_koniq], y=pred_sampled[idxs_koniq])[0]]
    plcc = [scipy.stats.mstats.pearsonr(x=mos_sampled[idxs_kadid], y=pred_sampled[idxs_kadid])[0], 
            scipy.stats.mstats.pearsonr(x=mos_sampled[idxs_koniq], y=pred_sampled[idxs_koniq])[0]]
    img_unlabeled, mos_unlabeled, mos_std_unlabeled = [], [], []
    for img, mos, mos_std in zip(img_names.tolist(), moss.tolist(), mos_stds.tolist()):
        if img not in img_sampled.tolist():
            img_unlabeled.append(img)
            mos_unlabeled.append(mos)
            mos_std_unlabeled.append(mos_std)

    return {"img_sampled": img_sampled, "mos_sampled": mos_sampled, 'mos_std_sampled': mos_std_sampled,\
            "img_unlabeled": np.array(img_unlabeled), "mos_unlabeled": np.array(mos_unlabeled), \
            "mos_std_unlabeled": np.array(mos_std_unlabeled),\
            "pred_sampled": pred_sampled, "srcc_sampled": srcc, "plcc_sampled": plcc}
                 
def init_train(train_path, config):
    imgs = np.loadtxt(train_path, dtype=str, delimiter='\t', usecols=(0))
    moss = np.loadtxt(train_path, dtype=float, delimiter='\t', usecols=(1))
    mos_stds = np.loadtxt(train_path, dtype=float, delimiter='\t', usecols=(2))
    idxs = [i for i in range(len(imgs.tolist()))]
    random.shuffle(idxs)
    idxs = idxs[:config.num_per_round]

    img_sampled, mos_sampled, mos_std_sampled, img_unlabeled, mos_unlabeled , mos_std_unlabeled= [], [], [], [], [], []
    
    for step, (img, mos, mos_std) in enumerate(zip(imgs.tolist(), moss.tolist(), mos_stds.tolist())):
        if step in idxs:
            img_sampled.append(img)
            mos_sampled.append(mos)
            mos_std_sampled.append(mos_std)
        else:
            img_unlabeled.append(img)
            mos_unlabeled.append(mos)
            mos_std_unlabeled.append(mos_std)
    return np.array(img_sampled), np.array(mos_sampled), np.array(mos_std_sampled), \
           np.array(img_unlabeled), np.array(mos_unlabeled), np.array(mos_std_unlabeled)

def pair_wise(img_sampled, mos_sampled, std_sampled, img_unlabeled, num_pairs=5000, train_txt=None):
    img_sampled = img_sampled.tolist()
    mos_sampled = mos_sampled.tolist()
    std_sampled = std_sampled.tolist()
    kadid_idxs = []
    koniq_idxs = []
    # split into koniq and kadid
    for step, (img, mos, std) in enumerate(zip(img_sampled, mos_sampled, std_sampled)):
        if 'ChallengeDB_release' in img or 'kadid10k' in img:
            kadid_idxs.append(step)
        else:
            koniq_idxs.append(step)
    # pairwise of KADID-10k        
    n = len(kadid_idxs)
    combs = combinations([i for i in range(n)], 2)
    comb_lists = []
    for item in combs:
        comb_lists.append(item)
    random.shuffle(comb_lists)
    comb_lists = comb_lists[:num_pairs] if len(comb_lists)>num_pairs else comb_lists
    unlabel_1 = copy.deepcopy(img_unlabeled)
    unlabel_2 = copy.deepcopy(img_unlabeled)
    random.shuffle(unlabel_2)
    with open(train_txt, 'w') as wfile:    
        # pairewise training data
        for step, (i, j) in enumerate(comb_lists):
            img1, img2 = img_sampled[kadid_idxs[i]], img_sampled[kadid_idxs[j]]
            diff = float(mos_sampled[kadid_idxs[i]]) - float(mos_sampled[kadid_idxs[j]])
            sq = np.sqrt(float(std_sampled[kadid_idxs[i]])*float(std_sampled[kadid_idxs[i]]) \
                       + float(std_sampled[kadid_idxs[j]])*float(std_sampled[kadid_idxs[j]])) + 1e-8
            prob_label = 0.5 * (1 + math.erf(diff / sq))
            binary_label = 1 if mos_sampled[kadid_idxs[i]]>mos_sampled[kadid_idxs[j]] else 0
            un_img1, un_img2 = unlabel_1[step%len(img_unlabeled)], unlabel_2[step%len(img_unlabeled)]
            wstr = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(img1,img2,un_img1,un_img2,binary_label,
                                            std_sampled[kadid_idxs[i]],std_sampled[kadid_idxs[j]],binary_label)
            wfile.write(wstr)
    # pairwise of KonIQ-10k       
    n = len(koniq_idxs)
    combs = combinations([i for i in range(n)], 2)
    comb_lists = []
    for item in combs:
        comb_lists.append(item)
    random.shuffle(comb_lists)
    comb_lists = comb_lists[:num_pairs] if len(comb_lists)>num_pairs else comb_lists
    unlabel_1 = copy.deepcopy(img_unlabeled)
    unlabel_2 = copy.deepcopy(img_unlabeled)
    random.shuffle(unlabel_2)
    with open(train_txt, 'a') as wfile:    
        # pairewise training data
        for step, (i, j) in enumerate(comb_lists):
            img1, img2 = img_sampled[koniq_idxs[i]], img_sampled[koniq_idxs[j]]
            diff = float(mos_sampled[koniq_idxs[i]]) - float(mos_sampled[koniq_idxs[j]])
            sq = np.sqrt(float(std_sampled[koniq_idxs[i]])*float(std_sampled[koniq_idxs[i]]) \
                       + float(std_sampled[koniq_idxs[j]])*float(std_sampled[koniq_idxs[j]])) + 1e-8
            prob_label = 0.5 * (1 + math.erf(diff / sq))
            binary_label = 1 if mos_sampled[koniq_idxs[i]]>mos_sampled[koniq_idxs[j]] else 0
            un_img1, un_img2 = unlabel_1[step%len(img_unlabeled)], unlabel_2[step%len(img_unlabeled)]
            wstr = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(img1,img2,un_img1,un_img2, prob_label,\
                                 std_sampled[koniq_idxs[i]],std_sampled[koniq_idxs[j]],binary_label)
            wfile.write(wstr)
    return 0


def print_tb(srcc, plcc, n):
    # evaluate after every epoch
    tb = pt.PrettyTable()
    tb.field_names = ["Model1", "KONIQ_VALID", "KONIQ_TEST", "KONIQ_ACTIVE", "KADID_VALID", "KADID_TEST", "KADID_ACTIVE"]
    tb.add_row(['SRCC', srcc["koniq10k_valid"]['model{}'.format(0)], srcc["koniq10k_test"]['model{}'.format(0)],\
                        srcc["koniq10k_active"]['model{}'.format(0)], srcc["kadid10k_valid"]['model{}'.format(0)],\
                        srcc["kadid10k_test"]['model{}'.format(0)], srcc["kadid10k_active"]['model{}'.format(0)]])
    tb.add_row(['PLCC', plcc["koniq10k_valid"]['model{}'.format(0)], plcc["koniq10k_test"]['model{}'.format(0)], \
                        plcc["koniq10k_active"]['model{}'.format(0)], plcc["kadid10k_valid"]['model{}'.format(0)],\
                        plcc["kadid10k_test"]['model{}'.format(0)], plcc["kadid10k_active"]['model{}'.format(0)]])
    
    for i in range(n-1): # do not include head1 and ensemble
        tb.add_row(["Model{}".format(i+2), "KONIQ_VALID", "KONIQ_TEST", "KONIQ_ACTIVE", "KADID_VALID", "KADID_TEST", "KADID_ACTIVE"])
        tb.add_row(['SRCC', srcc["koniq10k_valid"]['model{}'.format(i+1)], srcc["koniq10k_test"]['model{}'.format(i+1)], \
                            srcc["koniq10k_active"]['model{}'.format(i+1)], srcc["kadid10k_valid"]['model{}'.format(i+1)],\
                            srcc["kadid10k_test"]['model{}'.format(i+1)], srcc["kadid10k_active"]['model{}'.format(i+1)],])
        tb.add_row(['PLCC', plcc["koniq10k_valid"]['model{}'.format(i+1)], plcc["koniq10k_test"]['model{}'.format(i+1)], \
                            plcc["koniq10k_active"]['model{}'.format(i+1)], plcc["kadid10k_valid"]['model{}'.format(i+1)],\
                            plcc["kadid10k_test"]['model{}'.format(i+1)], plcc["kadid10k_active"]['model{}'.format(i+1)]])

    tb.add_row(["Ensemble", "KONIQ_VALID", "KONIQ_TEST", "KONIQ_ACTIVE", "KADID_VALID", "KADID_TEST", "KADID_ACTIVE"])
    tb.add_row(['SRCC', srcc["koniq10k_valid"]['ensemble'], srcc["koniq10k_test"]['ensemble'], \
                        srcc["koniq10k_active"]['ensemble'], srcc["kadid10k_valid"]['ensemble'],\
                        srcc["kadid10k_test"]['ensemble'], srcc["kadid10k_active"]['ensemble']])
    tb.add_row(['PLCC', plcc["koniq10k_valid"]['ensemble'], plcc["koniq10k_test"]['ensemble'], \
                        plcc["koniq10k_active"]['ensemble'], plcc["kadid10k_valid"]['ensemble'],  \
                        plcc["kadid10k_test"]['ensemble'], plcc["kadid10k_active"]['ensemble']])
    return tb