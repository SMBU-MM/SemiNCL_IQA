import argparse
import TrainModel
import os
import random
import numpy as np
from shutil import copyfile
from utils import sampling_diversity_driven, pair_wise


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_idx", type=str, default='4')
    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=19901116)
    parser.add_argument("--fz", type=bool, default=True)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--loss", type=str, default='semi_ncl') # ind | ncl | semi_ncl
    parser.add_argument("--unlabeled", type=str, default='koniq-10k') # koniq-10k | spaq 
    parser.add_argument("--train_txt", type=str, default="train_spaq_koniq_active.txt") # 

    parser.add_argument("--num_per_round", type=int, default=200) # the number of images selected in each round for active learning 
    parser.add_argument("--weight_ind", type=float, default=1.0)
    parser.add_argument("--weight_udiv", type=float, default=0.06)

    parser.add_argument("--split", type=int, default=2)
    parser.add_argument("--path_idx", type=int, default=7)
    parser.add_argument("--trainset", type=str, default="../IQA_database/")
    parser.add_argument("--spaq_set", type=str, default="../IQA_database/spaq/")
    parser.add_argument("--livec_set", type=str, default="../IQA_database/ChallengeDB_release/")
    parser.add_argument("--bid_set", type=str, default="../IQA_database/BID/")
    parser.add_argument("--koniq10k_set", type=str, default="../IQA_database/koniq-10k/")
    
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--batch_size2", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=384, help='None means random resolution')
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--max_epochs2", type=int, default=12)
    parser.add_argument("--num_round", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay_interval", type=int, default=1)
    parser.add_argument("--decay_ratio", type=float, default=0.5)
    parser.add_argument("--epochs_per_eval", type=int, default=1)
    parser.add_argument("--epochs_per_save", type=int, default=1)
    return parser.parse_args()

def copycodes(path):
    # create file folder if not exist
    if not os.path.exists(config.codes):
        os.makedirs(config.codes)
    # save the code
    copyfile('Main.py', os.path.join(path, 'Main.py'))
    copyfile('ImageDataset.py', os.path.join(path, 'ImageDataset.py'))
    copyfile('BaseCNN.py', os.path.join(path, 'BaseCNN.py'))
    copyfile('TrainModel.py', os.path.join(path, 'TrainModel.py'))
    copyfile('Transformers.py', os.path.join(path, 'Transformers.py'))
    copyfile('MNL_Loss.py', os.path.join(path, 'MNL_Loss.py'))
    copyfile('utils.py', os.path.join(path, 'utils.py'))
    return 0

def main(cfg):
    t = TrainModel.Trainer(cfg)
    t.fit()
    
if __name__ == "__main__":
    config = parse_config()
    # set the seed of random 
    np.random.seed(config.seed)
    random.seed(config.seed)

    config.split = 3
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_idx
    print("[*] gpu index: {}".format(config.gpu_idx))
    # construct the path of checkpoint
    config.ckpt = r"./{}_".format(config.loss) + \
                  config.train_txt.replace(".txt", "") +\
                  "_head{}_ind{}_udiv{}_num{}/".format(config.num_heads, \
                  config.weight_ind, config.weight_udiv, config.num_per_round)

    if not os.path.exists(config.ckpt): os.makedirs(config.ckpt)
    for i in range(config.num_round+1):
        if not os.path.exists(os.path.join(config.ckpt, "{}".format(i))):
            os.makedirs(os.path.join(config.ckpt, "{}".format(i)))
    # save the codes
    config.codes = os.path.join(config.ckpt, 'codes')
    copycodes(config.codes)
    
    for i in range(config.num_round):
        config.round = i
        #########################################################################################################################
        # create file folder
        #########################################################################################################################
        config.ckpt_path = os.path.join(config.ckpt, "{}".format(i), 'checkpoint')
        config.result_path = os.path.join(config.ckpt, "{}".format(i), 'results')
        config.p_path = os.path.join(config.ckpt, "{}".format(i), 'p')
        config.runs_path = os.path.join(config.ckpt, "{}".format(i), 'runs')
        config.ckpt_best_path = os.path.join(config.ckpt, "{}".format(i), 'best')
        config.trainset_path = os.path.join(config.ckpt, "{}".format(i), 'train_path')
        if not os.path.exists(config.ckpt_path):
            os.makedirs(config.ckpt_path)
        if not os.path.exists(config.result_path):
            os.makedirs(config.result_path)
        if not os.path.exists(config.p_path):
            os.makedirs(config.p_path)
        if not os.path.exists(config.runs_path):
            os.makedirs(config.runs_path)
        if not os.path.exists(config.ckpt_best_path):
            os.makedirs(config.ckpt_best_path)
        if not os.path.exists(config.trainset_path):
            os.makedirs(config.trainset_path)
        ############################################################################################################################
        # train model
        ############################################################################################################################
        # copy the train_txt and flive_test in 1st round
        if i == 0:
            copyfile(os.path.join(config.trainset, 'splits{}'.format(config.path_idx), str(3), config.train_txt),\
                     os.path.join(config.trainset_path, 'train_txt_round0.txt'))
            copyfile(os.path.join("../IQA_database/{}/splits{}/3/".format(config.unlabeled, config.path_idx), "{}_train_score.txt".format('koniq10k')),
                     os.path.join(config.trainset_path, 'unlabeled_round0.txt'))
        config.train_file = os.path.join(config.trainset_path, 'train_txt_round{}.txt'.format(i)) 
        config.unlabeled_test = os.path.join(config.trainset_path, 'unlabeled_round{}.txt'.format(i))
        config.unlabeled_set = config.spaq_set if config.unlabeled == 'spaq' else config.koniq10k_set
        #############################################################################################################################
        # retrain the model from scratch
        #############################################################################################################################
        # stage1: freezing previous layers, training fc in the first round
        config.fz = True
        config.resume = False  # resuming from the latest checkpoint of stage 1
        config.max_epochs = 3
        config.batch_size = 96
        main(config) 
        # stage2: fine-tuning the whole network
        config.fz = False
        config.resume = True  # resuming from the latest checkpoint of stage 1
        config.max_epochs = 12
        config.batch_size = 12
        main(config)
       
        ##############################################################################################################################
        # active sampling via QBC
        ##############################################################################################################################
        print("[*] Start Active Learning")
        ret_sampling = sampling_diversity_driven(config)
        # save the unlabeled test file for next round
        trainset_path_next = os.path.join(config.ckpt,"{}".format(i+1), 'train_path')
        if not os.path.exists(trainset_path_next):
            os.makedirs(trainset_path_next)
        # save the prediction, srcc, plcc of the images were sampled in each round for checking
        with open(os.path.join(trainset_path_next, 'sampling_round{}.txt'.format(i+1)), "w") as wfile:
            for img, mos, pred in zip(ret_sampling['img_sampling'].tolist(), ret_sampling['mos_sampling'].tolist(), ret_sampling['pred_sampling'].tolist()):
                wfile.write("{}\t{}\t{}\n".format(img, mos, pred))
        with open(os.path.join(trainset_path_next, 'sampling_srcc_plcc{}.txt'.format(i+1)), "w") as wfile:
            wfile.write("SRCC: {}\nPLCC:{}\n".format(ret_sampling['srcc_sampling'], ret_sampling['plcc_sampling']))
        ################################################################################################################################ 
        # append new labeled image for training
        ################################################################################################################################
        with open(os.path.join(trainset_path_next, 'unlabeled_round{}.txt'.format(i+1)), "w") as wfile:
            for img, mos in zip(ret_sampling['img_unlabeled'].tolist(), ret_sampling['mos_unlabeled'].tolist()):
                wfile.write("{}\t{}\n".format(img, mos))
        # copy old train_file into the next round path
        copyfile(os.path.join(config.trainset_path, 'train_txt_round{}.txt'.format(i)),\
                 os.path.join(trainset_path_next, 'train_txt_round{}.txt'.format(i+1)))
        # append new pairs in the new train_txt
        pair_wise(ret_sampling['img_sampling'], ret_sampling['mos_sampling'], ret_sampling['img_unlabeled'],
                  path = config.unlabeled, num_pairs=20000, train_txt = os.path.join(trainset_path_next, 'train_txt_round{}.txt'.format(i+1)))







