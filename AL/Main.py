import argparse
import TrainModel
import os
import random
import numpy as np
from shutil import copyfile
from utils import query_samples, pair_wise, init_train

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_idx", type=str, default='1')
    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=19901116)
    parser.add_argument("--fz", type=bool, default=True)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--loss", type=str, default='ssl') # naive | joint | ind_div | ssl
    parser.add_argument("--unlabeled", type=str, default='koniq-10k') # koniq-10k | spaq 
    parser.add_argument("--train_txt", type=str, default="train.txt") # 

    parser.add_argument("--num_per_round", type=int, default=200) # the number of images selected in each round for active learning 
    parser.add_argument("--weight_ind", type=float, default=1.0)
    parser.add_argument("--weight_div", type=float, default=0.06)

    parser.add_argument("--split", type=int, default=3)
    parser.add_argument("--path_idx", type=int, default=10)
    parser.add_argument("--trainset", type=str, default="/home/zhihua/Active_learning_head4/IQA_database/")
    parser.add_argument("--spaq_set", type=str, default="/home/zhihua/Active_learning_head4/IQA_database/spaq/")
    parser.add_argument("--livec_set", type=str, default="/home/zhihua/Active_learning_head4/IQA_database/ChallengeDB_release/")
    parser.add_argument("--bid_set", type=str, default="/home/zhihua/Active_learning_head4/IQA_database/BID/")
    parser.add_argument("--koniq10k_set", type=str, default="/home/zhihua/Active_learning_head4/IQA_database/koniq-10k/")
    parser.add_argument("--kadid10k_set", type=str, default="/home/zhihua/Active_learning_head4/IQA_database/kadid10k/")
    
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--batch_size2", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=384, help='None means random resolution')
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--max_epochs2", type=int, default=12)
    parser.add_argument("--num_round", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay_interval", type=int, default=3)
    parser.add_argument("--decay_ratio", type=float, default=0.1)
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
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_idx
    print("[*] gpu index: {}".format(config.gpu_idx))
    # construct the path of checkpoint
    config.ckpt = r"./{}_".format(config.loss) + \
                  config.train_txt.replace(".txt", "") +\
                  "{}_head{}_udiv{}_num{}/".format(config.split, config.num_heads, \
                  config.weight_ind, config.num_per_round)

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
            labeled_path = os.path.join("/home/zhihua/Active_learning_head4/IQA_database/splits{}/{}/".format(config.path_idx, config.split),"clive_live.txt")
            img_sampled = np.loadtxt(labeled_path, dtype=str, skiprows=0, comments='#', delimiter='\t', usecols=0)
            mos_sampled = np.loadtxt(labeled_path, dtype=float, skiprows=0, comments='#', delimiter='\t', usecols=1)
            mos_std_sampled = np.loadtxt(labeled_path, dtype=float, skiprows=0, comments='#', delimiter='\t', usecols=2)
            
            with open(os.path.join(config.trainset_path,'sampled_round0.txt'), 'w') as wfile:
                with open(os.path.join(config.trainset_path,'sampled_clive_round0.txt'), 'w') as wofile:
                    with open(os.path.join(config.trainset_path,'sampled_live_round0.txt'), 'w') as wafile:
                        for img, mos, mos_std in zip(img_sampled.tolist(), mos_sampled.tolist(), mos_std_sampled.tolist()):
                            if 'ChallengeDB_release' in img:
                                wofile.write("{}\t{}\t{}\n".format(img, mos, mos_std))
                            else:
                                wafile.write("{}\t{}\t{}\n".format(img, mos, mos_std))
                            wfile.write("{}\t{}\t{}\n".format(img, mos, mos_std))

            with open(os.path.join(config.trainset_path,'all_sampled_round0.txt'), 'w') as wfile:
                with open(os.path.join(config.trainset_path,'all_clive_sampled_round0.txt'), 'w') as wofile:
                    with open(os.path.join(config.trainset_path,'all_live_sampled_round0.txt'), 'w') as wafile:
                        for img, mos, mos_std in zip(img_sampled.tolist(), mos_sampled.tolist(), mos_std_sampled.tolist()):
                            if 'ChallengeDB_release' in img:
                                wofile.write("{}\t{}\t{}\n".format(img, mos, mos_std))
                            else:
                                wafile.write("{}\t{}\t{}\n".format(img, mos, mos_std))
                            wfile.write("{}\t{}\t{}\n".format(img, mos, mos_std))

            unlabeled_path = os.path.join("/home/zhihua/Active_learning_head4/IQA_database/splits{}/{}/".format(config.path_idx, config.split),"train_score.txt")
            img_unlabeled = np.loadtxt(unlabeled_path, dtype=str, skiprows=0, comments='#', delimiter='\t', usecols=0)
            mos_unlabeled = np.loadtxt(unlabeled_path, dtype=float, skiprows=0, comments='#', delimiter='\t', usecols=1)
            mos_std_unlabeled = np.loadtxt(unlabeled_path, dtype=float, skiprows=0, comments='#', delimiter='\t', usecols=2)
            with open(os.path.join(config.trainset_path,'unlabeled_round0.txt'), 'w') as wfile:
                with open(os.path.join(config.trainset_path,'unlabeled_koniq_round0.txt'), 'w') as wofile:
                    with open(os.path.join(config.trainset_path,'unlabeled_kadid_round0.txt'), 'w') as wafile:
                        for img, mos, mos_std in zip(img_unlabeled.tolist(), mos_unlabeled.tolist(), mos_std_unlabeled.tolist()):
                            if 'koniq-10k' in img:
                                wofile.write("{}\t{}\t{}\n".format(img, mos, mos_std))
                            else:
                                wafile.write("{}\t{}\t{}\n".format(img, mos, mos_std))
                            wfile.write("{}\t{}\t{}\n".format(img, mos, mos_std))
                    
            pair_wise(img_sampled, mos_sampled, mos_std_sampled, img_unlabeled, num_pairs=(i+1)*50000, \
                      train_txt = os.path.join(config.trainset_path, 'train_round{}.txt'.format(i)))
        config.train_file = os.path.join(config.trainset_path, 'train_round{}.txt'.format(i)) 
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
        config.batch_size = 16
        main(config)
        ##############################################################################################################################
        # active sampled via QBC
        ##############################################################################################################################
        print("[*] Start Active Learning")
        ret = query_samples(config)
        # save the unlabeled test file for next round
        trainset_path_next = os.path.join(config.ckpt,"{}".format(i+1), 'train_path')
        if not os.path.exists(trainset_path_next):
            os.makedirs(trainset_path_next)
        # save the prediction, srcc, plcc of the images were sampled in each round for checking
        with open(os.path.join(trainset_path_next, 'sampled_round{}.txt'.format(i+1)), "w") as wfile:
            with open(os.path.join(trainset_path_next, 'sampled_koniq_round{}.txt'.format(i+1)), "w") as wofile:
                with open(os.path.join(trainset_path_next, 'sampled_kadid_round{}.txt'.format(i+1)), "w") as wafile:
                    for img, mos, mos_std, pred in zip(ret['img_sampled'].tolist(), ret['mos_sampled'].tolist(),\
                                                       ret['mos_std_sampled'].tolist(), ret['pred_sampled'].tolist()):
                        if 'koniq-10k' in img:
                            wofile.write("{}\t{}\t{}\t{}\n".format(img, mos, mos_std, pred))
                        else:
                            wafile.write("{}\t{}\t{}\t{}\n".format(img, mos, mos_std, pred))
                        wfile.write("{}\t{}\t{}\t{}\n".format(img, mos, mos_std, pred))
        with open(os.path.join(trainset_path_next, 'sampled_srcc_plcc{}.txt'.format(i+1)), "w") as wfile:
            wfile.write("KADID SRCC: {}\nPLCC:{}\n".format(ret['srcc_sampled'][0], ret['plcc_sampled'][0]))
            wfile.write("KONIQ SRCC: {}\nPLCC:{}\n".format(ret['srcc_sampled'][1], ret['plcc_sampled'][1]))
        
        # save all sampled data
        copyfile(os.path.join(config.trainset_path, 'all_sampled_round{}.txt'.format(i)),\
                 os.path.join(trainset_path_next, 'all_sampled_round{}.txt'.format(i+1)))
        copyfile(os.path.join(config.trainset_path, 'all_koniq_sampled_round{}.txt'.format(i)),\
                 os.path.join(trainset_path_next, 'all_koniq_sampled_round{}.txt'.format(i+1)))
        copyfile(os.path.join(config.trainset_path, 'all_kadid_sampled_round{}.txt'.format(i)),\
                 os.path.join(trainset_path_next, 'all_kadid_sampled_round{}.txt'.format(i+1)))

        with open(os.path.join(trainset_path_next, 'all_sampled_round{}.txt'.format(i+1)), "a") as wfile:
            with open(os.path.join(trainset_path_next, 'all_koniq_sampled_round{}.txt'.format(i+1)), "a") as wofile:
                with open(os.path.join(trainset_path_next, 'all_kadid_sampled_round{}.txt'.format(i+1)), "a") as wafile:
                    for img, mos, mos_std in zip(ret['img_sampled'].tolist(), \
                                                 ret['mos_sampled'].tolist(), ret['mos_std_sampled'].tolist()):
                        if 'koniq-10k' in img:
                            wofile.write("{}\t{}\t{}\n".format(img, mos, mos_std))
                        else:
                            wafile.write("{}\t{}\t{}\n".format(img, mos, mos_std))
                        wfile.write("{}\t{}\t{}\n".format(img, mos, mos_std))
        ################################################################################################################################ 
        # append new labeled image for training
        ################################################################################################################################
        with open(os.path.join(trainset_path_next, 'unlabeled_round{}.txt'.format(i+1)), "w") as wfile:
            with open(os.path.join(trainset_path_next, 'unlabeled_koniq_round{}.txt'.format(i+1)), "w") as wofile:
                with open(os.path.join(trainset_path_next, 'unlabeled_kadid_round{}.txt'.format(i+1)), "w") as wafile:
                    for img, mos, mos_std in zip(ret['img_unlabeled'].tolist(), \
                                                  ret['mos_unlabeled'].tolist(), ret['mos_std_unlabeled'].tolist()):
                        if 'koniq-10k' in img:
                            wofile.write("{}\t{}\t{}\n".format(img, mos, mos_std))
                        else:
                            wafile.write("{}\t{}\t{}\n".format(img, mos, mos_std))
                        wfile.write("{}\t{}\t{}\n".format(img, mos, mos_std))
        # copy old train_file into the next round path
        # copyfile(os.path.join(config.trainset_path, 'train_round{}.txt'.format(i)),\
        #          os.path.join(trainset_path_next, 'train_round{}.txt'.format(i+1)))
        img_sampled = np.loadtxt(os.path.join(trainset_path_next, 'all_sampled_round{}.txt'.format(i+1)), \
            dtype=str, skiprows=0, comments='#', delimiter='\t', usecols=0)
        mos_sampled = np.loadtxt(os.path.join(trainset_path_next, 'all_sampled_round{}.txt'.format(i+1)), \
            dtype=float, skiprows=0, comments='#', delimiter='\t', usecols=1)
        mos_std_sampled = np.loadtxt(os.path.join(trainset_path_next, 'all_sampled_round{}.txt'.format(i+1)), \
            dtype=float, skiprows=0, comments='#', delimiter='\t', usecols=2)

        # append new pairs in the new train_txt
        pair_wise(img_sampled, mos_sampled, mos_std_sampled, ret['img_unlabeled'],
                  num_pairs=(i+2)*2000, train_txt = os.path.join(trainset_path_next, 'train_round{}.txt'.format(i+1)))
        







