import os
import time
import scipy.stats
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms
import torch.nn as nn
from ImageDataset import ImageDataset
from BaseCNN import BaseCNN
from MNL_Loss import Ncl_loss
from Transformers import AdaptiveResize
from tensorboardX import SummaryWriter
import utils

class Trainer(object):
    def __init__(self, config):
        torch.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.config = config
        self.loss_count = 0
        self.train_transform = transforms.Compose([
            #transforms.RandomRotation(3),
            AdaptiveResize(512),
            transforms.RandomCrop(config.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])
        self.test_transform = transforms.Compose([
            AdaptiveResize(768),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])

        # training set configuration
        self.train_loader = self._loader(csv_file = config.train_file, img_dir = config.trainset, \
                                         transform = self.train_transform, batch_size = config.batch_size, \
                                         drop_last = True)   
        self.valid_best= 0.0                                 
        # testing set configuration
        self.writer = SummaryWriter(os.path.join(config.runs_path, "{}".format(config.round)))
        self.model = nn.DataParallel(BaseCNN(config).cuda())
        self.model_name = type(self.model).__name__
        print(self.model)
        # loss function
        self.ncl_fn = Ncl_loss().cuda()
        self.initial_lr = config.lr
        if self.initial_lr is None:
            lr = 0.0005
        else:
            lr = self.initial_lr

        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters()}
            ], lr=lr, weight_decay=5e-4)

        # some states
        self.start_epoch = 0
        self.start_step = 0
        self.train_loss = []
        self.ckpt_path = config.ckpt_path
        self.ckpt_best_path = config.ckpt_best_path
        self.max_epochs = config.max_epochs
        self.epochs_per_eval = config.epochs_per_eval
        self.epochs_per_save = config.epochs_per_save

        # try load the model
        if config.resume or not config.train:
            ckpt = self._get_latest_checkpoint(path=config.ckpt_path)
            print("[*] ckpt:", ckpt)
            print('[*] start from the breakpoint')
            self._load_checkpoint(ckpt=ckpt)

        self.scheduler = lr_scheduler.StepLR(self.optimizer,
                            last_epoch=self.start_epoch-1,
                            step_size=config.decay_interval,
                            gamma=config.decay_ratio)

    def _loader(self, csv_file, img_dir, transform, test=False, batch_size=16, shuffle=True, pin_memory=True, num_workers=16, drop_last=False):
        data = ImageDataset(csv_file = csv_file,
                        img_dir = img_dir,
                        transform = transform,
                        test = test)
        train_loader = DataLoader(data,
                        batch_size = batch_size,
                        shuffle = shuffle,
                        pin_memory = pin_memory,
                        num_workers = num_workers,
                        drop_last = drop_last)
        return train_loader

    def fit(self):
        for epoch in range(self.start_epoch, self.max_epochs):
            _ = self._train_single_epoch(epoch)
            self.scheduler.step()

    def _train_single_batch(self, model, x1, x2, x3=None, x4=None, g=None, wfile=None):
        y1, y1_var, y1_ens, y1_var_ens = model(x1)
        y2, y2_var, y2_ens, y2_var_ens = model(x2)
        e2e_loss, ind_loss, ldiv_loss = self.ncl_fn(y1, y1_var, y2, y2_var, y1_ens, y1_var_ens, y2_ens, y2_var_ens, g)

        if not x3 == None:
            y3, y3_var, y3_ens, y3_var_ens = model(x3)
            y4, y4_var, y4_ens, y4_var_ens = model(x4)
            udiv_loss = self.ncl_fn(y3, y3_var, y4, y4_var, y3_ens, y3_var_ens, y4_ens, y4_var_ens)
            if not wfile == None:
                pass
                #self._save_quality(wfile, y1, y2, y3, y4)
            return e2e_loss, ind_loss, ldiv_loss, udiv_loss
        return e2e_loss, ind_loss, ldiv_loss, ldiv_loss

    def _save_quality(self, wfile, y1, y2, y3, y4):
        y = []
        for item in y1+y2+y3+y4:
            y.append(item.clone().view(-1).detach().cpu().numpy().tolist())
        n = len(y)
        for i in range(len(y[0])):
            wstr = ""
            for j in range(len(y)):
                wstr += "%.04f" % y[j][i]
                if j == len(y)-1:
                    wstr += '\n'
                else:
                    wstr += ','
            wfile.write(wstr)

    def _train_single_epoch(self, epoch):
        # initialize logging system
        num_steps_per_epoch = len(self.train_loader)
        local_counter = epoch * num_steps_per_epoch + 1
        start_time = time.time()
        beta = 0.9
        # for name, para in self.model.named_parameters():
        #     print('{} parameters requires_grad:{}'.format(name, para.requires_grad))

        running_loss = 0 if epoch == 0 else self.train_loss[-1][0]
        running_e2e_loss = 0 if epoch == 0 else self.train_loss[-1][1]
        running_ind_loss = 0 if epoch == 0 else self.train_loss[-1][2]
        running_ldiv_loss = 0 if epoch == 0 else self.train_loss[-1][3]
        running_udiv_loss = 0 if epoch == 0 else self.train_loss[-1][4]

        running_duration = 0.0

        # start training
        print('Adam learning rate: {:.8f}'.format(self.optimizer.param_groups[0]['lr']))
        self.model.train()
        #self.scheduler.step()
        with open(os.path.join(self.config.p_path, r'results_{}.txt'.format(epoch)), 'w') as wfile:
            for step, sample_batched in enumerate(self.train_loader, 0):
                if step < self.start_step:
                    continue
                x1, x2, g = Variable(sample_batched['I1']).cuda(), Variable(sample_batched['I2']).cuda(),\
                            Variable(sample_batched['y']).view(-1,1).cuda()
                            #Variable(sample_batched['I3']).cuda(), Variable(sample_batched['I4']).cuda(),\      
                self.optimizer.zero_grad()
                if self.config.loss == 'naive':
                    e2e_loss, ind_loss, ldiv_loss, udiv_loss = self._train_single_batch(self.model, x1=x1, x2=x2, g=g) #, udiv_loss
                    self.loss = torch.mean(ind_loss) #+ e2e_loss - self.config.weight_udiv*udiv_loss 
                elif self.config.loss == 'joint':
                    e2e_loss, ind_loss, ldiv_loss, udiv_loss = self._train_single_batch(self.model, x1=x1, x2=x2, g=g) #, udiv_loss
                    self.loss = torch.mean(ind_loss + e2e_loss)
                elif self.config.loss == 'ssl':
                    x3, x4 = Variable(sample_batched['I3']).cuda(), Variable(sample_batched['I4']).cuda()
                    e2e_loss, ind_loss, ldiv_loss, udiv_loss = self._train_single_batch(self.model, x1=x1, x2=x2, x3=x3, x4=x4, g=g) #, udiv_loss
                    self.loss = torch.mean(ind_loss + e2e_loss - self.config.weight_div*udiv_loss)
                else:
                    pass

                self.loss.backward()
                self.optimizer.step()

                # statistics
                running_loss = beta * running_loss + (1 - beta) * self.loss.data.item()
                loss_corrected = running_loss / (1 - beta ** local_counter)

                running_e2e_loss = beta * running_e2e_loss + (1 - beta) * torch.mean(e2e_loss).data.item()
                e2e_loss_corrected = running_e2e_loss / (1 - beta ** local_counter)

                running_ind_loss = beta * running_ind_loss + (1 - beta) * torch.mean(ind_loss).data.item()
                ind_loss_corrected = running_ind_loss / (1 - beta ** local_counter)

                running_ldiv_loss = beta * running_ldiv_loss + (1 - beta) * torch.mean(ldiv_loss).data.item()
                ldiv_loss_corrected = running_ldiv_loss / (1 - beta ** local_counter)

                running_udiv_loss = beta * running_udiv_loss + (1 - beta) * torch.mean(udiv_loss).data.item()
                udiv_loss_corrected = running_udiv_loss / (1 - beta ** local_counter)

                self.loss_count += 1
                if self.loss_count % 100 == 0:
                    self.writer.add_scalars('data/Corrected_Loss', {'loss': loss_corrected}, self.loss_count)
                    self.writer.add_scalars('data/E2E_corrected_loss', {'loss': e2e_loss_corrected}, self.loss_count)
                    self.writer.add_scalars('data/Ind_corrected_Loss', {'loss': ind_loss_corrected}, self.loss_count)
                    self.writer.add_scalars('data/Label_diversity_loss', {'loss': ldiv_loss_corrected}, self.loss_count)
                    self.writer.add_scalars('data/UnLabel_diversity_loss', {'loss': udiv_loss_corrected}, self.loss_count)
                
                current_time = time.time()
                duration = current_time - start_time
                running_duration = beta * running_duration + (1 - beta) * duration
                duration_corrected = running_duration / (1 - beta ** local_counter)
                examples_per_sec = self.config.batch_size / duration_corrected
                format_str = ('(R: %d, E:%d, S:%d / %d) [Loss = %.4f E2E Loss = %.4f, Ind Loss = %.4f, LDiv Loss = %.8f'
                            ', UDiv Loss = %.08f] (%.1f samples/sec; %.3f  sec/batch)')
                print(format_str % (self.config.round, epoch, step, num_steps_per_epoch, loss_corrected, e2e_loss_corrected, 
                      ind_loss_corrected, ldiv_loss_corrected, udiv_loss_corrected, examples_per_sec, duration_corrected))

                local_counter += 1
                self.start_step = 0
                start_time = time.time()

        self.train_loss.append([loss_corrected, e2e_loss_corrected, ind_loss_corrected, ldiv_loss_corrected, udiv_loss_corrected])
        # evaluate after every epoch
       
        valid_srcc = 0
        if epoch > 5:
            # evaluate after every epoch
            srcc, plcc, n = self._eval(self.model) # n is the number of heads
            valid_srcc = srcc["koniq10k_valid"]['ensemble'] + srcc["kadid10k_valid"]['ensemble']
            tb = utils.print_tb(srcc, plcc, n)
            print(tb)
            f = open(os.path.join(self.config.result_path, r'results_{}.txt'.format(epoch)), 'w')
            f.write(str(tb))
            f.close()

        if (epoch+1) % self.epochs_per_save == 0:
            if epoch == 2:
                model_name = '{}-{:0>5d}.pt'.format(self.model_name, epoch)
                model_name = os.path.join(self.ckpt_path, model_name)
                self._save_checkpoint({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'train_loss': self.train_loss,
                }, os.path.join(self.ckpt_path, 'checkpoint.pt'))

        if self.valid_best < valid_srcc and epoch > 6:
            # save best path
            model_name = 'best.pt'
            model_name = os.path.join(self.ckpt_best_path, model_name)
            self._save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                # 'optimizer': self.optimizer.state_dict(),
                # 'train_loss': self.train_loss,
            }, model_name)
            # save best result
            f = open(os.path.join(self.ckpt_best_path, r'best.txt'.format(epoch)), 'w')
            f.write(str(tb))
            f.close()
            # updata valid_best
            self.valid_best = valid_srcc

        return self.loss.data.item()

    def _eval_single(self, model, loader):
        srcc, plcc = {}, {}
        q_mos, q_ens = [], []
        for step, sample_batched in enumerate(loader, 0):
            x, y = Variable(sample_batched['I']).cuda(), sample_batched['mos']
            y_bar, _, y_ens, _ = model(x)
            for item in y.data.numpy().tolist():
                q_mos.append(item)
            for item in y_ens.cpu().data.numpy().tolist():
                q_ens.append(item)
            if step == 0:
                # claim a list
                q_hat = [[] for i in range(len(y_bar))] 
                
            for i in range(len(y_bar)):
                for item in y_bar[i].cpu().data.numpy().tolist():
                    q_hat[i].append(item)

        for i in range(len(q_hat)):
            srcc['model{}'.format(i)] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat[i])[0]
            plcc['model{}'.format(i)] = scipy.stats.mstats.pearsonr(x=q_mos, y=q_hat[i])[0]

        srcc['ensemble'] = scipy.stats.mstats.spearmanr(x=q_mos, y=q_ens)[0]
        plcc['ensemble'] = scipy.stats.mstats.pearsonr(x=q_mos, y=q_ens)[0]

        return srcc, plcc, len(q_hat)

    def _eval(self, model):
        srcc, plcc = {}, {}
        model.eval()
        start_time = time.time()
        path = "/home/zhihua/Active_learning_head4/IQA_database"
        koniq_valid_loader = self._loader(csv_file = os.path.join(path, 'splits{}'.format(self.config.path_idx), str(self.config.split), 'koniq10k_valid_score.txt'),
                                img_dir = self.config.koniq10k_set, batch_size=8, transform = self.test_transform, test = True, shuffle = False, 
                                pin_memory = True, num_workers = 8)
        koniq_active_loader = self._loader(csv_file = os.path.join(self.config.trainset_path,'unlabeled_koniq_round{}.txt'.format(self.config.round)),
                                img_dir = self.config.trainset, batch_size=8, transform = self.test_transform, test = True, shuffle = False, 
                                pin_memory = True, num_workers = 8)
        koniq_test_loader = self._loader(csv_file = os.path.join(path, 'splits{}'.format(self.config.path_idx), str(self.config.split), 'koniq10k_test_score.txt'),
                                img_dir = self.config.koniq10k_set, batch_size=8, transform = self.test_transform, test = True, shuffle = False, 
                                pin_memory = True, num_workers = 8)
        kadid_valid_loader = self._loader(csv_file = os.path.join(path, 'splits{}'.format(self.config.path_idx), str(self.config.split), 'kadid10k_valid_score.txt'),
                                img_dir = self.config.kadid10k_set, batch_size=8, transform = self.test_transform, test = True, shuffle = False, 
                                pin_memory = True, num_workers = 8)
        kadid_active_loader = self._loader(csv_file = os.path.join(self.config.trainset_path,'unlabeled_kadid_round{}.txt'.format(self.config.round)),
                                img_dir = self.config.trainset, batch_size=8, transform = self.test_transform, test = True, shuffle = False, 
                                pin_memory = True, num_workers = 8)
        kadid_test_loader = self._loader(csv_file = os.path.join(path, 'splits{}'.format(self.config.path_idx), str(self.config.split), 'kadid10k_test_score.txt'),
                                img_dir = self.config.kadid10k_set, batch_size=8, transform = self.test_transform, test = True, shuffle = False, 
                                pin_memory = True, num_workers = 8)
        srcc['koniq10k_valid'], plcc['koniq10k_valid'], _ = self._eval_single(model, koniq_valid_loader)
        srcc['koniq10k_active'], plcc['koniq10k_active'], _ = self._eval_single(model, koniq_active_loader)
        srcc['koniq10k_test'], plcc['koniq10k_test'], n = self._eval_single(model, koniq_test_loader)
        
        srcc['kadid10k_valid'], plcc['kadid10k_valid'], _ = self._eval_single(model, kadid_valid_loader)
        srcc['kadid10k_active'], plcc['kadid10k_active'], _ = self._eval_single(model, kadid_active_loader)
        srcc['kadid10k_test'], plcc['kadid10k_test'], n = self._eval_single(model, kadid_test_loader)

        print('[*] testing time: {}'.format(time.time()-start_time))
        return srcc, plcc, n

    def _load_checkpoint(self, ckpt):
        if os.path.isfile(ckpt):
            print("[*] loading checkpoint '{}'".format(ckpt))
            checkpoint = torch.load(ckpt)
            self.start_epoch = checkpoint['epoch']+1
            self.train_loss = checkpoint['train_loss']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.initial_lr is not None:
                for param_group in self.optimizer.param_groups:
                    param_group['initial_lr'] = self.initial_lr
            print("[*] loaded checkpoint '{}' (epoch {})"
                  .format(ckpt, checkpoint['epoch']))
            # delete the file
            os.remove(ckpt)
        else:
            print("[!] no checkpoint found at '{}'".format(ckpt))

    @staticmethod
    def _get_latest_checkpoint(path):
        ckpts = os.listdir(path)
        ckpts = [ckpt for ckpt in ckpts if not os.path.isdir(os.path.join(path, ckpt))]
        all_times = sorted(ckpts, reverse=True)
        return os.path.join(path, all_times[0])

    # save checkpoint
    @staticmethod
    def _save_checkpoint(state, filename='checkpoint.pth.tar'):
        torch.save(state, filename)