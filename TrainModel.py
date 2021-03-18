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
from MNL_Loss import Fidelity_Loss, Ncl_loss
from Transformers import AdaptiveResize
from tensorboardX import SummaryWriter
import prettytable as pt

class Trainer(object):
    def __init__(self, config):
        torch.manual_seed(config.seed)
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
        self.train_loader = self._loader(csv_file = os.path.join(config.trainset, 'splits2', str(config.split), config.train_txt),
                                        img_dir = config.trainset, transform = self.train_transform, batch_size = config.batch_size)                                    
        # testing set configuration
        self.kadid10k_loader = self._loader(csv_file = os.path.join(config.kadid10k_set, 'splits2', str(config.split), 'kadid10k_test_10125.txt'),
                                        img_dir = config.kadid10k_set, transform = self.test_transform, test = True, shuffle = False, 
                                        pin_memory = True, num_workers = 0)
        self.livec_loader = self._loader(csv_file = os.path.join(config.livec_set, 'splits2', str(config.split), 'clive_test.txt'),
                                        img_dir = config.livec_set, transform = self.test_transform, test = True, shuffle = False, 
                                        pin_memory = True, num_workers = 0)
        self.spaq_loader = self._loader(csv_file = os.path.join(config.spaq_set, 'spaq_test.txt'),
                                        img_dir = config.spaq_set, transform = self.test_transform, test = True, shuffle = False, 
                                        pin_memory = True, num_workers = 8)
        self.koniq10k_loader = self._loader(csv_file = os.path.join(config.koniq10k_set, 'splits2', str(config.split), 'koniq10k_test.txt'),
                                        img_dir = config.koniq10k_set, transform = self.test_transform, test = True, shuffle = False, 
                                        pin_memory = True, num_workers = 0)

        self.writer = SummaryWriter(config.runs_path)
        self.model = nn.DataParallel(BaseCNN(config).cuda())
        self.model_name = type(self.model).__name__
        print(self.model)
        # loss function
        self.loss_fn = Fidelity_Loss().cuda()
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
        self.max_epochs = config.max_epochs
        self.epochs_per_eval = config.epochs_per_eval
        self.epochs_per_save = config.epochs_per_save

        # try load the model
        if config.resume or not config.train:
            ckpt = self._get_latest_checkpoint(path=config.ckpt_path)
            print('**********************************************************************************')
            print("ckpt:", ckpt)
            print('start from the pretrained model of Save Model')
            print('**********************************************************************************')
            self._load_checkpoint(ckpt=ckpt)

        self.scheduler = lr_scheduler.StepLR(self.optimizer,
                            last_epoch=self.start_epoch-1,
                            step_size=config.decay_interval,
                            gamma=config.decay_ratio)

    def _loader(self, csv_file, img_dir, transform, test=False, batch_size=1, shuffle=True, pin_memory=True, num_workers=32):
        data = ImageDataset(csv_file = csv_file,
                        img_dir = img_dir,
                        transform = transform,
                        test = test)
        train_loader = DataLoader(data,
                        batch_size = batch_size,
                        shuffle = shuffle,
                        pin_memory = pin_memory,
                        num_workers = num_workers)
        return train_loader

    def fit(self):
        for epoch in range(self.start_epoch, self.max_epochs):
            _ = self._train_single_epoch(epoch)
            self.scheduler.step()

    def _train_single_batch(self, model, x1, x2, x3, x4, g=None, wfile=None):
        y11, y11_var, y12, y12_var, y13, y13_var, y14, y14_var = model(x1)
        y21, y21_var, y22, y22_var, y23, y23_var, y24, y24_var = model(x2)
        e2e_loss, ind_loss, ldiv_loss = self.ncl_fn([y11, y12, y13, y14], [y11_var, y12_var, y13_var, y14_var],\
                                                    [y21, y22, y23, y24], [y21_var, y22_var, y23_var, y24_var], g)

        y31, y31_var, y32, y32_var, y33, y33_var, y34, y34_var = model(x3)
        y41, y41_var, y42, y42_var, y43, y43_var, y44, y44_var = model(x4)

        udiv_loss, con_loss = self.ncl_fn([y31, y32, y33, y34], [y31_var, y32_var, y33_var, y34_var],\
                                [y41, y42, y43, y44], [y41_var, y42_var, y43_var, y44_var])
        if not wfile == None:
            self._save_quality(wfile, y11, y12, y13, y14, y21, y22, y23, y24, \
                                      y31, y32, y33, y34, y41, y42, y43, y44)
        return e2e_loss, ind_loss, ldiv_loss, udiv_loss, con_loss

    def _save_quality(self, wfile, x1, x2, x3, x4, y1, y2, y3, y4, xu1, xu2, xu3, xu4, yu1, yu2, yu3, yu4):
        x1 = x1.clone().view(-1).detach().cpu().numpy().tolist()
        x2 = x2.clone().view(-1).detach().cpu().numpy().tolist()
        x3 = x3.clone().view(-1).detach().cpu().numpy().tolist()
        x4 = x4.clone().view(-1).detach().cpu().numpy().tolist()
        y1 = y1.clone().view(-1).detach().cpu().numpy().tolist()
        y2 = y2.clone().view(-1).detach().cpu().numpy().tolist()
        y3 = y3.clone().view(-1).detach().cpu().numpy().tolist()
        y4 = y4.clone().view(-1).detach().cpu().numpy().tolist()
        xu1 = xu1.clone().view(-1).detach().cpu().numpy().tolist()
        xu2 = xu2.clone().view(-1).detach().cpu().numpy().tolist()
        xu3 = xu3.clone().view(-1).detach().cpu().numpy().tolist()
        xu4 = xu4.clone().view(-1).detach().cpu().numpy().tolist()
        yu1 = yu1.clone().view(-1).detach().cpu().numpy().tolist()
        yu2 = yu2.clone().view(-1).detach().cpu().numpy().tolist()
        yu3 = yu3.clone().view(-1).detach().cpu().numpy().tolist()
        yu4 = yu4.clone().view(-1).detach().cpu().numpy().tolist()
        for i in range(len(x1)):
            wstr = "[%.04f,%.04f,%.04f,%.04f] "% (x1[i], x2[i], x3[i], x4[i]) + \
                   "[%.04f,%.04f,%.04f,%.04f] "% (y1[i], y2[i], y3[i], y4[i]) + \
                   "[%.04f,%.04f,%.04f,%.04f] "% (xu1[i], xu2[i], xu3[i], xu4[i]) + \
                   "[%.04f,%.04f,%.04f,%.04f] \n"% (yu1[i], yu2[i], yu3[i], yu4[i])
            wfile.write(wstr)

    def _train_single_epoch(self, epoch):
        # initialize logging system
        num_steps_per_epoch = len(self.train_loader)
        local_counter = epoch * num_steps_per_epoch + 1
        start_time = time.time()
        beta = 0.9
        for name, para in self.model.named_parameters():
            print('{} parameters requires_grad:{}'.format(name, para.requires_grad))

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
                x1, x2, x3, x4, g = Variable(sample_batched['I1']).cuda(), Variable(sample_batched['I2']).cuda(),\
                                    Variable(sample_batched['I3']).cuda(), Variable(sample_batched['I4']).cuda(),\
                                    Variable(sample_batched['y']).view(-1,1).cuda()

                self.optimizer.zero_grad()
                e2e_loss, ind_loss, ldiv_loss, udiv_loss, con_loss = self._train_single_batch(self.model, x1, x2, x3, x4, g, wfile)
                self.loss = e2e_loss + self.config.weight_ind*ind_loss - self.config.weight_ldiv*ldiv_loss -\
                            self.config.weight_udiv*udiv_loss + self.weight_con*con_loss
                            
                self.loss.backward()
                self.optimizer.step()

                # statistics
                running_loss = beta * running_loss + (1 - beta) * self.loss.data.item()
                loss_corrected = running_loss / (1 - beta ** local_counter)

                running_e2e_loss = beta * running_e2e_loss + (1 - beta) * e2e_loss.data.item()
                e2e_loss_corrected = running_e2e_loss / (1 - beta ** local_counter)

                running_ind_loss = beta * running_ind_loss + (1 - beta) * ind_loss.data.item()
                ind_loss_corrected = running_ind_loss / (1 - beta ** local_counter)

                running_ldiv_loss = beta * running_ldiv_loss + (1 - beta) * ldiv_loss.data.item()
                ldiv_loss_corrected = running_ldiv_loss / (1 - beta ** local_counter)

                running_udiv_loss = beta * running_udiv_loss + (1 - beta) * udiv_loss.data.item()
                udiv_loss_corrected = running_udiv_loss / (1 - beta ** local_counter)

                self.loss_count += 1
                if self.loss_count % 100 == 0:
                    self.writer.add_scalars('data/Corrected_Loss', {'loss': loss_corrected}, self.loss_count)
                    self.writer.add_scalars('data/E2E_corrected_loss', {'loss': e2e_loss_corrected}, self.loss_count)
                    self.writer.add_scalars('data/Ind_corrected_Loss', {'loss': ind_loss_corrected}, self.loss_count)
                    self.writer.add_scalars('data/Label_diversity_loss', {'loss': running_ldiv_loss}, self.loss_count)
                    self.writer.add_scalars('data/UnLabel_diversity_loss', {'loss': running_udiv_loss}, self.loss_count)
                
                current_time = time.time()
                duration = current_time - start_time
                running_duration = beta * running_duration + (1 - beta) * duration
                duration_corrected = running_duration / (1 - beta ** local_counter)
                examples_per_sec = self.config.batch_size / duration_corrected
                format_str = ('(E:%d, S:%d / %d) [Loss = %.4f E2E Loss = %.4f, Ind Loss = %.4f, LDiv Loss = %.8f, UDiv Loss = %.08f] (%.1f samples/sec; %.3f '
                            'sec/batch)')
                print(format_str % (epoch, step, num_steps_per_epoch, loss_corrected, e2e_loss_corrected, ind_loss_corrected,
                                    ldiv_loss_corrected, udiv_loss_corrected, examples_per_sec, duration_corrected))

                local_counter += 1
                self.start_step = 0
                start_time = time.time()

        self.train_loss.append([loss_corrected, e2e_loss_corrected, ind_loss_corrected, ldiv_loss_corrected, udiv_loss_corrected])

        # evaluate after every other epoch
        ret_eval = self._eval(self.model)
        ret_kadd = ret_eval['kadid10k']
        ret_live = ret_eval['livec']
        ret_spaq = ret_eval['spaq']
        ret_konk = ret_eval['koniq10k']
        # calculate ensemble
        # kadid
        ensemble = ret_kadd["model1"][3] + ret_kadd["model2"][3] + ret_kadd["model3"][3] + ret_kadd["model1"][3]
        skadid = scipy.stats.mstats.spearmanr(x=ret_kadd["model1"][2], y= ensemble)[0]
        pkadid = scipy.stats.mstats.pearsonr(x=ret_kadd["model1"][2], y= ensemble)[0]
        
        ensemble = ret_live["model1"][3] + ret_live["model2"][3] + ret_live["model3"][3] + ret_live["model1"][3]
        slivec = scipy.stats.mstats.spearmanr(x=ret_live["model1"][2], y= ensemble)[0]
        plivec = scipy.stats.mstats.pearsonr(x=ret_live["model1"][2], y= ensemble)[0]
        
        ensemble = ret_spaq["model1"][3] + ret_spaq["model2"][3] + ret_spaq["model3"][3] + ret_spaq["model1"][3]
        sspaq = scipy.stats.mstats.spearmanr(x=ret_spaq["model1"][2], y= ensemble)[0]
        pspaq = scipy.stats.mstats.pearsonr(x=ret_spaq["model1"][2], y= ensemble)[0]
        
        ensemble = ret_konk["model1"][3] + ret_konk["model2"][3] + ret_konk["model3"][3] + ret_konk["model1"][3]
        skoniq = scipy.stats.mstats.spearmanr(x=ret_konk["model1"][2], y= ensemble)[0]
        pkoniq = scipy.stats.mstats.pearsonr(x=ret_konk["model1"][2], y= ensemble)[0]

        tb = pt.PrettyTable()
        tb.field_names = ["Model1", "KADID10K", "LIVEC", "SPAQ", "KONIQ10K"]
        tb.add_row(['SRCC', ret_kadd["model1"][0], ret_live["model1"][0], ret_spaq["model1"][0], ret_konk["model1"][0]])
        tb.add_row(['PLCC', ret_kadd["model1"][1], ret_live["model1"][1], ret_spaq["model1"][1], ret_konk["model1"][1]])
        tb.add_row(["Model2", "KADID10K", "LIVEC", "SPAQ", "KONIQ10K"])
        tb.add_row(['SRCC', ret_kadd["model2"][0], ret_live["model2"][0], ret_spaq["model2"][0], ret_konk["model2"][0]])
        tb.add_row(['PLCC', ret_kadd["model2"][1], ret_live["model2"][1], ret_spaq["model2"][1], ret_konk["model2"][1]])
        tb.add_row(["Model3", "KADID10K", "LIVEC", "SPAQ", "KONIQ10K"])
        tb.add_row(['SRCC', ret_kadd["model3"][0], ret_live["model3"][0], ret_spaq["model3"][0], ret_konk["model3"][0]])
        tb.add_row(['PLCC', ret_kadd["model3"][1], ret_live["model3"][1], ret_spaq["model3"][1], ret_konk["model3"][1]])
        tb.add_row(["Model4", "KADID10K", "LIVEC", "SPAQ", "KONIQ10K"])
        tb.add_row(['SRCC', ret_kadd["model4"][0], ret_live["model4"][0], ret_spaq["model4"][0], ret_konk["model4"][0]])
        tb.add_row(['PLCC', ret_kadd["model4"][1], ret_live["model4"][1], ret_spaq["model4"][1], ret_konk["model4"][1]])
        tb.add_row(["Ensemble", "KADID10K", "LIVEC", "SPAQ", "KONIQ10K"])
        tb.add_row(['SRCC', skadid, slivec, sspaq, skoniq])
        tb.add_row(['PLCC', pkadid, plivec, pspaq, pkoniq])

        print(tb)
        f = open(os.path.join(self.config.result_path, r'results_{}.txt'.format(epoch)), 'w')
        f.write(str(tb))
        f.close()

        if (epoch+1) % self.epochs_per_save == 0:
            model_name = '{}-{:0>5d}.pt'.format(self.model_name, epoch)
            model_name = os.path.join(self.ckpt_path, model_name)
            self._save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'train_loss': self.train_loss,
            }, model_name)
        return self.loss.data.item()

    def _eval_single(self, model, loader):
        q_mos = []
        q_hat1, q_hat2, q_hat3, q_hat4 = [], [], [], []
        for step, sample_batched in enumerate(loader, 0):
            x, y = Variable(sample_batched['I']).cuda(), sample_batched['mos']
            y_bar1, _, y_bar2, _, y_bar3, _, y_bar4, _ = model(x)
            q_mos.append(y.data.numpy())
            q_hat1.append(y_bar1.cpu().data.numpy())
            q_hat2.append(y_bar2.cpu().data.numpy())
            q_hat3.append(y_bar3.cpu().data.numpy())
            q_hat4.append(y_bar4.cpu().data.numpy())

        srcc1 = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat1)[0]
        plcc1 = scipy.stats.mstats.pearsonr(x=q_mos, y=q_hat1)[0]
        srcc2 = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat2)[0]
        plcc2 = scipy.stats.mstats.pearsonr(x=q_mos, y=q_hat2)[0]
        srcc3 = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat3)[0]
        plcc3 = scipy.stats.mstats.pearsonr(x=q_mos, y=q_hat3)[0]
        srcc4 = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat4)[0]
        plcc4 = scipy.stats.mstats.pearsonr(x=q_mos, y=q_hat4)[0]
        return {"model1": [srcc1, plcc1, np.array(q_mos), np.array(q_hat1)],
                "model2": [srcc2, plcc2, np.array(q_mos), np.array(q_hat2)],
                "model3": [srcc3, plcc3, np.array(q_mos), np.array(q_hat3)],
                "model4": [srcc4, plcc4, np.array(q_mos), np.array(q_hat4)]}

    def _eval(self, model):
        sp = {}
        model.eval()
        sp['kadid10k'] = self._eval_single(model, self.kadid10k_loader)
        sp['livec'] = self._eval_single(model, self.livec_loader)
        sp['spaq'] = self._eval_single(model, self.spaq_loader)
        sp['koniq10k'] = self._eval_single(model, self.koniq10k_loader)
        return sp

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