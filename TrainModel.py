import imp
import os, math, time
import scipy.stats
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import prettytable as pt
import matplotlib.pyplot as plt
from model import Siamese
from utils.ImageDataset import ImageDataset
from utils.Transformers import AdaptiveResize
from utils.utils import get_latest_checkpoint,save_checkpoint
from utils.coeff_func import compute_stress, coeff_fit
from  utils.EMA import EMA
from tqdm import tqdm
from utils.queue import ft_Queue

class Trainer(object):
    def __init__(self, config):
        torch.manual_seed(config.seed)
        self.config = config
        self.train_transform = transforms.Compose([
                transforms.RandomRotation(3),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomInvert(p=0.5),
                transforms.RandomCrop(768),
                transforms.ToTensor(),
            ])

        self.test_transform = transforms.Compose([
            AdaptiveResize(1024),
            transforms.ToTensor(),
        ])
        self.train_batch_size = config.batch_size
        self.test_batch_size = config.batch_size
        self.initial_lr = config.lr
        self.netF = nn.DataParallel(Siamese(config)).cuda()
        self.cube = Variable(torch.rand(12,12)).cuda()
        self.cube.requires_grad=True
        self.optimizer = optim.Adam([
        {'params': self.netF.parameters(), 'lr': self.initial_lr},
        {'params': self.cube, 'lr': self.initial_lr},
        ])
        self.model_name = 'CD'
        # some states
        self.start_epoch = 0
        self.start_step = 0
        self.mseloss = torch.nn.MSELoss()
        self.train_loss = []
        self.ckpt_path = config.ckpt_path
        self.max_epochs = config.max_epochs
        self.epochs_per_eval = config.epochs_per_eval
        self.epochs_per_save = config.epochs_per_save
        self.result_path = config.result_path
        self.preds_path = config.preds_path
        self.best_validation_srcc = 0.0
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer,
                                    last_epoch = self.start_epoch-1,
                                    milestones = config.scheduler_milestones, 
                                    gamma = config.scheduler_gamma)
        
        if config.resume or not config.train:
            if config.ckpt:
                ckpt = os.path.join(config.ckpt)
            else:
                ckpt = get_latest_checkpoint(path=config.ckpt_path)
            self._load_checkpoint(ckpt=ckpt)
        self.ema = EMA(self.netF, 0.999)
        self.ema.register()
     
        print('scheduler:',config.scheduler_milestones,'gamma:',config.scheduler_gamma)
       

    def loader(self, csv_file, img_dir, transform, test, batch_size, shuffle, num_workers):
        data = ImageDataset(csv_file=csv_file, img_dir=img_dir, transform=transform, test=test,cfg=self.config)
        loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle,pin_memory=True, num_workers=num_workers)
        return loader

    def fit(self):
        self.best_srcc=0
        self.best_epoch=0
        for epoch in range(self.start_epoch, self.max_epochs):
            self.train_loader = self.loader(csv_file=os.path.join(self.config.files, self.config.train_file),
                    img_dir=self.config.image_path, transform=self.train_transform, test=False, 
                    batch_size=self.train_batch_size, shuffle=True, num_workers=8)
            _ = self._train_single_epoch(epoch)
        if epoch!=0:
            srcc_m = 0
            maxepoch = 0
            val_loader = self.loader(csv_file=os.path.join(self.config.files,self.config.val_file), 
                        img_dir = self.config.image_path, transform=self.test_transform, test=True, 
                        batch_size=self.test_batch_size, shuffle=False, num_workers=8)
            for i in range(1,self.max_epochs):
                self._load_checkpoint(self.config.checkpoint+'/checkpoints/CD-'+'%05d.pt'%(i))
                sr, pl,stress,_,_ = self.eval_once(loader=val_loader)
                print('val: srcc: {:.4f} plcc: {:.4f} stress: {:.4f}'.format(sr, pl, stress))
                if sr>srcc_m:
                    srcc_m=sr
                    maxepoch = i
                print('max srcc:{:.4f} epoch:{}'.format(srcc_m,maxepoch))
            print('epoch{} is best epoch!!'.format(maxepoch))
            self._load_checkpoint(self.config.checkpoint+'/checkpoints/CD-'+'%05d.pt'%(maxepoch))
            _, _,_ = self.eval(maxepoch)

    def evaleveryepoch(self, epoch):
        maxepoch=2
        self._load_checkpoint(self.config.checkpoint+'/checkpoints/CD-'+'%05d.pt'%(5))
        _, _ = self.eval(5)
        return 0
        
    def _train_single_epoch(self, epoch):
        time_s = time.time()
        loader_steps = len(self.train_loader)
        start_steps = epoch * len(self.train_loader)
        total_steps = self.config.max_epochs * len(self.train_loader)
        local_counter = epoch * total_steps + 1
        start_time = time.time()

        print('Adam learning rate: {:f}'.format(self.optimizer.param_groups[0]['lr']))
        for step, sample_batched in enumerate(self.train_loader, 0):
            I1, I2, mos = sample_batched['I1'], sample_batched['I2'],sample_batched['mos']
            I1, I2, mos = Variable(I1).cuda(), Variable(I2).cuda(), Variable(mos).cuda()
            self.optimizer.zero_grad()
            output = self.netF(I1, I2, self.cube) #feature generator
            self.loss = self.mseloss(mos.to(torch.float32),output.to(torch.float32))
            self.loss.backward()
            self.optimizer.step()
            self.ema.update()
            current_time = time.time()
            duration = current_time - start_time
            running_duration = duration
            duration_corrected = running_duration 
            examples_per_sec = self.train_batch_size / duration_corrected
            if step%30==0:
                format_str = ('(E:%d, S:%d/%d) [Loss = %.4f, (%.1f samples/sec; %.3f sec/batch)')
                print(format_str % (epoch, step, loader_steps, self.loss.data.item(), examples_per_sec, duration_corrected))    
            local_counter += 1
            self.start_step = 0
            start_time = time.time()
        # self.train_loss.append([loss_corrected])
        print('finish train: {}'.format(time.time()-time_s))
        self.ema.apply_shadow()
        self.netF.eval()
        config = self.config
        val_loader = self.loader(csv_file=os.path.join(self.config.files,config.val_file), 
                img_dir = config.image_path, transform=self.test_transform, test=True, 
                batch_size=self.test_batch_size, shuffle=False, num_workers=1)
        sr, pl,stress,_,_ = self.eval_once(loader=val_loader)
        print('val: srcc: {:.4f} plcc: {:.4f} stress: {:.4f}'.format(sr, pl, stress))
        if sr>self.best_srcc:
            self.best_srcc=sr
            self.best_epoch=epoch
        print('best srcc:{} best epoch:{}'.format(self.best_srcc,self.best_epoch))
        del val_loader   

        if (epoch+1) % self.epochs_per_save == 0:
            model_name = '{}-{:0>5d}.pt'.format(self.model_name, epoch)
            model_name = os.path.join(self.ckpt_path, model_name)
            save_checkpoint({
                    'epoch': epoch,
                    'netF_dict': self.netF.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'cube': self.cube
                }, model_name)
            print('save:'+ model_name)
        else:
            print('not save epoch:{}'.format(epoch))
        self.scheduler.step()
        self.ema.restore()
        return 1
        
    def eval_once(self, loader):
        q_mos, q_hat = [],[]
        self.netF.eval()
        for step, sample_batched in enumerate(loader, 0):
            x, y, mos = sample_batched['I1'], sample_batched['I2'], sample_batched['mos']
            x, y, mos = Variable(x).cuda(),Variable(y).cuda(),Variable(mos).cuda()
            y_bar= self.netF(x,y,self.cube)
            y_bar.cpu()
            pred=y_bar.cpu().detach().numpy().tolist()
            if isinstance(pred,list):
                q_hat.extend(pred)
                q_mos.extend(mos.cpu().detach().numpy().tolist())
            else:
                q_hat.append(np.array(pred))
                q_mos.append(mos.cpu().detach().numpy())
            if step%10==0: print("completed:{}/{}".format(step, len(loader)))
        q_mos,q_hat = np.array(q_mos),np.array(q_hat)
        stress = compute_stress(q_hat,q_mos)
        _, cc_v, srocc_v, _, _ = coeff_fit(q_hat, q_mos)
        srcc,plcc,stress=srocc_v,cc_v,stress
        print(srcc, plcc,stress)
        return srcc, plcc, stress, q_hat, q_mos

    def writemyfile(self,name,list):
        Name=[]
        Name.append(name)
        test = pd.DataFrame(columns = Name,data=list,index=None)
        test.to_csv(self.config.result_path+'/{}.csv'.format(name))
  
    def eval(self, epoch):
        srcc, plcc,stress = {}, {}, {}
        config = self.config
         # testing set configuration
        print('Evaluating...epoch{}'.format(epoch))
        val_loader = self.loader(csv_file=os.path.join(self.config.files,config.val_file), 
                img_dir = config.image_path, transform=self.test_transform, test=True, 
                batch_size=self.test_batch_size, shuffle=False, num_workers=4)
        srcc['val'], plcc['val'],stress['val'],q_hat,q_mos = self.eval_once(loader=val_loader)
        print('val: srcc: {:.4f} plcc: {:.4f} stress: {:.4f}'.format(srcc['val'], plcc['val'], stress['val']))
        self.writemyfile(name='val_hat', list = q_hat)
        self.writemyfile(name='val_mos', list = q_mos)
        del val_loader

        val_loader = self.loader(csv_file=os.path.join(self.config.files,config.test_file), 
                img_dir = config.image_path, transform=self.test_transform, test=True, 
                batch_size=self.test_batch_size, shuffle=False, num_workers=4)
        srcc['test'], plcc['test'],stress['test'],q_hat,q_mos = self.eval_once(loader=val_loader)
        print('test: srcc: {:.4f} plcc: {:.4f} stress: {:.4f}'.format(srcc['test'], plcc['test'], stress['test']))
        self.writemyfile(name='test_hat', list = q_hat)
        self.writemyfile(name='test_mos', list = q_mos)
        del val_loader

        val_loader = self.loader(csv_file=os.path.join(self.config.files,config.testaligned_file), 
                img_dir = config.image_path, transform=self.test_transform, test=True, 
                batch_size=self.test_batch_size, shuffle=False, num_workers=4)
        srcc['align'], plcc['align'],stress['align'],q_hat,q_mos = self.eval_once(loader=val_loader)
        print('align: srcc: {:.4f} plcc: {:.4f} stress: {:.4f}'.format(srcc['align'], plcc['align'], stress['align']))
        self.writemyfile(name='align_hat', list = q_hat)
        self.writemyfile(name='align_mos', list = q_mos)
        del val_loader
        
        val_loader = self.loader(csv_file=os.path.join(self.config.files,config.testnotaligned_file), 
                img_dir = config.image_path, transform=self.test_transform, test=True, 
                batch_size=self.test_batch_size, shuffle=False, num_workers=4)
        srcc['notalign'], plcc['notalign'],stress['notalign'],q_hat,q_mos = self.eval_once(loader=val_loader)
        print('notalign: srcc: {:.4f} plcc: {:.4f} stress: {:.4f}'.format(srcc['notalign'], plcc['notalign'], stress['notalign']))
        self.writemyfile(name='notalign_hat', list = q_hat)
        self.writemyfile(name='notalign_mos', list = q_mos)
        del val_loader

        del config
        with open(os.path.join(self.result_path, 'result_{}.txt'.format(epoch)), 'w') as txt_file:
            tb = pt.PrettyTable()
            tb.field_names = ["---","test_all", "test_aligned", "test_notaligned", "VALIDATION"]
            tb.add_row(['SRCC',     srcc['test'], srcc['align'],   srcc['notalign'],   srcc['val']])
            tb.add_row(['PLCC',     plcc['test'], plcc['align'],   plcc['notalign'],   plcc['val']])
            tb.add_row(['stress', stress['test'], stress['align'], stress['notalign'], stress['val']])
            print(tb)
            txt_file.write(str(tb))
        return srcc, plcc, stress

    def _load_checkpoint(self, ckpt):
        if os.path.isfile(ckpt):
            print("[*] loading checkpoint '{}'".format(ckpt))
            checkpoint = torch.load(ckpt)
            self.start_epoch = checkpoint['epoch']+1
            self.netF.load_state_dict(checkpoint['netF_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.cube=checkpoint['cube'].cuda()
            print("[*] loaded checkpoint '{}' (epoch {})  restart epoch{}!"
                  .format(ckpt, checkpoint['epoch'],self.start_epoch))
        else:
            print("[!] no checkpoint found at '{}'".format(ckpt))
