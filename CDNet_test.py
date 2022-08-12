import torch
from model import Siamese
from DataLoader import test_256,test_512,test_1024
from colorspace_conversion import rgb2xyz,rgb2yiq,myRGB2Lab,xyz2rgb
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
import os
from coeff_func import *
from trainnet import test
from loss import createLossAndOptimizer,createLossAndOptimizer_ablation
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
workpath='workpath/final10split_2/ourmodel1/2/'
pairspath='data/map_detect_mean/2'

pt=workpath+'checkpoint_best/ModelParams_Best_val.pt'
backbone='ours'
checkpoint = torch.load(pt)
cube = checkpoint['cube']
net = Siamese().cuda()
allpairs = np.genfromtxt(open(pairspath+'/test.csv'), delimiter=',', dtype=str)
noalignpairs = np.genfromtxt(open(pairspath+'/test_notaligned.csv'), delimiter=',', dtype=str)
alignpairs = np.genfromtxt(open(pairspath+'/test_aligned.csv'), delimiter=',', dtype=str)
training_datadir = '/home/zhihua/CD_metric/database/images'
val_split =0

all_test1024 = test_1024(allpairs[val_split:], root_dir=training_datadir, test=True)
noaligndata_test1024 = test_1024(noalignpairs[val_split:], root_dir=training_datadir, test=True)
aligndata_test1024 = test_1024(alignpairs[val_split:], root_dir=training_datadir, test=True)
test_loader1024 = DataLoader(all_test1024, batch_size=8, shuffle=False, pin_memory=True, num_workers=8)
noaligndata_test_loader1024 = DataLoader(noaligndata_test1024, batch_size=8, shuffle=False, pin_memory=True, num_workers=8)
aligndata_test_loader1024 = DataLoader(aligndata_test1024, batch_size=8, shuffle=False, pin_memory=True, num_workers=8)
net.load_state_dict(checkpoint['state_dict'])
net.eval()
dist = []
y_true = []

if backbone=='ours':
    loss, optimizer, scheduler = createLossAndOptimizer(net, 0, 'mse', 1, 0.1, cube)
else:
    loss, optimizer, scheduler = createLossAndOptimizer_ablation(net, 0, 'mse', 1, 0.1)
srocc_v0,total_val_loss, val_counter,cc_v0, krocc_v, rmse_v, stress0, dist0, y_true0=test(test_loader1024,'rgb',net,cube,loss,backbone)
print("allTest 1024: PCC{:.5}, SROCC{:.5}, STRESS{:.6}, RMSE{:.5}".format(cc_v0, srocc_v0, stress0, rmse_v))

srocc_v1,total_val_loss, val_counter,cc_v1, krocc_v, rmse_v, stress1, dist1, y_true1=test(noaligndata_test_loader1024,'rgb',net,cube,loss,backbone)
print("not aligned Test1024: PCC{:.5}, SROCC{:.6}, STRESS{:.5}, RMSE{:.5}".format(cc_v1, srocc_v1, stress1, rmse_v))

srocc_v3,total_val_loss, val_counter,cc_v3, krocc_v, rmse_v, stress3, dist3, y_true3=test(aligndata_test_loader1024,'rgb',net,cube,loss,backbone)
print("align Test1024: PCC{:.4}, SROCC{:.4}, STRESS{:.4}, RMSE{:.4}".format(cc_v3, srocc_v3, stress3, rmse_v))
prediction = pd.DataFrame(columns=['dv'])
performance = pd.DataFrame(columns=['stress','plcc','srcc'])
prediction.loc[:, 'dv'] = y_true0
prediction.loc[:, 'allTest 1024'] = dist0
prediction.to_csv(workpath+'prediction_all.csv', index=None)
prediction = pd.DataFrame(columns=['dv'])
prediction.loc[:, 'dv'] = y_true1
prediction.loc[:, 'not aligned Test1024'] = dist1
prediction.to_csv(workpath+'prediction_notalign.csv', index=None)
prediction = pd.DataFrame(columns=['dv'])
prediction.loc[:, 'dv'] = y_true3
prediction.loc[:, 'align Test1024'] = dist3
prediction.to_csv(workpath+'prediction_align.csv', index=None)
performance.loc['{}'.format(1),'stress']=stress0
performance.loc['{}'.format(1),'plcc']=cc_v0
performance.loc['{}'.format(1),'srcc']=srocc_v0
performance.loc['{}'.format(2),'stress']=stress1
performance.loc['{}'.format(2),'plcc']=cc_v1
performance.loc['{}'.format(2),'srcc']=srocc_v1
performance.loc['{}'.format(3),'stress']=stress3
performance.loc['{}'.format(3),'plcc']=cc_v3
performance.loc['{}'.format(3),'srcc']=srocc_v3
performance.to_csv(workpath+'performance.csv', index=None)

