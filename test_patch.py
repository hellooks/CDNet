import torch
from model import Siamese
from DataLoader import CD_npy as DataLoad_cd_npy
from colorspace_conversion import rgb2xyz,rgb2yiq,myRGB2Lab,xyz2rgb
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
import os
from coeff_func import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
pt='/home/zhihua/CD_metric/PAMI_CD1.2/workpath/zhihua/1*1+11*11dilation/5/2/checkpoint_best/ModelParams_Best_val.pt'
checkpoint = torch.load(pt)
cube = checkpoint['cube']
net = Siamese().cuda()
pairs = np.genfromtxt(open('data/patch/rit_npy.csv'), delimiter=',', dtype=str)
training_datadir = '/home/zhihua/CD_metric/database/patches/rit_p3rgb_npy'
val_split = 0
data_test = DataLoad_cd_npy(pairs[val_split:], root_dir=training_datadir, test=True)
data_test_loader = DataLoader(data_test, batch_size=4, shuffle=False, pin_memory=True, num_workers=8)
net.load_state_dict(checkpoint['state_dict'])
net.eval()
dist = []
y_true = []
count = 0

for _, data in enumerate(data_test_loader, 0):
    with torch.no_grad():
        # Get input images and ground truth dissimilarity score from the training dataloader
        refs_org, tests_org, gts = data
        y_val = gts.numpy()
        # Wrap the tensors in a Variable object
        refs_org, tests_org, gts = \
            Variable(refs_org).type(torch.cuda.FloatTensor), \
            Variable(tests_org).type(torch.cuda.FloatTensor), \
            Variable(gts).type(torch.cuda.FloatTensor)
        output = net(refs_org, tests_org, cube)
        count += 1
        print(count)
        pred = (torch.squeeze(output)).cpu().detach().numpy()
        for elm in pred:
            dist.append(elm)
        for elm in y_val:
            y_true.append(elm)
dist = np.array(dist)
y_true = np.array(y_true)
np.savetxt('wrong.txt', dist)
np.savetxt('y_true.txt', y_true)
stress = compute_stress(dist,y_true)
_, cc_v, srocc_v, krocc_v, rmse_v = coeff_fit(dist, y_true)
print("Test set: PCC{:.4}, SROCC{:.4}, STRESS{:.4}, RMSE{:.4}".format(cc_v, srocc_v, stress, rmse_v))
new = pd.DataFrame(columns=['score'], data=dist)
new.loc[:,'true']=y_true
new.to_csv('prediction/rit_pre.csv', index=None)
