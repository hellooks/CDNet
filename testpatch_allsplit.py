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
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def test(data_val_loader,net,cube):
    # total_val_loss = 0
    val_counter = 0
    dist = []
    y_true = []
    for _, data in enumerate(data_val_loader, 0):
        with torch.no_grad():
            refs_org, tests_org, gts = data
            y_val = gts.numpy()
            refs_org, tests_org, gts = \
                Variable(refs_org).type(torch.cuda.FloatTensor), \
                Variable(tests_org).type(torch.cuda.FloatTensor), \
                Variable(gts).type(torch.cuda.FloatTensor)
            # colorspace conversion
            ################################################################################
            # if colorspace == 'rgb2xyz':
            #     refs_org = rgb2xyz(refs_org)
            #     tests_org = rgb2xyz(tests_org)
            # if colorspace == 'rgb2yiq':
            #     refs_org = rgb2yiq(refs_org)
            #     tests_org = rgb2yiq(tests_org)
            # if colorspace == 'rgb2lab':
            #     refs_org = myRGB2Lab(refs_org)
            #     tests_org = myRGB2Lab(tests_org)
            ###############################################################################
            output = net(refs_org, tests_org, cube)
            ################################################################################
            # loss_size = loss(output, gts)
            # total_val_loss += loss_size.cpu().numpy()
            val_counter += 1
            pred = (torch.squeeze(output)).cpu().detach().numpy().tolist()
            if isinstance(pred,list):
                dist.extend(pred)
                y_true.extend(y_val.tolist())
            else:
                dist.append(np.array(pred))
                y_true.append(y_val)
        # Calculate correlation coefficients between the predicted values and ground truth values on validation set.
    dist_np = np.array(dist)
    y_true_np = np.array(y_true)
    stress = compute_stress(dist_np,y_true_np)
    _, cc_v, srocc_v, krocc_v, rmse_v = coeff_fit(dist_np, y_true_np)
    return srocc_v, val_counter,cc_v, krocc_v, rmse_v, stress, dist, y_true
prediction_bfd = pd.DataFrame(columns=['dv'])
prediction_leeds = pd.DataFrame(columns=['dv'])
prediction_rit = pd.DataFrame(columns=['dv'])
prediction_witt = pd.DataFrame(columns=['dv'])
performance = pd.DataFrame(columns=['bfd','leeds','rit','witt'])
for i in range(1,11):
    pt='/home/zhihua/CD_metric/PAMI_CD1.2/workpath/10split/10splitresults/{}/checkpoint_best/ModelParams_Best_val.pt'.format(i)
    checkpoint = torch.load(pt)
    cube = checkpoint['cube']
    net = Siamese().cuda()
    bfdpairs = np.genfromtxt(open('data/patch/bfd_npy.csv'), delimiter=',', dtype=str)
    bfdcpairs = np.genfromtxt(open('data/patch/bfdc_npy.csv'), delimiter=',', dtype=str)
    bfdmpairs = np.genfromtxt(open('data/patch/bfdm_npy.csv'), delimiter=',', dtype=str)
    leedspairs = np.genfromtxt(open('data/patch/leeds_npy.csv'), delimiter=',', dtype=str)
    wittpairs = np.genfromtxt(open('data/patch/witt_npy.csv'), delimiter=',', dtype=str)
    ritpairs = np.genfromtxt(open('data/patch/rit_npy.csv'), delimiter=',', dtype=str)
    training_datadir_bfd = '/home/zhihua/CD_metric/database/patches/bfd_p3rgb_npy'
    training_datadir_bfdc = '/home/zhihua/CD_metric/database/patches/bfdc_p3rgb_npy'
    training_datadir_bfdm = '/home/zhihua/CD_metric/database/patches/bfdm_p3rgb_npy'
    training_datadir_leeds = '/home/zhihua/CD_metric/database/patches/leeds_p3rgb_npy'
    training_datadir_witt = '/home/zhihua/CD_metric/database/patches/witt_p3rgb_npy'
    training_datadir_rit = '/home/zhihua/CD_metric/database/patches/rit_p3rgb_npy'
    data_test_bfd = DataLoad_cd_npy(bfdpairs[:], root_dir=training_datadir_bfd, test=True)
    data_test_bfdc = DataLoad_cd_npy(bfdcpairs[:], root_dir=training_datadir_bfdc, test=True)
    data_test_bfdm = DataLoad_cd_npy(bfdmpairs[:], root_dir=training_datadir_bfdm, test=True)
    data_test_leeds = DataLoad_cd_npy(leedspairs[:], root_dir=training_datadir_leeds, test=True)
    data_test_witt = DataLoad_cd_npy(wittpairs[:], root_dir=training_datadir_witt, test=True)
    data_test_rit = DataLoad_cd_npy(ritpairs[:], root_dir=training_datadir_rit, test=True)
    data_test_loader_bfd = DataLoader(data_test_bfd, batch_size=4, shuffle=False, pin_memory=True, num_workers=8)
    data_test_loader_bfdc = DataLoader(data_test_bfdc, batch_size=4, shuffle=False, pin_memory=True, num_workers=8)
    data_test_loader_bfdm = DataLoader(data_test_bfdm, batch_size=4, shuffle=False, pin_memory=True, num_workers=8)
    data_test_loader_leeds = DataLoader(data_test_leeds, batch_size=4, shuffle=False, pin_memory=True, num_workers=8)
    data_test_loader_rit = DataLoader(data_test_rit, batch_size=4, shuffle=False, pin_memory=True, num_workers=8)
    data_test_loader_witt = DataLoader(data_test_witt, batch_size=4, shuffle=False, pin_memory=True, num_workers=8)
    net.load_state_dict(checkpoint['state_dict'])
    
    net.eval()
    score=[]
    l_true=[]
    count = 0
    srocc_v, val_counter,cc_v, krocc_v, rmse_v, stress, dist, y_true=test(data_test_loader_bfd,net,cube)
    score.extend(dist)
    l_true.extend(y_true)
    srocc_v, val_counter,cc_v, krocc_v, rmse_v, stress, dist, y_true=test(data_test_loader_bfdc,net,cube)
    score.extend(dist)
    l_true.extend(y_true)
    srocc_v, val_counter,cc_v, krocc_v, rmse_v, stress, dist, y_true=test(data_test_loader_bfdm,net,cube)
    score.extend(dist)
    l_true.extend(y_true)
    score_np = np.array(score)
    l_true_np = np.array(l_true)
    stress = compute_stress(score_np,l_true_np)
    performance.loc['{}'.format(i),'bfd']=stress
    if i==1:
        prediction_bfd.loc[:, 'dv'] = l_true
    prediction_bfd.loc[:, '{}'.format(i)] = score
    ##########################################################################
    srocc_v, val_counter,cc_v, krocc_v, rmse_v, stress, dist, y_true=test(data_test_loader_leeds,net,cube)
    performance.loc['{}'.format(i),'leeds']=stress
    if i==1:
        prediction_leeds.loc[:, 'dv'] = dist
    prediction_leeds.loc[:, '{}'.format(i)] = y_true
    srocc_v, val_counter,cc_v, krocc_v, rmse_v, stress, dist, y_true=test(data_test_loader_rit,net,cube)
    performance.loc['{}'.format(i),'rit']=stress
    if i==1:
        prediction_rit.loc[:, 'dv'] = dist
    prediction_rit.loc[:, '{}'.format(i)] = y_true
    srocc_v, val_counter,cc_v, krocc_v, rmse_v, stress, dist, y_true=test(data_test_loader_witt,net,cube)
    performance.loc['{}'.format(i),'witt']=stress
    if i==1:
        prediction_witt.loc[:, 'dv'] = dist
    prediction_witt.loc[:, '{}'.format(i)] = y_true
    performance.to_csv('workpath/10split/stress_patch.csv', index=None)
    prediction_bfd.to_csv('workpath/10split/bfd.csv')
    prediction_leeds.to_csv('workpath/10split/leeds.csv')
    prediction_rit.to_csv('workpath/10split/rit.csv')
    prediction_witt.to_csv('workpath/10split/witt.csv')
