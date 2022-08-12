import torch
from model import Siamese
from DataLoader import CD_128 as DataLoad_cd_npy
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
            # if backbone=='ours':
            #     output = net(refs_org, tests_org, cube)
            # else:
            #     output = net(refs_org, tests_org)
            output = net(refs_org, tests_org, cube)
            ################################################################################
            # loss_size = loss(output, gts)
            # total_val_loss += loss_size.cpu().numpy()
            val_counter += 1
            pred = (torch.squeeze(output)).cpu().detach().numpy().tolist()
            # Store prediction values for future correlation calculation.
            # for elm in pred:
            #     dist.append(elm) 
            # # for elm in y_val:
            #     y_true.append(elm)
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

performance = pd.DataFrame(columns=['a_stress','a_plcc','a_srcc','n_stress','n_plcc','n_srcc','all_stress','all_plcc','all_srcc'])
for i in range(1,11):
    prediction_all = pd.DataFrame(columns=['dv'])
    prediction_a = pd.DataFrame(columns=['dv'])
    prediction_n = pd.DataFrame(columns=['dv'])
    pt='/home/zhihua/CD_metric/PAMI_CD1.2/workpath/10split/10splitresults/{}/checkpoint_best/ModelParams_Best_val.pt'.format(i)
    checkpoint = torch.load(pt)
    cube = checkpoint['cube']
    net = Siamese().cuda()
    pairs = np.genfromtxt(open('data/map_detect_mean/{}/test.csv'.format(i)), delimiter=',', dtype=str)
    alignedpairs = np.genfromtxt(open('data/map_detect_mean/{}/test_aligned.csv'.format(i)), delimiter=',', dtype=str)
    notalignedpairs = np.genfromtxt(open('data/map_detect_mean/{}/test_notaligned.csv'.format(i)), delimiter=',', dtype=str)
    training_datadir = '/home/zhihua/CD_metric/database/images'
    data_test = DataLoad_cd_npy(pairs[:], root_dir=training_datadir, test=True)
    data_test_aligned = DataLoad_cd_npy(alignedpairs[:], root_dir=training_datadir, test=True)
    data_test_notaligned = DataLoad_cd_npy(notalignedpairs[:], root_dir=training_datadir, test=True)
    
    data_test_loader = DataLoader(data_test, batch_size=12, shuffle=False, pin_memory=True, num_workers=8)
    data_test_loader_a = DataLoader(data_test_aligned, batch_size=12, shuffle=False, pin_memory=True, num_workers=8)
    data_test_loader_n = DataLoader(data_test_notaligned, batch_size=12, shuffle=False, pin_memory=True, num_workers=8)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    score=[]
    l_true=[]
    count = 0
    srocc_v, val_counter,cc_v, krocc_v, rmse_v, stress, dist, y_true=test(data_test_loader,net,cube)
    performance.loc['{}'.format(i),'all_stress']=stress
    performance.loc['{}'.format(i),'all_plcc']=cc_v
    performance.loc['{}'.format(i),'all_srcc']=srocc_v
    prediction_all.loc[:, 'dv'] = y_true
    prediction_all.loc[:, '{}'.format(i)] = dist

    srocc_v, val_counter,cc_v, krocc_v, rmse_v, stress, dist, y_true=test(data_test_loader_a,net,cube)
    performance.loc['{}'.format(i),'a_stress']=stress
    performance.loc['{}'.format(i),'a_plcc']=cc_v
    performance.loc['{}'.format(i),'a_srcc']=srocc_v
    prediction_a.loc[:, 'dv'] = y_true
    prediction_a.loc[:, '{}'.format(i)] = dist
   
    srocc_v, val_counter,cc_v, krocc_v, rmse_v, stress, dist, y_true=test(data_test_loader_n,net,cube)
    performance.loc['{}'.format(i),'n_stress']=stress
    performance.loc['{}'.format(i),'n_plcc']=cc_v
    performance.loc['{}'.format(i),'n_srcc']=srocc_v

    prediction_n.loc[:, 'dv'] = y_true
    prediction_n.loc[:, '{}'.format(i)] = dist
    
    performance.to_csv('workpath/10split/stress_ourdataset.csv', index=None)
    prediction_all.to_csv('workpath/10split/10split_prediction/test_all{}.csv'.format(i),index=None)
    prediction_a.to_csv('workpath/10split/10split_prediction/test_aligned{}.csv'.format(i),index=None)
    prediction_n.to_csv('workpath/10split/10split_prediction/test_notaligned{}.csv'.format(i),index=None)
