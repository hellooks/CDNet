import os
from trainnet import trainNet
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import argparse
import torch
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default = 100)  
    parser.add_argument("--resume_path", type=str, default = None) 
    parser.add_argument("--lossname", type=str, default='mse') 
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--scheduler_step", type=int, default=20)
    parser.add_argument("--scheduler_gamma", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_epochs", type=int, default=50)

    parser.add_argument("--training_datadir", type=str, default= '/home/zhihua/CD_metric/database/images')
    parser.add_argument("--colorspace", type=str, default='rgb') #'rgb2xyz' 'rgb' 'rgb2yiq'
    parser.add_argument("--trainpath1", type=str, default='trainnet.py')
    parser.add_argument("--trainpath2", type=str, default='main.py')
    parser.add_argument("--trainpath3", type=str, default='model_complex9.py')
    parser.add_argument("--work_path", type=str, default='workpath/table4/E94')
    parser.add_argument('--backbone',type=str,default='ours') #'resnet18''ours''resnet34''vgg16''resnext101'
    
    parser.add_argument("--datapath", type=str, default='data/E00_map_detect_mean')
    parser.add_argument("--trainset", type=str, default='train.csv')
    parser.add_argument("--valset", type=str, default='val.csv')
    parser.add_argument("--testset", type=str, default='test.csv')
    parser.add_argument("--test_aligned_path", type=str, default=None)
    parser.add_argument("--test_notaligned_path", type=str, default=None)

    return parser.parse_args()

if __name__ == '__main__':
    config = parse_config()
    path = config.datapath
    modelprediction = pd.DataFrame(columns=['no'])
    modelprediction_aligned=pd.DataFrame(columns=['no'])
    modelprediction_notaligned=pd.DataFrame(columns=['no'])
    work_path=config.work_path
    trainpath=config.trainset
    valpath=config.valset
    testpath=config.testset
    performance = pd.DataFrame(columns=['stress','plcc','srcc','stress_aligned','plcc_aligned','srcc_aligned','stress_notaligned','plcc_notaligned','srcc_notaligned'])
    if config.resume_path is not None:
        checkpoint = torch.load(config.resume_path)
        times = checkpoint['times']
        checkpoint_startshuffle=times
        for i in range(times, 10):
            print('------------------------------------shuffle{}---------------------------------'.format(i+1))
            config.datapath = path+'/{}.csv'.format(i+1)
            config.work_path = work_path+'/{}'.format(i+1)
            config.trainset= path+'/{}/'.format(i+1)+trainpath
            config.valset= path+'/{}/'.format(i+1)+valpath
            config.testset= path+'/{}/'.format(i+1)+testpath
            config.test_aligned_path = path+'/{}/test_aligned.csv'.format(i+1)
            config.test_notaligned_path = path+'/{}/test_notaligned.csv'.format(i+1)
            if i==checkpoint_startshuffle:
                dist1, y_true1, stress1, cc_v1, srocc_v1, dist2,\
                y_true2, stress2, cc_v2, srocc_v2, dist3,y_true3,stress3,cc_v3, srocc_v3 = trainNet(config,i,mark=1)
            else:
                dist1, y_true1, stress1, cc_v1, srocc_v1, dist2, y_true2, stress2, cc_v2, srocc_v2, \
                dist3, y_true3, stress3, cc_v3, srocc_v3 = trainNet(config, i)
            modelprediction.loc[:, '{}_1'.format(i + 1)] = dist1
            modelprediction.loc[:, '{}_2'.format(i + 1)] = y_true1
            modelprediction_aligned.loc[:, '{}_1'.format(i+1)] = dist2
            modelprediction_aligned.loc[:, '{}_2'.format(i + 1)] = y_true2
            modelprediction_notaligned.loc[:, '{}_1'.format(i+1)] = dist3
            modelprediction_notaligned.loc[:, '{}_2'.format(i + 1)] = y_true3
            modelprediction.to_csv(config.work_path + '/modelprediction.csv', index=None)
            performance.loc['{}'.format(i),'stress']=stress1
            performance.loc['{}'.format(i),'plcc']=cc_v1
            performance.loc['{}'.format(i),'srcc']=srocc_v1
            performance.loc['{}'.format(i),'stress_aligned']=stress2
            performance.loc['{}'.format(i),'plcc_aligned']=cc_v2
            performance.loc['{}'.format(i),'srcc_aligned']=srocc_v2
            performance.loc['{}'.format(i),'stress_notaligned']=stress3
            performance.loc['{}'.format(i),'plcc_notaligned']=cc_v3
            performance.loc['{}'.format(i),'srcc_notaligned']=srocc_v3
            performance.to_csv(config.work_path + '/modelperformance.csv', index=None)


    else:
        for i in range(1,2):
            print('------------------------------------shuffle{}---------------------------------'.format(i+1))
            config.datapath = path+'/{}.csv'.format(i+1)
            config.work_path = work_path+'/{}'.format(i+1)
            config.trainset= path+'/{}/'.format(i+1)+trainpath
            config.valset= path+'/{}/'.format(i+1)+valpath
            config.testset= path+'/{}/'.format(i+1)+testpath
            config.test_aligned_path = path+'/{}/test_aligned.csv'.format(i+1)
            config.test_notaligned_path = path+'/{}/test_notaligned.csv'.format(i+1)
            dist1, y_true1, stress1, cc_v1, srocc_v1, dist2, y_true2, stress2, cc_v2, srocc_v2,\
            dist3,y_true3,stress3,cc_v3, srocc_v3 = trainNet(config,i)
            modelprediction.loc[:, '{}_1'.format(i+1)] = dist1
            modelprediction.loc[:, '{}_2'.format(i + 1)] = y_true1
            modelprediction_aligned.loc[:, '{}_1'.format(i+1)] = dist2
            modelprediction_aligned.loc[:, '{}_2'.format(i + 1)] = y_true2
            modelprediction_notaligned.loc[:, '{}_1'.format(i+1)] = dist3
            modelprediction_notaligned.loc[:, '{}_2'.format(i + 1)] = y_true3
            modelprediction.to_csv(config.work_path + '/modelprediction.csv', index=None)
            performance.loc['{}'.format(i),'stress']=stress1
            performance.loc['{}'.format(i),'plcc']=cc_v1
            performance.loc['{}'.format(i),'srcc']=srocc_v1
            performance.loc['{}'.format(i),'stress_aligned']=stress2
            performance.loc['{}'.format(i),'plcc_aligned']=cc_v2
            performance.loc['{}'.format(i),'srcc_aligned']=srocc_v2
            performance.loc['{}'.format(i),'stress_notaligned']=stress3
            performance.loc['{}'.format(i),'plcc_notaligned']=cc_v3
            performance.loc['{}'.format(i),'srcc_notaligned']=srocc_v3
            performance.to_csv(config.work_path + '/modelperformance.csv', index=None)
    modelprediction.to_csv(config.work_path + '/modelprediction.csv', index=None)
    performance.to_csv(config.work_path + '/modelperformance.csv', index=None)
        