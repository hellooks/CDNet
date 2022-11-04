import torch
import os
from shutil import copyfile
def initialcode(config):
    config.result_path = os.path.join(config.checkpoint, 'results')
    config.codes = os.path.join(config.checkpoint, 'codes')
    config.ckpt_path = os.path.join(config.checkpoint, 'checkpoints')
    config.preds_path = os.path.join(config.checkpoint, 'preds')
    config.runs_path = os.path.join(config.checkpoint, 'runs')
    if config.train ==True:
        copyfiles(config.codes)
        print('train------>backup')
    if not os.path.exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    if not os.path.exists(config.runs_path):
        os.makedirs(config.runs_path)
    if not os.path.exists(config.preds_path):
        os.makedirs(config.preds_path)
    return '----------------------------initial completed!!!!-------------------------------'

def get_latest_checkpoint(path):
    ckpts = os.listdir(path)
    ckpts = [ckpt for ckpt in ckpts if not os.path.isdir(os.path.join(path, ckpt))]
    all_times = sorted(ckpts, reverse=True)
    return os.path.join(path, all_times[0])

# save checkpoint
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def copyfiles(path):
    # create file folder if not exist
    if not os.path.exists(path):
        os.makedirs(path)
    copyfile('Main.py', os.path.join(path, 'Main.py'))
    copyfile('model.py', os.path.join(path, 'model.py'))
    copyfile('utils/ImageDataset.py', os.path.join(path, 'ImageDataset.py'))
    copyfile('TrainModel.py', os.path.join(path, 'TrainModel.py'))
    copyfile('utils/Transformers.py', os.path.join(path, 'Transformers.py'))
    copyfile('utils/MNL_Loss.py', os.path.join(path, 'MNL_Loss.py'))
    copyfile('utils/np_transforms.py', os.path.join(path, 'np_transforms.py'))
    copyfile('utils/utils.py', os.path.join(path, 'utils.py'))
    copyfile('ksrun.sh', os.path.join(path, 'ksrun.sh'))
    return 0
