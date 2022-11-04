import argparse
import os, torch, random 
import numpy as np
import TrainModel
from utils.utils import initialcode
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=True)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=19981025)
    parser.add_argument("--files", type=str, default="data/Eab/2/")
    parser.add_argument("--image_path", type=str, default="/media/h428ti/SSD/image_final/image")
    parser.add_argument('--train_file', type=str, default='train.csv')
    parser.add_argument('--val_file', type=str, default='val.csv')
    parser.add_argument('--test_file', type=str, default='test.csv')
    parser.add_argument('--testaligned_file', type=str, default='test_aligned.csv')
    parser.add_argument('--testnotaligned_file', type=str, default='test_notaligned.csv')
    parser.add_argument('--checkpoint', default="modeltest", type=str, metavar='PATH', help='path to checkpoints')
    parser.add_argument('--ckpt', default=None, type=str, help='checkpoint path')
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--fz", type=bool, default=True)
    parser.add_argument("--epochs_per_eval", type=int, default=1)
    parser.add_argument("--epochs_per_save", type=int, default=1)
    parser.add_argument("--scheduler_milestones", type=int,nargs='+', default=[80])
    parser.add_argument("--scheduler_gamma", type=float, default=0.5)
    parser.add_argument("--projection", type=int, default=128)
    return parser.parse_args()

def main(cfg):
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)  
    t = TrainModel.Trainer(cfg)
    if cfg.train:
        # print('train mode:{}'.format(cfg.train_txt))
        print(cfg)
        print('start training')
        t.fit()
    else:
        print('test mode')
        print('start testing')
        t.evaleveryepoch(epoch=1)
        return 0

if __name__ == "__main__":
    config = parse_config()
    config.train = True # modify when test
    print(initialcode(config=config))
    if config.train:
        # config.scheduler_milestones=[1]
        # config.fz = True
        # config.batch_size = 8
        # config.resume = False
        # config.max_epochs = 1
        # main(config)

        config.scheduler_milestones=[5,10,15,20,30,35]
        # config.fz = False
        config.batch_size = 8
        config.resume = False  # resuming from the latest checkpoint of stage 1
        config.max_epochs = 50
        main(config)
    else:
        main(config)


