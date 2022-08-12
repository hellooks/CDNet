import time
from turtle import back
from EMA import EMA
import torch
from torch.utils.data import DataLoader
from model import Siamese
from model_ablation import Siamese as Siamese_ablation
from DataLoader import CD_128,CD_resnext
from colorspace_conversion import rgb2xyz,rgb2yiq,myRGB2Lab
from coeff_func import *
import os
from loss import createLossAndOptimizer,createLossAndOptimizer_ablation
from mywriter import writertest,writerval,writertrain
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.tensorboard import SummaryWriter
from myfunction import  setup_seed, copy_codes

    
def trainNet(config,times,mark=0):
    resume_path = config.resume_path  # none | path to checkpiont
    lossname = config.lossname  # mae| mae| mae+mse
    learning_rate = config.learning_rate
    scheduler_step = config.scheduler_step
    scheduler_gamma = config.scheduler_gamma
    batch_size=config.batch_size
    n_epochs=config.n_epochs
    training_datadir = config.training_datadir
    colorspace = config.colorspace
    trainpath1 = config.trainpath1
    trainpath2 = config.trainpath2
    trainpath3 = config.trainpath3
    workspace = config.work_path
    backbone = config.backbone
    # set random seed
    setup_seed(config.seed)
    writer = SummaryWriter(os.path.join(workspace, 'runs'))
    if not os.path.exists(workspace):
        os.mkdir(workspace)
    if not os.path.exists(os.path.join(workspace, 'codes')):
        os.mkdir(os.path.join(workspace, 'codes'))
    if not os.path.exists(os.path.join(workspace, 'checkpoint')):
        os.mkdir(os.path.join(workspace, 'checkpoint'))
    if not os.path.exists(os.path.join(workspace, 'checkpoint_best')):
        os.mkdir(os.path.join(workspace, 'checkpoint_best'))
    copy_codes(trainpath1=trainpath1, trainpath2=trainpath2, trainpath3=trainpath3,
               path1=os.path.join(workspace, 'codes/trainNet.py'), \
               path2=os.path.join(workspace, 'codes/main.py'), path3=os.path.join(workspace, 'codes/net.py'))
    print("============ HYPERPARAMETERS ==========")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print('loss=', lossname)
    print('learning rate=', learning_rate)
    print('scheduler_step=', scheduler_step)
    print('scheduler_gamma=', scheduler_gamma)
    print('training dir=', training_datadir)
    print('colorspace=', colorspace)
    print(config.trainset)
    print(config.valset)
    print(config.testset)
    print(config.test_aligned_path)
    print(config.test_notaligned_path)
    train_pairs = np.genfromtxt(open(config.trainset,encoding='UTF-8-sig'), delimiter=',', dtype=str)
    val_pairs = np.genfromtxt(open(config.valset,encoding='UTF-8-sig'), delimiter=',', dtype=str)
    test_pairs = np.genfromtxt(open(config.testset,encoding='UTF-8-sig'), delimiter=',', dtype=str)
    test_aligned_pairs = np.genfromtxt(open(config.test_aligned_path), delimiter=',', dtype=str)
    test_notaligned_pairs = np.genfromtxt(open(config.test_notaligned_path), delimiter=',', dtype=str)

    #################################################################################
    if backbone =='ours':
        data_train = CD_128(train_pairs[:], root_dir=training_datadir, test=False)
        data_val = CD_128(val_pairs[:], root_dir=training_datadir, test=True)
        data_test = CD_128(test_pairs[:], root_dir=training_datadir, test=True)
        test_aligned = CD_128(test_aligned_pairs[:], root_dir=training_datadir, test=True)
        test_notaligned = CD_128(test_notaligned_pairs[:], root_dir=training_datadir, test=True)
        cube = Variable(torch.rand(12,12)).cuda()
        cube.requires_grad=True
        net = Siamese().cuda()
        loss, optimizer, scheduler = createLossAndOptimizer(net, learning_rate, lossname, scheduler_step, scheduler_gamma, cube)
    else:
        data_train = CD_resnext(train_pairs[:], root_dir=training_datadir, test=False)
        data_val = CD_resnext(val_pairs[:], root_dir=training_datadir, test=True)
        data_test = CD_resnext(test_pairs[:], root_dir=training_datadir, test=True)
        test_aligned = CD_resnext(test_aligned_pairs[:], root_dir=training_datadir, test=True)
        test_notaligned = CD_resnext(test_notaligned_pairs[:], root_dir=training_datadir, test=True)
        net = Siamese_ablation(backbone).cuda()
        cube=0
        loss, optimizer, scheduler = createLossAndOptimizer_ablation(net, learning_rate, lossname, scheduler_step, scheduler_gamma)

    data_train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
    data_val_loader = DataLoader(data_val, batch_size=6, shuffle=True, pin_memory=True, num_workers=8)
    data_test_loader = DataLoader(data_test, batch_size=6, shuffle=False, pin_memory=True, num_workers=8)
    data_test_aligned_loader = DataLoader(test_aligned, batch_size=6, shuffle=False, pin_memory=True, num_workers=8)
    data_test_notaligned_loader = DataLoader(test_notaligned, batch_size=6, shuffle=False, pin_memory=True, num_workers=8)

    # Create our loss and optimizer functions
    #################################################################################
    if resume_path is not None and mark==1:
        checkpoint = torch.load(resume_path)
        start_epoch = checkpoint['epoch']+1
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if backbone=='ours':
            cube = checkpoint['cube']
        print('continue to train: shuffle{} epoch{} '.format(times+1, start_epoch))
    else:
        start_epoch = 0
    # load checkpont from breakpoint
    ################################################################################
    training_start_time = time.time()
    rows, columns = train_pairs.shape
    n_batches = rows// batch_size
    testsrcc=0
    valsrcc=0
    ema = EMA(net, 0.999)
    ema.register()
    autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, n_epochs):
        # initiate parameters for statistic recordings.
        dist = []
        y_true = []
        running_loss = 0.0
        total_train_loss = 0
        start_time = time.time()
        print_every = 20
        train_counter = 0
        net.train()
        print("---------------------train mode-------epoch{}--------------------------".format(epoch))
        for i, data in enumerate(data_train_loader, 0):
            train_counter = train_counter + 1
            refs_org, tests_org,  gts = data
            y_val = gts.numpy()
            refs_org, tests_org, gts =\
                Variable(refs_org).type(torch.cuda.FloatTensor),  \
                Variable(tests_org).type(torch.cuda.FloatTensor),  \
                Variable(gts).type(torch.cuda.FloatTensor)
            # colorspace conversion
            ################################################################################
            if colorspace == 'rgb2xyz':
                refs_org = rgb2xyz(refs_org)
                tests_org = rgb2xyz(tests_org)
            if colorspace == 'rgb2yiq':
                refs_org = rgb2yiq(refs_org)
                tests_org = rgb2yiq(tests_org)
            if colorspace == 'rgb2lab':
                refs_org = myRGB2Lab(refs_org)
                tests_org = myRGB2Lab(tests_org)
            ################################################################################
            # Set the parameter gradients to zero
            optimizer.zero_grad()
            if config.backbone=='ours':
                output= net(refs_org, tests_org, cube)
            else:
                output= net(refs_org, tests_org)
            loss_size = loss(output, gts)
            loss_size.backward()
            optimizer.step()
            ema.update()
            running_loss += loss_size.item()
            total_train_loss += loss_size.item()
            pred = (torch.squeeze(output)).cpu().detach().numpy().tolist()
            if isinstance(pred,list):
                dist.extend(pred)
                y_true.extend(y_val.tolist())
            else:
                dist.append(np.array(pred))
                y_true.append(y_val)
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.6f} took: {:.2f}s".format(
                    epoch + 1, int(100 * (i + 1) / n_batches), running_loss / print_every, time.time() - start_time))
                # Reset running loss and time
                running_loss = 0.0
                start_time = time.time()
            if config.backbone=='ours':
                torch.save({"state_dict": net.state_dict(), 'epoch': epoch, 'optimizer': optimizer.state_dict(),'cube':cube,'times':times}, \
                os.path.join(workspace, 'checkpoint', 'ModelParams_checkpoint.pt'))
            else:
                torch.save({"state_dict": net.state_dict(), 'epoch': epoch, 'optimizer': optimizer.state_dict(),'times':times}, \
                 os.path.join(workspace, 'checkpoint', 'ModelParams_checkpoint.pt'))

        # Calculate correlation coefficients between the predicted values and ground truth values on training set.
        dist = np.array(dist)
        y_true = np.array(y_true)
        _, cc_v, srocc_v, krocc_v, rmse_v = coeff_fit(dist, y_true)
        print("Training set: PCC{:.4}, SROCC{:.4}, KROCC{:.4}, RMSE{:.4}".format(cc_v, srocc_v, krocc_v, rmse_v))
        writertrain(writer, total_train_loss, train_counter, epoch, cc_v, srocc_v, rmse_v)
        ################################################################################
        # validation
        ema.apply_shadow()
        net.eval()
        print("----------------------------validation mode---------------------------------")
        srocc_v, total_val_loss, val_counter, cc_v, krocc_v, rmse_v, stress, dist, y_true=test(data_val_loader,colorspace,net,cube,loss,backbone)
        # save best weight
        ################################################################################
        if srocc_v >valsrcc:
            valsrcc=srocc_v
            if backbone=='ours':
                torch.save({"state_dict": net.state_dict(),'cube':cube},os.path.join(workspace, 'checkpoint_best','ModelParams_Best_val.pt'))
            else:
                torch.save({"state_dict": net.state_dict()},os.path.join(workspace, 'checkpoint_best','ModelParams_Best_val.pt'))
            print('update  best model...')
        print("VALIDATION:  PCC{:.4}, SROCC{:.4}, STRESS{:.4}, RMSE{:.4}".format(cc_v, srocc_v, stress, rmse_v))
        print("loss = {:.6}".format(total_val_loss / val_counter))
        writerval(writer, total_val_loss, val_counter, epoch, cc_v, srocc_v, stress, rmse_v, optimizer, valsrcc)
        ema.restore()
        scheduler.step()
    print('#############################################################################')
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
    pt=workspace+'/checkpoint_best/'+'ModelParams_Best_val.pt'
    checkpoint = torch.load(pt)
    if backbone=='ours':
        cube = checkpoint['cube']
        net = Siamese().cuda()
        net.load_state_dict(checkpoint['state_dict'])
    else:
        net = Siamese_ablation(backbone).cuda()
        net.load_state_dict(checkpoint['state_dict'])
    srocc_v1,total_val_loss, val_counter,cc_v1, krocc_v, rmse_v, stress1, dist1, y_true1=test(data_test_loader,colorspace,net,cube,loss,backbone)
    print('best performance: plcc{} srcc{}'.format(cc_v1, srocc_v1))    
    srocc_v2,total_val_loss, val_counter,cc_v2, krocc_v, rmse_v, stress2, dist2, y_true2=test(data_test_aligned_loader,colorspace,net,cube,loss,backbone)
    print('best performance in Pixel-wise aligned: plcc{} srcc{}'.format(cc_v2, srocc_v2))
    srocc_v3,total_val_loss, val_counter,cc_v3, krocc_v, rmse_v, stress3, dist3, y_true3=test(data_test_notaligned_loader,colorspace,net,cube,loss,backbone)
    print('best performance in non-Pixel-wise aligned: plcc{} srcc{}'.format(cc_v3, srocc_v3))
    return dist1, y_true1, stress1, cc_v1, srocc_v1, dist2, y_true2, stress2, cc_v2, srocc_v2, dist3,y_true3,stress3,cc_v3, srocc_v3

def test(data_val_loader,colorspace,net,cube,loss,backbone):
    total_val_loss = 0
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
            if colorspace == 'rgb2xyz':
                refs_org = rgb2xyz(refs_org)
                tests_org = rgb2xyz(tests_org)
            if colorspace == 'rgb2yiq':
                refs_org = rgb2yiq(refs_org)
                tests_org = rgb2yiq(tests_org)
            if colorspace == 'rgb2lab':
                refs_org = myRGB2Lab(refs_org)
                tests_org = myRGB2Lab(tests_org)
            ###############################################################################
            if backbone=='ours':
                output = net(refs_org, tests_org, cube)
            else:
                output = net(refs_org, tests_org)
            ################################################################################
            loss_size = loss(output, gts)
            total_val_loss += loss_size.cpu().numpy()
            val_counter += 1
            pred = (torch.squeeze(output)).cpu().detach().numpy().tolist()
            if isinstance(pred,list):
                dist.extend(pred)
                y_true.extend(y_val.tolist())
            else:
                dist.append(np.array(pred))
                y_true.append(y_val)
    dist_np = np.array(dist)
    y_true_np = np.array(y_true)
    stress = compute_stress(dist_np,y_true_np)
    _, cc_v, srocc_v, krocc_v, rmse_v = coeff_fit(dist_np, y_true_np)
    return srocc_v,total_val_loss, val_counter,cc_v, krocc_v, rmse_v, stress, dist, y_true

