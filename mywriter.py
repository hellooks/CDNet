from torch.utils.tensorboard import SummaryWriter

def writertest(writer,total_val_loss,val_counter,epoch,cc_v, srocc_v, stress, rmse_v,testsrcc):
    writer.add_scalar('Loss/test', total_val_loss / val_counter, epoch)
    writer.add_scalars('acc/test', {'PCC': cc_v}, epoch)
    writer.add_scalars('acc/test', {'SROCC': srocc_v}, epoch)
    writer.add_scalars('acc/test', {'STRESS': stress}, epoch)
    writer.add_scalars('acc/test', {'RMSE': rmse_v}, epoch)
    writer.add_scalars('srcc', {'test': testsrcc}, epoch)

def writerval(writer,total_val_loss,val_counter,epoch,cc_v,srocc_v,stress,rmse_v,optimizer,valsrcc ):
    writer.add_scalar('Loss/validation', total_val_loss / val_counter, epoch)
    writer.add_scalars('acc/val', {'PCC': cc_v}, epoch)
    writer.add_scalars('acc/val', {'SROCC': srocc_v}, epoch)
    writer.add_scalars('acc/val', {'STRESS': stress}, epoch)
    writer.add_scalars('acc/val', {'RMSE': rmse_v}, epoch)
    writer.add_scalar('lr=', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
    writer.add_scalars('srcc', {'val': valsrcc}, epoch)

def writertrain(writer,total_train_loss,train_counter,epoch,cc_v,srocc_v, rmse_v ):
    writer.add_scalar('Loss/train', total_train_loss / train_counter, epoch)
    writer.add_scalars('acc/train', {'PCC': cc_v}, epoch)
    writer.add_scalars('acc/train', {'SROCC': srocc_v}, epoch)
    writer.add_scalars('acc/train', {'RMSE': rmse_v}, epoch)