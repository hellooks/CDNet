dir="cdnet"
if [ ! -d "$dir" ];then 
mkdir $dir
echo "Create files successfully."
else
echo "Files have existed"
fi

CUDA_VISIBLE_DEVICES=0 nohup python3 -u Main.py \
--lr 1e-3 \
--batch_size 128 \
--scheduler_gamma 0.5 \
--epochs_per_save 1 \
--epochs_per_eval 1 \
--projection 128 \
--checkpoint $dir\
> $dir'/checkpoint_continue.txt' 2>&1 &