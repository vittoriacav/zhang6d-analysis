device=0
ngpu=1
logname='exp1-bottle'

CUDA_VISIBLE_DEVICES=$device python -m torch.distributed.launch --nproc_per_node=$ngpu --master_port 11112 train.py \
    --flagfile 'config/bottle_wild6d/base_config.txt' --logger 'tensorboard' --checkpoint_dir 'log' --name $logname \
    --train --ngpu $ngpu --save_freq 2000 --vis_freq 2000 --dataset_path ~/vittoria/kaifeng/self-corr-pose/data/Wild6D/bottle/
