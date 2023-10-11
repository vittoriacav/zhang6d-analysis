device=1
ngpu=1

## per prove
# logname='PROVA'
# CUDA_VISIBLE_DEVICES=$device python -m torch.distributed.launch --nproc_per_node=$ngpu --master_port $RANDOM train_vitto.py \
#     --flagfile 'config/bottle_wild6d/base_config_PROVA.txt' --logger 'tensorboard' --checkpoint_dir 'log' --name $logname \
#     --train --ngpu $ngpu --save_freq 1 --vis_freq 1 --dataset_path ~/vittoria/kaifeng/self-corr-pose/data/wild6d/bottle1o_mixRot_DEPTH/ --rot_enc_post


## original Kaifeng model trained on original dataset
# logname='exp19-kaifAUG_camera'
# CUDA_VISIBLE_DEVICES=$device python -m torch.distributed.launch --nproc_per_node=$ngpu --master_port $RANDOM train_vitto.py \
#     --flagfile 'config/camera_wild6d/base_config.txt' --logger 'tensorboard' --checkpoint_dir 'log' --name $logname \
#     --train --ngpu $ngpu --save_freq 2000 --vis_freq 2000 --vis_freq2 100 --dataset_path ~/vittoria/kaifeng/self-corr-pose/data/wild6d/camera_mixRot/


## original Kaifeng model trained on augmented dataset
# logname='exp2-kaifengAUG'
# CUDA_VISIBLE_DEVICES=$device python -m torch.distributed.launch --nproc_per_node=$ngpu --master_port $RANDOM train_vitto.py \
#     --flagfile 'config/bottle_wild6d/base_config.txt' --logger 'tensorboard' --checkpoint_dir 'log' --name $logname \
#     --train --ngpu $ngpu --save_freq 2000 --vis_freq 2000 --vis_freq2 100 --dataset_path ~/vittoria/kaifeng/self-corr-pose/data/wild6d/bottle_mixRot/\

## rotate canonical mesh before being fed to shape predictor
# logname='exp3-RotEncPreAUG'
# CUDA_VISIBLE_DEVICES=$device python -m torch.distributed.launch --nproc_per_node=$ngpu --master_port $RANDOM train_vitto.py \
#     --flagfile 'config/bottle_wild6d/base_config.txt' --logger 'tensorboard' --checkpoint_dir 'log' --name $logname \
#     --train --ngpu $ngpu --save_freq 2000 --vis_freq 2000 --vis_freq2 100 --dataset_path ~/vittoria/kaifeng/self-corr-pose/data/wild6d/bottle_mixRot/\
#     --rot_enc

## rotate predicted mesh after being computed from shape predictor
# logname='exp4-RotEncPostAUG'
# CUDA_VISIBLE_DEVICES=$device python -m torch.distributed.launch --nproc_per_node=$ngpu --master_port $RANDOM train_vitto.py \
#     --flagfile 'config/bottle_wild6d/base_config.txt' --logger 'tensorboard' --checkpoint_dir 'log' --name $logname \
#     --train --ngpu $ngpu --save_freq 2000 --vis_freq 500 --vis_freq2 100 --dataset_path ~/vittoria/kaifeng/self-corr-pose/data/wild6d/bottle_mixRot/\
#     --rot_enc_post

## original Kaifeng model trained on all +90° images
# logname='exp5-kaifeng90'
# CUDA_VISIBLE_DEVICES=$device python -m torch.distributed.launch --nproc_per_node=$ngpu --master_port $RANDOM train_vitto.py \
#     --flagfile 'config/bottle_wild6d/base_config.txt' --logger 'tensorboard' --checkpoint_dir 'log' --name $logname \
#     --train --ngpu $ngpu --save_freq 2000 --vis_freq 2000 --vis_freq2 100 --dataset_path ~/vittoria/kaifeng/self-corr-pose/data/wild6d/bottle_90/\

## original Kaifeng model trained on augmented dataset
# logname='exp6-kaifengAUG_MiDaS'
# CUDA_VISIBLE_DEVICES=$device python -m torch.distributed.launch --nproc_per_node=$ngpu --master_port $RANDOM train_vitto.py \
#     --flagfile 'config/bottle_wild6d/base_config.txt' --logger 'tensorboard' --checkpoint_dir 'log' --name $logname \
#     --train --ngpu $ngpu --save_freq 2000 --vis_freq 2000 --vis_freq2 100 --dataset_path ~/vittoria/kaifeng/self-corr-pose/data/wild6d/bottle_mixRot_DEPTH/

## original Kaifeng model trained on augmented dataset --> using directly last 512-D feature vector of resnet and not transforming it in the ushape: Ushape=img_code
# logname='exp16-kaifAUG_laptop_512code'
# CUDA_VISIBLE_DEVICES=$device python -m torch.distributed.launch --nproc_per_node=$ngpu --master_port $RANDOM train_vitto.py \
#     --flagfile 'config/laptop_wild6d/base_config.txt' --logger 'tensorboard' --checkpoint_dir 'log' --name $logname \
#     --train --ngpu $ngpu --save_freq 2000 --vis_freq 2000 --vis_freq2 100 --dataset_path ~/vittoria/kaifeng/self-corr-pose/data/wild6d/laptop_mixRot/ \
#     --codedim 512

## no Ushape at all
# logname='exp17-kaifAUG_laptop_noUshape'
# CUDA_VISIBLE_DEVICES=$device python -m torch.distributed.launch --nproc_per_node=$ngpu --master_port $RANDOM train_vitto.py \
#     --flagfile 'config/laptop_wild6d/base_config.txt' --logger 'tensorboard' --checkpoint_dir 'log' --name $logname \
#     --train --ngpu $ngpu --save_freq 2000 --vis_freq 2000 --vis_freq2 100 --dataset_path ~/vittoria/kaifeng/self-corr-pose/data/wild6d/laptop_mixRot/ \
#     --no_deform

## original Kaifeng model trained on original dataset with MiDaS
# logname='exp9-kaifeng_MiDaS'
# CUDA_VISIBLE_DEVICES=$device python -m torch.distributed.launch --nproc_per_node=$ngpu --master_port $RANDOM train_vitto.py \
#     --flagfile 'config/bottle_wild6d/base_config.txt' --logger 'tensorboard' --checkpoint_dir 'log' --name $logname \
#     --train --ngpu $ngpu --save_freq 2000 --vis_freq 2000 --vis_freq2 100 --dataset_path ~/vittoria/kaifeng/self-corr-pose/data/wild6d/bottle_DEPTH/\


######## NUOVI ESPERIMENTI ########

## LAST 2 FOR THE CAMERA

## original Kaifeng model trained on augmented dataset --> using directly last 512-D feature vector of resnet and not transforming it in the ushape: Ushape=img_code
#logname='exp20-kaifAUG_camera_512code'
#CUDA_VISIBLE_DEVICES=$device python -m torch.distributed.launch --nproc_per_node=$ngpu --master_port $RANDOM train_vitto.py \
#    --flagfile 'config/camera_wild6d/base_config.txt' --logger 'tensorboard' --checkpoint_dir 'log' --name $logname \
#    --train --ngpu $ngpu --save_freq 2000 --vis_freq 2000 --vis_freq2 100 --dataset_path ~/vittoria/kaifeng/self-corr-pose/data/wild6d/camera_mixRot/ \
#    --codedim 512

## no Ushape at all
# logname='exp21-kaifAUG_camera_noUshape'
# CUDA_VISIBLE_DEVICES=$device python -m torch.distributed.launch --nproc_per_node=$ngpu --master_port $RANDOM train_vitto.py \
#    --flagfile 'config/camera_wild6d/base_config.txt' --logger 'tensorboard' --checkpoint_dir 'log' --name $logname \
#    --train --ngpu $ngpu --save_freq 2000 --vis_freq 2000 --vis_freq2 100 --dataset_path ~/vittoria/kaifeng/self-corr-pose/data/wild6d/camera_mixRot/ \
#    --no_deform

## 4 EXPERIMENTS FOR THE BOWL

## original Kaifeng model trained on original dataset
#logname='exp22-kaif_bowl'
#CUDA_VISIBLE_DEVICES=$device python -m torch.distributed.launch --nproc_per_node=$ngpu --master_port $RANDOM train_vitto.py \
#    --flagfile 'config/bowl_wild6d/base_config.txt' --logger 'tensorboard' --checkpoint_dir 'log' --name $logname \
#    --train --ngpu $ngpu --save_freq 2000 --vis_freq 2000 --vis_freq2 100 --dataset_path ~/vittoria/kaifeng/self-corr-pose/data/wild6d/bowl/

## original Kaifeng model trained on augmented dataset
#  logname='exp23-kaifAUG_bowl'
#  CUDA_VISIBLE_DEVICES=$device python -m torch.distributed.launch --nproc_per_node=$ngpu --master_port $RANDOM train_vitto.py \
#     --flagfile 'config/bowl_wild6d/base_config.txt' --logger 'tensorboard' --checkpoint_dir 'log' --name $logname \
#     --train --ngpu $ngpu --save_freq 2000 --vis_freq 2000 --vis_freq2 100 --dataset_path ~/vittoria/kaifeng/self-corr-pose/data/wild6d/bowl_mixRot/

## original Kaifeng model trained on augmented dataset --> using directly last 512-D feature vector of resnet and not transforming it in the ushape: Ushape=img_code
# logname='exp24-kaifAUG_bowl_512code'
# CUDA_VISIBLE_DEVICES=$device python -m torch.distributed.launch --nproc_per_node=$ngpu --master_port $RANDOM train_vitto.py \
#     --flagfile 'config/bowl_wild6d/base_config.txt' --logger 'tensorboard' --checkpoint_dir 'log' --name $logname \
#     --train --ngpu $ngpu --save_freq 2000 --vis_freq 2000 --vis_freq2 100 --dataset_path ~/vittoria/kaifeng/self-corr-pose/data/wild6d/bowl_mixRot/ \
#     --codedim 512

## no Ushape at all
# logname='exp25-kaifAUG_bowl_noUshape'
# CUDA_VISIBLE_DEVICES=$device python -m torch.distributed.launch --nproc_per_node=$ngpu --master_port $RANDOM train_vitto.py \
#    --flagfile 'config/bowl_wild6d/base_config.txt' --logger 'tensorboard' --checkpoint_dir 'log' --name $logname \
#    --train --ngpu $ngpu --save_freq 2000 --vis_freq 2000 --vis_freq2 100 --dataset_path ~/vittoria/kaifeng/self-corr-pose/data/wild6d/bowl_mixRot/ \
#    --no_deform



