device=1
batch_size=16

checkpoint_dir='log'
# logname='exp2-BottleKNetMixRot'

# logname='PROVA'
# model_path="${checkpoint_dir}/${logname}/pred_net_1.pth"
# flagfile="${checkpoint_dir}/${logname}/config.txt"
# vis_path="${checkpoint_dir}/${logname}/TestMixRot/"

# CUDA_VISIBLE_DEVICES=$device python predict_vitto.py --flagfile $flagfile --local_rank -1 \
#     --test --ngpu 1 --model_path $model_path --name $logname --checkpoint_dir $checkpoint_dir --vis_path $vis_path \
#     --batch_size $batch_size --repeat 1 --num_workers 8 --dframe_eval 8 --use_depth --eval --eval_nocs \
#     --test_dataset_path ~/vittoria/kaifeng/self-corr-pose/data/wild6d/test_mixRot/bottle/ \
#     --test_list config/bottle_wild6d/test_list_subset1o.txt \
#     --vis_pred --visualize_gt --visualize_mesh --visualize_tex --visualize_conf --visualize_bbox --visualize_match 

# logname='exp10-kaif_mug'
# model_path="${checkpoint_dir}/${logname}/pred_net_20000.pth"
# flagfile="${checkpoint_dir}/${logname}/config.txt"
# vis_path="${checkpoint_dir}/${logname}/TestMixRot/"

# CUDA_VISIBLE_DEVICES=$device python predict_vitto.py --flagfile $flagfile --local_rank -1 \
#     --test --ngpu 1 --model_path $model_path --name $logname --checkpoint_dir $checkpoint_dir --vis_path $vis_path \
#     --batch_size $batch_size --repeat 1 --num_workers 8 --dframe_eval 8 --use_depth --eval --eval_nocs \
#     --test_dataset_path ~/vittoria/kaifeng/self-corr-pose/data/wild6d/test_mixRot/mug/ \
#     --test_list config/mug_wild6d/test_list_all.txt \
#     --vis_pred --visualize_gt --visualize_mesh --visualize_tex --visualize_conf --visualize_bbox --visualize_match 

# *******************************************************************************************************************************#

logname='exp1-kaifeng'
model_path="${checkpoint_dir}/${logname}/pred_net_20000.pth"
flagfile="${checkpoint_dir}/${logname}/config.txt"
vis_path="${checkpoint_dir}/${logname}/PROVA_canc/"

CUDA_VISIBLE_DEVICES=$device python predict_vitto.py --flagfile $flagfile --local_rank -1 \
    --test --ngpu 1 --model_path $model_path --name $logname --checkpoint_dir $checkpoint_dir --vis_path $vis_path \
    --batch_size $batch_size --repeat 1 --num_workers 8 --dframe_eval 8 --use_depth --eval --eval_nocs \
    --test_dataset_path ~/vittoria/kaifeng/self-corr-pose/data/wild6d/test_set_DEPTH16/bottle/ \
    --test_list config/bottle_wild6d/test_list_all.txt \
    --vis_pred --visualize_gt --visualize_mesh --visualize_tex --visualize_conf --visualize_bbox --visualize_match \
    --no_deform

# 36272635
# 36272635

# provare exp1-kaifeng con test set dritto con depth stimata
# 
# *******************************************************************************************************************************#

# kaifeng augmented
# logname='exp2-kaifengAUG'
# model_path="${checkpoint_dir}/${logname}/pred_net_20000.pth"
# flagfile="${checkpoint_dir}/${logname}/config.txt"
# vis_path="${checkpoint_dir}/${logname}/TestMixRotMiDaS/"

# CUDA_VISIBLE_DEVICES=$device python predict_vitto.py --flagfile $flagfile --local_rank -1 \
#     --test --ngpu 1 --model_path $model_path --name $logname --checkpoint_dir $checkpoint_dir --vis_path $vis_path \
#     --batch_size $batch_size --repeat 1 --num_workers 8 --dframe_eval 8 --use_depth --eval --eval_nocs \
#     --test_dataset_path ~/vittoria/kaifeng/self-corr-pose/data/wild6d/test_mixRot_DEPTH/bottle/ \
#     --test_list config/bottle_wild6d/test_list_all.txt \
#     --vis_pred --visualize_gt --visualize_mesh --visualize_tex --visualize_conf --visualize_bbox --visualize_match 

# exp 3 rot enc pre, aug
# logname='exp3-RotEncPreAUG'
# model_path="${checkpoint_dir}/${logname}/pred_net_20000.pth"
# flagfile="${checkpoint_dir}/${logname}/config.txt"
# vis_path="${checkpoint_dir}/${logname}/TestMixRot/"

# CUDA_VISIBLE_DEVICES=$device python predict_vitto.py --flagfile $flagfile --local_rank -1 \
#     --test --ngpu 1 --model_path $model_path --name $logname --checkpoint_dir $checkpoint_dir --vis_path $vis_path \
#     --batch_size $batch_size --repeat 1 --num_workers 8 --dframe_eval 8 --use_depth --eval --eval_nocs \
#     --test_dataset_path ~/vittoria/kaifeng/self-corr-pose/data/wild6d/test_mixRot/bottle/ \
#     --test_list config/bottle_wild6d/test_list_all.txt --rot_enc\
#     --vis_pred --visualize_gt --visualize_mesh --visualize_tex --visualize_conf --visualize_bbox --visualize_match 

# exp 4 rot enc post, aug
# logname='exp4-RotEncPostAUG'
# model_path="${checkpoint_dir}/${logname}/pred_net_20000.pth"
# flagfile="${checkpoint_dir}/${logname}/config.txt"
# vis_path="${checkpoint_dir}/${logname}/TestMixRot/"

# CUDA_VISIBLE_DEVICES=$device python predict_vitto.py --flagfile $flagfile --local_rank -1 \
#     --test --ngpu 1 --model_path $model_path --name $logname --checkpoint_dir $checkpoint_dir --vis_path $vis_path \
#     --batch_size $batch_size --repeat 1 --num_workers 8 --dframe_eval 8 --use_depth --eval --eval_nocs \
#     --test_dataset_path ~/vittoria/kaifeng/self-corr-pose/data/wild6d/test_mixRot/bottle/ \
#     --test_list config/bottle_wild6d/test_list_all.txt --rot_enc_post\
#     --vis_pred --visualize_gt --visualize_mesh --visualize_tex --visualize_conf --visualize_bbox --visualize_match 

# exp5 kaifeng 90
# logname='exp5-kaifeng90'
# model_path="${checkpoint_dir}/${logname}/pred_net_20000.pth"
# flagfile="${checkpoint_dir}/${logname}/config.txt"
# vis_path="${checkpoint_dir}/${logname}/TestMixRot/"

# CUDA_VISIBLE_DEVICES=$device python predict_vitto.py --flagfile $flagfile --local_rank -1 \
#     --test --ngpu 1 --model_path $model_path --name $logname --checkpoint_dir $checkpoint_dir --vis_path $vis_path \
#     --batch_size $batch_size --repeat 1 --num_workers 8 --dframe_eval 8 --use_depth --eval --eval_nocs \
#     --test_dataset_path ~/vittoria/kaifeng/self-corr-pose/data/wild6d/test_mixRot/bottle/ \
#     --test_list config/bottle_wild6d/test_list_all.txt \
#     --vis_pred --visualize_gt --visualize_mesh --visualize_tex --visualize_conf --visualize_bbox --visualize_match 

# exp5 kaifeng 90 2
# logname='exp5-kaifeng90'
# model_path="${checkpoint_dir}/${logname}/pred_net_20000.pth"
# flagfile="${checkpoint_dir}/${logname}/config.txt"
# vis_path="${checkpoint_dir}/${logname}/Test90/"

# CUDA_VISIBLE_DEVICES=$device python predict_vitto.py --flagfile $flagfile --local_rank -1 \
#     --test --ngpu 1 --model_path $model_path --name $logname --checkpoint_dir $checkpoint_dir --vis_path $vis_path \
#     --batch_size $batch_size --repeat 1 --num_workers 8 --dframe_eval 8 --use_depth --eval --eval_nocs \
#     --test_dataset_path ~/vittoria/kaifeng/self-corr-pose/data/wild6d/test_set90/bottle/ \
#     --test_list config/bottle_wild6d/test_list_all.txt \
#     --vis_pred --visualize_gt --visualize_mesh --visualize_tex --visualize_conf --visualize_bbox --visualize_match 

## test with midas
# logname='exp6-kaifengAUG_MiDaS'
# model_path="${checkpoint_dir}/${logname}/pred_net_20000.pth"
# flagfile="${checkpoint_dir}/${logname}/config.txt"
# vis_path="${checkpoint_dir}/${logname}/TestMixRotMiDaS/"

# CUDA_VISIBLE_DEVICES=$device python predict_vitto.py --flagfile $flagfile --local_rank -1 \
#     --test --ngpu 1 --model_path $model_path --name $logname --checkpoint_dir $checkpoint_dir --vis_path $vis_path \
#     --batch_size $batch_size --repeat 1 --num_workers 8 --dframe_eval 8 --use_depth --eval --eval_nocs \
#     --test_dataset_path ~/vittoria/kaifeng/self-corr-pose/data/wild6d/test_mixRot_DEPTH/bottle/ \
#     --test_list config/bottle_wild6d/test_list_all.txt \
#     --vis_pred --visualize_gt --visualize_mesh --visualize_tex --visualize_conf --visualize_bbox --visualize_match 

## original Kaifeng model trained on augmented dataset --> using directly last 512-D feature vector of resnet and not transforming it in the ushape: Ushape=img_code
# logname='exp7-kaifengAUG_512code'
# model_path="${checkpoint_dir}/${logname}/pred_net_20000.pth"
# flagfile="${checkpoint_dir}/${logname}/config.txt"
# vis_path="${checkpoint_dir}/${logname}/TestMixRot/"

# CUDA_VISIBLE_DEVICES=$device python predict_vitto.py --flagfile $flagfile --local_rank -1 \
#     --test --ngpu 1 --model_path $model_path --name $logname --checkpoint_dir $checkpoint_dir --vis_path $vis_path \
#     --batch_size $batch_size --repeat 1 --num_workers 8 --dframe_eval 8 --use_depth --eval --eval_nocs \
#     --test_dataset_path ~/vittoria/kaifeng/self-corr-pose/data/wild6d/test_mixRot/bottle/ \
#     --test_list config/bottle_wild6d/test_list_all.txt \
#     --vis_pred --visualize_gt --visualize_mesh --visualize_tex --visualize_conf --visualize_bbox --visualize_match \
#     --codedim 512

## no ushape at all
# logname='exp8-kaifengAUG_noUshape'
# model_path="${checkpoint_dir}/${logname}/pred_net_20000.pth"
# flagfile="${checkpoint_dir}/${logname}/config.txt"
# vis_path="${checkpoint_dir}/${logname}/TestMixRot/"

# CUDA_VISIBLE_DEVICES=$device python predict_vitto.py --flagfile $flagfile --local_rank -1 \
#     --test --ngpu 1 --model_path $model_path --name $logname --checkpoint_dir $checkpoint_dir --vis_path $vis_path \
#     --batch_size $batch_size --repeat 1 --num_workers 8 --dframe_eval 8 --use_depth --eval --eval_nocs \
#     --test_dataset_path ~/vittoria/kaifeng/self-corr-pose/data/wild6d/test_mixRot/bottle/ \
#     --test_list config/bottle_wild6d/test_list_all.txt \
#     --vis_pred --visualize_gt --visualize_mesh --visualize_tex --visualize_conf --visualize_bbox --visualize_match \
#     --no_deform