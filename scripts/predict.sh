device=0
batch_size=16

checkpoint_dir='log'
logname='exp1-bottle'

model_path="${checkpoint_dir}/${logname}/pred_net_20000.pth"
flagfile="${checkpoint_dir}/${logname}/config.txt"
vis_path="${checkpoint_dir}/${logname}/visualization/"

CUDA_VISIBLE_DEVICES=$device python predict.py --flagfile $flagfile --local_rank -1 \
    --test --ngpu 1 --model_path $model_path --name $logname --checkpoint_dir $checkpoint_dir --vis_path $vis_path \
    --batch_size $batch_size --repeat 1 --num_workers 8 --dframe_eval 8 --use_depth --eval --eval_nocs \
    --test_dataset_path ~/vittoria/kaifeng/self-corr-pose/data/wild6d/test_set/bottle/ \
    --vis_pred --visualize_gt --visualize_mesh --visualize_tex --visualize_conf --visualize_bbox --visualize_match


