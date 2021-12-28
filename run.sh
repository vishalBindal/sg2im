if [[ $PYTHONPATH =~ $PWD ]]
then
    echo ""
else
    export PYTHONPATH=$PWD:$PYTHONPATH
fi

python3 scripts/train.py \
--robot_train_image_dir datasets/train_images \
--robot_val_image_dir datasets/val_images \
--robot_train_instances_json datasets/instances_train.json \
--robot_val_instances_json datasets/instances_val.json \
--image_size '96,128' \
--num_val_samples 454 \
--loader_num_workers 0 \
--mask_size 32 \
--batch_size 8 \
--num_iterations 20000 \
--eval_mode_after 10000 \
--checkpoint_every 1000 \
--embedding_dim 256
