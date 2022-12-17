GPUS=$1
MODEL=$2
PORT=${PORT:-29500}
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
	    main_pretrain.py --batch_size 64 --epochs 800 --accum_iter 8 --model $MODEL --norm_pix_loss --blr 1.5e-4 --warmup_epochs 40 --mask_ratio 0.75 --data_path /raid/imagenet --output_dir ./output/$MODEL
