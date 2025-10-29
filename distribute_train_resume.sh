cd /18018998051/SAMWISE
/root/anaconda3/envs/mamba/bin/python -m torch.distributed.launch --nproc_per_node 2 --use_env main.py --eval --config models/config/SVR_inter_replace.yaml --dataset_file refavs --name_exp SVR_inter_replace --epochs 20 --batch_size 4 --output_dir save --resume /18018998051/SAMWISE/save/SVR_inter_replace/checkpoint_latest.pth --resume_optimizer  --lr 1e-4 --lr_drop 21168
