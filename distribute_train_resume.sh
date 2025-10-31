cd /18018998051/SAMWISE
/root/anaconda3/envs/mamba/bin/python -m torch.distributed.launch --nproc_per_node 2 --use_env main.py --eval --config models/config/SVR_full_bs8.yaml --dataset_file refavs --name_exp SVR_full_bs8 --epochs 20 --batch_size 8 --output_dir save --resume /18018998051/SAMWISE/save/SVR_full_bs8/checkpoint_latest.pth --resume_optimizer  --lr 1e-5 --lr_drop 60000
