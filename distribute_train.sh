cd /18018998051/SAMWISE
/root/anaconda3/envs/mamba/bin/python -m torch.distributed.launch --nproc_per_node 2 --use_env main.py --config models/config/SVR_base.yaml --dataset_file refavs --name_exp SVR_base --epochs 10 --batch_size 4 --output_dir save
