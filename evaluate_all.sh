#!/bin/bash
cd /18018998051/SAMWISE
# 设置只使用GPU 0
export CUDA_VISIBLE_DEVICES=0

# 定义任务列表：每个任务包含split和name_exp
tasks=(
    "test_s:SVR_base"
    "test_u:SVR_base"
    "test_n:SVR_base"
)

# 遍历任务列表执行
for task in "${tasks[@]}"; do
    split=$(echo "$task" | cut -d':' -f1)
    name_exp=$(echo "$task" | cut -d':' -f2)

    # 动态生成配置文件和权重路径（基于name_exp）
    config_path="models/config/${name_exp}.yaml"
    resume_path="/18018998051/SAMWISE/save/${name_exp}/checkpoint_latest.pth"

    echo "===== Running task: $split (name_exp: $name_exp) ====="
    echo "  Config: $config_path"
    echo "  Resume: $resume_path"

    /root/anaconda3/envs/mamba/bin/python /18018998051/SAMWISE/inference_refavs.py \
        --resume "$resume_path" \
        --split "$split" \
        --batch_size_val 1 \
        --name_exp "$name_exp" \
        --config "$config_path" \
        --save_pred_masks
done