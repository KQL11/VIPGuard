# 4 * 22GiB
# vit/merger lr 1e-5; llm lora lr 1e-4
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MAX_PIXELS=200704 \
swift sft \
    --model YOUR_MODEL \
    --template 'kq_qwen2_5_vl' \
    --dataset YOUR_DATA \
    --torch_dtype 'bfloat16' \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 3e-5 \
    --weight_decay 1e-3 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --gradient_accumulation_steps 72 \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 8192 \
    --max_pixels 200704 \
    --output_dir 'output' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --train_type custom \
    --optimizer custom \
    --external_plugins './ms-swift/custom_plugin.py' \
    --lazy_tokenize true \
    --custom_register_path './ms-swift/custom_model.py' \
    --gradient_checkpointing False \
