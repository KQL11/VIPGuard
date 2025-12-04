#!/bin/bash
# 启动你的程序，比如：
# 启动命令，例如：
# 4 * 22GiB
# vit/merger lr 1e-5; llm lora lr 1e-4
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
NPROC_PER_NODE=7 \
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
MAX_PIXELS=200704 \
swift sft \
    --model /data0/linkaiqing/code/MLLM/VIP_DFD_LLM_V3/checkpoints_merged_attr_Stage1.1_choice_region_long/qwen2.5-3B_batch_72_VL_lora_r64_a128_lr1e4 \
    --template 'qwen2_5_vl' \
    --dataset '/data0/linkaiqing/code/MLLM/VIP_DFD_LLM_V3/DATA_NEW_wo_Score/New_Answer_V2_Stage2.1_New_Mixed_Score_subset_ID_Discrimination_wo_face_pad.json' \
    --torch_dtype 'bfloat16' \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
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
    --output_dir '/data0/linkaiqing/code/MLLM/VIP_DFD_LLM_V4/checkpoints_attr_Stage2.1_mixed_score_new_answer_wo_face_pad' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --train_type custom \
    --optimizer custom \
    --external_plugins 'custom_plugin.py' \
    --lazy_tokenize true \
    --custom_register_path 'custom_model.py' \
    --max_memory '{1: "78GB", 2: "78GB", 3: "78GB", 4: "78GB", 5: "78GB", 6: "78GB", 7: "78GB"}' \
    --gradient_checkpointing False \
#    --attn_impl flash_attn \
#    --save_only_model true
#    --deepspeed zero2 \
