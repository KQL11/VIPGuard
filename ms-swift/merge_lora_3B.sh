swift export \
    --model /data0/linkaiqing/code/MLLM/VIP_DFD_LLM_V4/Model_Stage1_Checkpoints/qwen_2_5_vl_3B_FaceAttr \
    --template 'kq_qwen2_5_vl' \
    --adapters '/data0/linkaiqing/code/MLLM/VIP_DFD_LLM_V4/checkpoints_attr_Stage2.1_mixed_score_Qwen2_5_VL_3B_new_answer/v0-20250505-043717/checkpoint-265' \
    --output_dir '/data0/linkaiqing/code/MLLM/VIP_DFD_LLM_V4/checkpoints_attr_Stage2.1_merge_mixed_score_new_answer_3B' \
    --merge_lora true \
    --custom_register_path 'custom_model.py' \
    --max_memory '{0: "78GB", 1: "78GB", 2: "78GB", 3: "78GB", 4: "78GB", 5: "78GB"}' \
