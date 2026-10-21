CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
python unlearning_methods/MANU_incremental.py \
	--model_save_name MANU \
	--model_id model/base_model/llava-onevision-qwen2-7b-si-hf \
	--original_dir model/original_model/llava-onevision-qwen2-7b-original \
	--data_split_dir data/S-MLLMUn-data/unlearning_data \
    --batch_size 1  \
	--max_length 384 \
	--forget_split_ratio 5 \
	--prune_percent 10 \
	--save_path model/unlearned_model/llava-onevision/split_ratio_5