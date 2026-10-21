CUDA_VISIBLE_DEVICES=0,1 \
python unlearning_methods/GA.py \
	--model_id model/base_model/llava-onevision-qwen2-7b-si-hf \
	--original_dir model/original_model/llava-onevision-qwen2-7b-original \
	--data_split_dir data/S-MLLMUn-data/unlearning_data \
	--forget_split_ratio 5 \
	--save_dir model/unlearned_model/llava-onevision/split_ratio_5/GA \
	--batch_size 2 \
	--lr 2e-5 \
	--num_epochs 1