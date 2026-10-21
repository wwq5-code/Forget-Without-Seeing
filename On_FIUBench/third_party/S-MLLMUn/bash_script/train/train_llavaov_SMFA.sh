CUDA_VISIBLE_DEVICES=0,1,3,4,5 \
python unlearning_methods/SMFA_train.py \
	--model_id model/base_model/llava-onevision-qwen2-7b-si-hf \
	--original_dir model/original_model/llava-onevision-qwen2-7b-original \
	--data_split_dir data/S-MLLMUn-data/unlearning_data \
	--forget_split_ratio 5 \
	--save_dir model/unlearned_model/llava-onevision/split_ratio_5/SMFA \
	--batch_size 2 \
	--lr 1e-4 \
	--num_epochs 5

CUDA_VISIBLE_DEVICES=0,1,3,4,5 \
python unlearning_methods/SMFA_sculpt.py \
	--model_id model/base_model/llava-onevision-qwen2-7b-si-hf \
	--original_dir model/original_model/llava-onevision-qwen2-7b-original \
	--save_dir model/unlearned_model/llava-onevision/split_ratio_5/SMFA/sculpted \
    --MFA_multi model/unlearned_model/llava-onevision/split_ratio_5/SMFA/epoch_3/MFA_multi \
    --MFA_text model/unlearned_model/llava-onevision/split_ratio_5/SMFA/epoch_3/MFA_text \
    --RA model/unlearned_model/llava-onevision/split_ratio_5/SMFA/epoch_3/RA \
    --multi_k 5 \
    --text_k 10