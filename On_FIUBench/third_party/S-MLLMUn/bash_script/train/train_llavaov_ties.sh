CUDA_VISIBLE_DEVICES=0,1,3,4,5 \
python unlearning_methods/ties_train.py \
    --model_id model/base_model/llava-onevision-qwen2-7b-si-hf \
    --original_dir model/original_model/llava-onevision-qwen2-7b-original \
    --data_split_dir data/S-MLLMUn-data/unlearning_data \
    --forget_split_ratio 5 \
    --save_dir model/unlearned_model/llava-onevision/split_ratio_5/ties_train \
    --batch_size 2 \
    --lr 1e-4 \
    --num_epochs 3

CUDA_VISIBLE_DEVICES=0 \
python unlearning_methods/ties_merging.py \
    --model_id model/base_model/llava-onevision-qwen2-7b-si-hf \
    --original_dir model/original_model/llava-onevision-qwen2-7b-original \
    --save_dir model/unlearned_model/llava-onevision/split_ratio_5/ties_merged0.2 \
    --MFA_multi model/unlearned_model/llava-onevision/split_ratio_5/ties_train/epoch_3/MFA_multi \
    --MFA_text model/unlearned_model/llava-onevision/split_ratio_5/ties_train/epoch_3/MFA_text \
    --RA model/unlearned_model/llava-onevision/split_ratio_5/ties_train/epoch_3/RA \
    --density 0.2
