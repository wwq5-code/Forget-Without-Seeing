CUDA_VISIBLE_DEVICES=0 \
python unlearning_methods/model_tailor.py \
    --model_id model/base_model/llava-onevision-qwen2-7b-si-hf \
    --original_dir model/original_model/llava-onevision-qwen2-7b-original \
    --data_split_dir data/S-MLLMUn-data/unlearning_data \
    --forget_split_ratio 5 \
    --save_dir model/unlearned_model/llava-onevision/split_ratio_5/ModelTailor \
    --batch_size 1 \
    --nsamples 32 \
    --sparsity_level 0.9 \
    --start_layer 0
