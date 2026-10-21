CUDA_VISIBLE_DEVICES=0,1  python eval.py \
 --model_id model/base_model/llava-onevision-qwen2-7b-si-hf \
 --cache_path model/unlearned_model/llava-onevision/split_ratio_5/SMFA/sculpted \
 --data_split_folder data/S-MLLMUn-data/eval_data \
 --output_file SMFA \
 --output_folder eval_result/llava-onevision/split_ratio_5/SMFA \
 --forget_ratio 5