CUDA_VISIBLE_DEVICES=0,1  python eval.py \
 --model_id model/base_model/llava-onevision-qwen2-7b-si-hf \
 --cache_path model/unlearned_model/llava-onevision/split_ratio_5/GA_Difference \
 --data_split_folder data/S-MLLMUn-data/eval_data \
 --output_file GA_Difference \
 --output_folder eval_result/llava-onevision/split_ratio_5/GA_Difference \
 --forget_ratio 5
