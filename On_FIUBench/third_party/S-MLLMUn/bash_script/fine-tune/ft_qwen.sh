CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
python finetune.py \
--model_id model/base_model/Qwen2.5-VL-7B-Instruct \
--save_dir model/original_model/Qwen2.5-VL-7B-original \
--data_dir data/S-MLLMUn-data/ft_data/ft_data.parquet \
--batch_size 2 \
--lr 1e-4 \
--num_epochs 20 \
--max_length 384