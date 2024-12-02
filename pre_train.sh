set -e

# Pre-train 3B
accelerate launch train_causal_lm.py --per_device_train_batch_size 4 --block_size 8192 --seed 42 --pretrained_model_name_or_path /home/zhanghang/opt/models/bigcode_starcoder2-3b --epochs 1 --lr 5e-5 --warmup_ratio 0.0 --checkpointing_steps 5000 --tensorboard_log_dir ./train_logs/pt-starcoder2-text2plsql-3b --mode pt --output_ckpt_dir ./ckpts/pt-starcoder2-text2plsql-3b --pt_data_dir /home/zhanghang/opt/projects/researchprojects/text2PLSQL/datasets/tokenized_text2plsql_corpus_starcoder2_15B.bin

# Pre-train 7B
accelerate launch train_causal_lm.py --per_device_train_batch_size 4 --block_size 8192 --seed 42 --pretrained_model_name_or_path /home/zhanghang/opt/models/bigcode_starcoder2-7b --epochs 1 --lr 5e-5 --warmup_ratio 0.0 --checkpointing_steps 5000 --tensorboard_log_dir ./train_logs/pt-starcoder2-text2plsql-7b --mode pt --output_ckpt_dir ./ckpts/pt-starcoder2-text2plsql-7b --pt_data_dir /home/zhanghang/opt/projects/researchprojects/text2PLSQL/datasets/tokenized_text2plsql_corpus_starcoder2_15B.bin

# Pre-train 15B
accelerate launch train_causal_lm.py --per_device_train_batch_size 2 --block_size 8192 --seed 42 --pretrained_model_name_or_path /home/zhanghang/opt/models/bigcode_starcoder2-15b --epochs 1 --lr 5e-5 --warmup_ratio 0.0 --checkpointing_steps 5000 --tensorboard_log_dir ./train_logs/pt-starcoder2-text2plsql-15b --mode pt --output_ckpt_dir ./ckpts/pt-starcoder2-text2plsql-15b --pt_data_dir /home/zhanghang/opt/projects/researchprojects/text2PLSQL/datasets/tokenized_text2plsql_corpus_starcoder2_15B.bin