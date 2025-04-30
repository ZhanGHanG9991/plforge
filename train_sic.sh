set -e

# Train schema filter using simple
python -u train_schema_item_filter.py \
    --batch_size 4 \
    --gradient_descent_step 8 \
    --device 0 \
    --learning_rate 1e-5 \
    --gamma 2.0 \
    --alpha 0.75 \
    --epochs 64 \
    --patience 8 \
    --seed 42 \
    --save_path ./sic_ckpts/sic_simple \
    --tensorboard_save_path ./train_logs/sic_simple \
    --train_filepath ./data/sft_data_simple_train_1200.json \
    --dev_filepath ./data/sft_data_simple_dev_300.json \
    --model_name_or_path /home/zhanghang/opt/models/facebookai_roberta-large \
    --mode train

# Train schema filter using hard
python -u train_schema_item_filter.py \
    --batch_size 4 \
    --gradient_descent_step 8 \
    --device 0 \
    --learning_rate 1e-5 \
    --gamma 2.0 \
    --alpha 0.75 \
    --epochs 64 \
    --patience 8 \
    --seed 42 \
    --save_path ./sic_ckpts/sic_hard \
    --tensorboard_save_path ./train_logs/sic_hard \
    --train_filepath ./data/sft_data_hard_train_1200.json \
    --dev_filepath ./data/sft_data_hard_dev_300.json \
    --model_name_or_path /home/zhanghang/opt/models/facebookai_roberta-large \
    --mode train