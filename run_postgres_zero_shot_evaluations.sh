set -e

CUDA_VISIBLE_DEVICES=5 python -u postgres_text2plsql_zero_shot.py \
    --llm_path /home/zhanghang/opt/models/bigcode_starcoder2-3b \
    --dataset_path /home/zhanghang/opt/projects/researchprojects/text2PLSQL/data/postgres_data_procbench_test_300.json \
    --max_tokens 8192 \
    --max_new_tokens 256 \
    --sic_path /home/zhanghang/opt/projects/researchprojects/text2PLSQL/sic/sic_ckpts/pre/postgres_sic_procbench \
    --skeleton_predictor_path /home/zhanghang/opt/projects/researchprojects/text2PLSQL/skeleton/models/postgres_skeleton_predictor_procbench \
    --table_num 6 \
    --column_num 10 \
    --is_filter_schema 1 \
    --cot 1 \
    --skeleton 1 \
    --test_plsql_skeletons_path /home/zhanghang/opt/projects/researchprojects/text2PLSQL/codes/cache/postgres_plsql_skeletons_procbench_test_300.cache