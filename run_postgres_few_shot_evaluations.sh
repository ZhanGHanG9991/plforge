set -e

CUDA_VISIBLE_DEVICES=6 python -u postgres_text2plsql_few_shot.py \
    --llm_path /home/zhanghang/opt/models/bigcode_starcoder2-3b \
    --dataset_path /home/zhanghang/opt/projects/researchprojects/text2PLSQL/data/postgres_data_procbench_test_300.json \
    --demonstration_set_path /home/zhanghang/opt/projects/researchprojects/text2PLSQL/data/postgres_data_procbench_train_1500.json \
    --num_of_demonstrations 3 \
    --max_tokens 8192 \
    --max_new_tokens 256 \
    --sic_path /home/zhanghang/opt/projects/researchprojects/text2PLSQL/sic/sic_ckpts/pre/postgres_sic_procbench \
    --table_num 6 \
    --column_num 10 \
    --skeleton_predictor_path /home/zhanghang/opt/projects/researchprojects/text2PLSQL/skeleton/models/postgres_skeleton_predictor_procbench \
    --is_filter_schema 1 \
    --cot 1 \
    --skeleton 1 \
    --new_similarity 1 \
    --similarity_w 0.5 \
    --test_question_skeletons_path /home/zhanghang/opt/projects/researchprojects/text2PLSQL/codes/cache/postgres_question_skeletons_procbench_test_300.cache \
    --test_plsql_skeletons_path /home/zhanghang/opt/projects/researchprojects/text2PLSQL/codes/cache/postgres_plsql_skeletons_procbench_test_300.cache \
    --plsql_skeleton_similarity_path /home/zhanghang/opt/projects/researchprojects/text2PLSQL/codes/cache/postgres_skeleton_similarity_procbench_test_300.cache