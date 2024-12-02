set -e

# simple dataset

# pt-starcoder2-text2plsql-3b

CUDA_VISIBLE_DEVICES=5 python -u text2plsql_zero_shot.py --llm_path /home/zhanghang/opt/projects/researchprojects/text2PLSQL/models/pt-starcoder2-text2plsql-3b --dataset_path /home/zhanghang/opt/projects/researchprojects/text2PLSQL/experiments/data/data_simple_test_300.json --max_tokens 8192 --max_new_tokens 256 --sic_path /home/zhanghang/opt/projects/gitprojects/codes/sic_ckpts/sic_spider --table_num 6 --column_num 10 --is_filter_schema 1 --cot 1 --skeleton 1

# pt-starcoder2-text2plsql-7b

CUDA_VISIBLE_DEVICES=5 python -u text2plsql_zero_shot.py --llm_path /home/zhanghang/opt/projects/researchprojects/text2PLSQL/models/pt-starcoder2-text2plsql-7b --dataset_path /home/zhanghang/opt/projects/researchprojects/text2PLSQL/experiments/data/data_simple_test_300.json --max_tokens 8192 --max_new_tokens 256 --sic_path /home/zhanghang/opt/projects/gitprojects/codes/sic_ckpts/sic_spider --table_num 6 --column_num 10 --is_filter_schema 1 --cot 1 --skeleton 1

# pt-starcoder2-text2plsql-15b

CUDA_VISIBLE_DEVICES=5 python -u text2plsql_zero_shot.py --llm_path /home/zhanghang/opt/projects/researchprojects/text2PLSQL/models/pt-starcoder2-text2plsql-15b --dataset_path /home/zhanghang/opt/projects/researchprojects/text2PLSQL/experiments/data/data_simple_test_300.json --max_tokens 8192 --max_new_tokens 256 --sic_path /home/zhanghang/opt/projects/gitprojects/codes/sic_ckpts/sic_spider --table_num 6 --column_num 10 --is_filter_schema 1 --cot 1 --skeleton 1


# hard dataset

# pt-starcoder2-text2plsql-3b

CUDA_VISIBLE_DEVICES=5 python -u text2plsql_zero_shot.py --llm_path /home/zhanghang/opt/projects/researchprojects/text2PLSQL/models/pt-starcoder2-text2plsql-3b --dataset_path /home/zhanghang/opt/projects/researchprojects/text2PLSQL/experiments/data/data_hard_test_300.json --max_tokens 8192 --max_new_tokens 256 --sic_path /home/zhanghang/opt/projects/gitprojects/codes/sic_ckpts/sic_spider --table_num 6 --column_num 10 --is_filter_schema 1 --cot 1 --skeleton 1

# pt-starcoder2-text2plsql-7b

CUDA_VISIBLE_DEVICES=5 python -u text2plsql_zero_shot.py --llm_path /home/zhanghang/opt/projects/researchprojects/text2PLSQL/models/pt-starcoder2-text2plsql-7b --dataset_path /home/zhanghang/opt/projects/researchprojects/text2PLSQL/experiments/data/data_hard_test_300.json --max_tokens 8192 --max_new_tokens 256 --sic_path /home/zhanghang/opt/projects/gitprojects/codes/sic_ckpts/sic_spider --table_num 6 --column_num 10 --is_filter_schema 1 --cot 1 --skeleton 1

# pt-starcoder2-text2plsql-15b

CUDA_VISIBLE_DEVICES=5 python -u text2plsql_zero_shot.py --llm_path /home/zhanghang/opt/projects/researchprojects/text2PLSQL/models/pt-starcoder2-text2plsql-15b --dataset_path /home/zhanghang/opt/projects/researchprojects/text2PLSQL/experiments/data/data_hard_test_300.json --max_tokens 8192 --max_new_tokens 256 --sic_path /home/zhanghang/opt/projects/gitprojects/codes/sic_ckpts/sic_spider --table_num 6 --column_num 10 --is_filter_schema 1 --cot 1 --skeleton 1