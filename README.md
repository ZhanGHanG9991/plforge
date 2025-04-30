# PLForge: Enhancing Language Models for Natural Language to Procedural Extensions of SQL

## Environment Requirement

### Step 1: Install Java

```bash
apt-get update
apt-get install -y openjdk-11-jdk
```

### Step 2: Create Python Environment

Create a new Anaconda environment and install the required modules:

```bash
conda create -n codes python=3.8.5
conda activate codes
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
git clone https://github.com/lihaoyang-ruc/SimCSE.git
cd SimCSE
python setup.py install
cd ..
```

### Step 3: Create Database Environment

Install and configure PostgreSQL:

```bash
apt install postgresql postgresql-contrib
pip install psycopg[binary] -i https://pypi.tuna.tsinghua.edu.cn/simple
service postgresql start
service postgresql status
```

Set the password for the `postgres` user:

```bash
sudo -i -u postgres
psql
\l
ALTER USER postgres PASSWORD '123456';
\q
exit
```


## Run Experiments

To execute the script `text2plsql_zero_shot.py`, use the following command format:

```bash
python text2plsql_zero_shot.py \
  --llm_path <path_to_language_model> \
  --sic_path <path_to_sic_model> \
  --table_num <number_of_tables> \
  --column_num <number_of_columns> \
  --dataset_path <path_to_dataset> \
  --max_tokens <maximum_input_tokens> \
  --max_new_tokens <maximum_output_tokens> \
  --skeleton_predictor_path <path_to_skeleton_predictor> \
  --is_filter_schema <0_or_1> \
  --cot <0_or_1> \
  --skeleton <0_or_1>
```

### Parameter Descriptions

| Parameter                | Type   | Description                                                                 |
|--------------------------|--------|-----------------------------------------------------------------------------|
| `--llm_path`             | `str`  | Path to the pretrained language model directory.                            |
| `--sic_path`             | `str`  | Path to the semantic information component model.                           |
| `--table_num`            | `int`  | Number of tables to include (default: `6`).                                 |
| `--column_num`           | `int`  | Number of columns per table (default: `10`).                                |
| `--dataset_path`         | `str`  | Path to the input dataset (in JSON or other supported format).              |
| `--max_tokens`           | `int`  | Maximum number of input tokens for the LLM (default: `4096`).               |
| `--max_new_tokens`       | `int`  | Maximum number of tokens to generate (default: `256`).                      |
| `--skeleton_predictor_path` | `str` | Path to the skeleton predictor checkpoint directory.                        |
| `--is_filter_schema`     | `int`  | Whether to enable schema filtering (`1` for true, `0` for false).           |
| `--cot`                  | `int`  | Whether to use chain-of-thought prompting (`1` for true, `0` for false).    |
| `--skeleton`             | `int`  | Whether to include skeleton guidance (`1` for true, `0` for false).         |



To execute the script `text2plsql_few_shot.py`, use the following command format:

```bash
python text2plsql_few_shot.py \
  --llm_path <path_to_language_model> \
  --sic_path <path_to_sic_model> \
  --table_num <number_of_tables> \
  --column_num <number_of_columns> \
  --dataset_path <path_to_dataset> \
  --demonstration_set_path <path_to_demo_set> \
  --num_of_demonstrations <number_of_demos> \
  --max_tokens <maximum_input_tokens> \
  --max_new_tokens <maximum_output_tokens> \
  --skeleton_predictor_path <path_to_skeleton_predictor> \
  --is_filter_schema <0_or_1> \
  --cot <0_or_1> \
  --skeleton <0_or_1> \
  --new_similarity <0_or_1> \
  --similarity_w <similarity_weight>
```

### Parameter Descriptions

| Parameter                    | Type    | Description                                                                 |
|------------------------------|---------|-----------------------------------------------------------------------------|
| `--llm_path`                 | `str`   | Path to the pretrained language model directory.                            |
| `--sic_path`                 | `str`   | Path to the semantic information component model.                           |
| `--table_num`                | `int`   | Number of tables to include in the schema (default: `5`).                   |
| `--column_num`               | `int`   | Number of columns per table (default: `6`).                                 |
| `--dataset_path`            | `str`   | Path to the evaluation or generation dataset.                               |
| `--demonstration_set_path`  | `str`   | Path to the demonstration (few-shot) example set.                           |
| `--num_of_demonstrations`   | `int`   | Number of demonstrations to include in the prompt.                         |
| `--max_tokens`              | `int`   | Maximum number of input tokens to the model.                                |
| `--max_new_tokens`          | `int`   | Maximum number of tokens the model is allowed to generate.                 |
| `--skeleton_predictor_path` | `str`   | Path to the skeleton predictor checkpoint directory.                        |
| `--is_filter_schema`        | `int`   | Whether to enable schema filtering (`1` = yes, `0` = no).                   |
| `--cot`                     | `int`   | Whether to use chain-of-thought prompting (`1` = yes, `0` = no).            |
| `--skeleton`                | `int`   | Whether to use skeleton guidance (`1` = yes, `0` = no).                     |
| `--new_similarity`          | `int`   | Whether to use the new similarity calculation (`1` = yes, `0` = no).        |
| `--similarity_w`            | `float` | Weight for similarity scoring in demonstration selection.                   |
