# Revision for PLForge

## Data

The dataset is stored in JSON format, where each entry represents a text-to-PL/SQL task. Below is the structure and description of each key in the JSON data:

### Core Fields

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | A comprehensive, detailed version of the task description (`detailed_text` as the input of model). |
| `plsql` | `str` | The target PL/SQL stored procedure code that accomplishes the described task. |
| `call` | `list[str]` | Procedure call statements demonstrating how to invoke the stored procedure. |
| `database` | `str` | The name of the database schema being used. |
| `table` | `list[str]` | List of table names referenced in the task. |
| `schema` | `dict` | Complete database schema information including table structures. |
| `skeleton` | `str` | Skeleton of the target PL/SQL procedure with placeholders. |

### Task Characteristics

| Field | Type | Description |
|-------|------|-------------|
| `table_num` | `int` | Number of tables involved in the task. |
| `parameter` | `int` | Number of parameters the procedure accepts. |
| `insert` | `int` | Binary indicator (0/1) for INSERT operations. |
| `select` | `int` | Binary indicator (0/1) for SELECT operations. |
| `update` | `int` | Binary indicator (0/1) for UPDATE operations. |
| `delete` | `int` | Binary indicator (0/1) for DELETE operations. |
| `if` | `int` | Binary indicator (0/1) for conditional (IF) statements. |
| `loop` | `int` | Binary indicator (0/1) for loop constructs. |
| `declare` | `int` | Binary indicator (0/1) for variable declarations. |

### Text Variations

| Field | Type | Description |
|-------|------|-------------|
| `concise_text` | `str` | A shortened, simplified version of the task description. |
| `detailed_text` | `str` | A comprehensive, detailed version of the task description. |

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
```

### Step 3: Create Database Environment

Install and configure PostgreSQL:

```bash
apt install postgresql postgresql-contrib
pip install psycopg[binary]
service postgresql start
service postgresql status

pip install oracledb==3.0.0
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

### PostgreSQL Zero-Shot Text-to-PL/SQL

To execute the script `postgres_text2plsql_zero_shot.py`, use the following command format:

```bash
python postgres_text2plsql_zero_shot.py \
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
  --skeleton <0_or_1> \
  --test_plsql_skeletons_path <path_to_test_plsql_skeletons>
```

#### Parameter Descriptions

| Parameter                     | Type   | Description                                                                 |
|-------------------------------|--------|-----------------------------------------------------------------------------|
| `--llm_path`                  | `str`  | Path to the pretrained language model directory.                           |
| `--sic_path`                  | `str`  | Path to the semantic information component model.                          |
| `--table_num`                 | `int`  | Number of tables to include (default: `6`).                                |
| `--column_num`                | `int`  | Number of columns per table (default: `10`).                               |
| `--dataset_path`              | `str`  | Path to the input dataset (in JSON format).                                |
| `--max_tokens`                | `int`  | Maximum number of input tokens for the LLM (default: `4096`).              |
| `--max_new_tokens`            | `int`  | Maximum number of tokens to generate (default: `256`).                     |
| `--skeleton_predictor_path`   | `str`  | Path to the skeleton predictor checkpoint directory.                       |
| `--is_filter_schema`          | `int`  | Whether to enable schema filtering (`1` for true, `0` for false).          |
| `--cot`                       | `int`  | Whether to use chain-of-thought prompting (`1` for true, `0` for false).   |
| `--skeleton`                  | `int`  | Whether to include skeleton guidance (`1` for true, `0` for false).        |
| `--test_plsql_skeletons_path` | `str`  | Path to the test PLSQL skeletons file.                                     |

### PostgreSQL Few-Shot Text-to-PL/SQL

To execute the script `postgres_text2plsql_few_shot.py`, use the following command format:

```bash
python postgres_text2plsql_few_shot.py \
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
  --similarity_w <similarity_weight> \
  --test_question_skeletons_path <path_to_test_question_skeletons> \
  --test_plsql_skeletons_path <path_to_test_plsql_skeletons> \
  --plsql_skeleton_similarity_path <path_to_plsql_skeleton_similarity>
```

#### Parameter Descriptions

| Parameter                          | Type    | Description                                                                 |
|------------------------------------|---------|-----------------------------------------------------------------------------|
| `--llm_path`                       | `str`   | Path to the pretrained language model directory.                           |
| `--sic_path`                       | `str`   | Path to the semantic information component model.                          |
| `--table_num`                      | `int`   | Number of tables to include in the schema (default: `5`).                  |
| `--column_num`                     | `int`   | Number of columns per table (default: `6`).                                |
| `--dataset_path`                   | `str`   | Path to the evaluation or generation dataset.                              |
| `--demonstration_set_path`         | `str`   | Path to the demonstration (few-shot) example set.                          |
| `--num_of_demonstrations`          | `int`   | Number of demonstrations to include in the prompt.                         |
| `--max_tokens`                     | `int`   | Maximum number of input tokens to the model.                               |
| `--max_new_tokens`                 | `int`   | Maximum number of tokens the model is allowed to generate.                 |
| `--skeleton_predictor_path`        | `str`   | Path to the skeleton predictor checkpoint directory.                       |
| `--is_filter_schema`               | `int`   | Whether to enable schema filtering (`1` = yes, `0` = no).                  |
| `--cot`                            | `int`   | Whether to use chain-of-thought prompting (`1` = yes, `0` = no).           |
| `--skeleton`                       | `int`   | Whether to use skeleton guidance (`1` = yes, `0` = no).                    |
| `--new_similarity`                 | `int`   | Whether to use the new similarity calculation (`1` = yes, `0` = no).       |
| `--similarity_w`                   | `float` | Weight for similarity scoring in demonstration selection.                  |
| `--test_question_skeletons_path`   | `str`   | Path to the test question skeletons cache file.                           |
| `--test_plsql_skeletons_path`      | `str`   | Path to the test PLSQL skeletons cache file.                              |
| `--plsql_skeleton_similarity_path` | `str`   | Path to the PLSQL skeleton similarity cache file.                         |

### Oracle Zero-Shot Text-to-PL/SQL

To execute the script `oracle_text2plsql_zero_shot.py`, use the following command format:

```bash
python oracle_text2plsql_zero_shot.py \
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
  --skeleton <0_or_1> \
  --test_plsql_skeletons_path <path_to_test_plsql_skeletons> \
  --host <oracle_host> \
  --port <oracle_port>
```

#### Parameter Descriptions

| Parameter                     | Type   | Description                                                                 |
|-------------------------------|--------|-----------------------------------------------------------------------------|
| `--llm_path`                  | `str`  | Path to the pretrained language model directory.                           |
| `--sic_path`                  | `str`  | Path to the semantic information component model.                          |
| `--table_num`                 | `int`  | Number of tables to include (default: `6`).                                |
| `--column_num`                | `int`  | Number of columns per table (default: `10`).                               |
| `--dataset_path`              | `str`  | Path to the input dataset (in JSON format).                                |
| `--max_tokens`                | `int`  | Maximum number of input tokens for the LLM (default: `4096`).              |
| `--max_new_tokens`            | `int`  | Maximum number of tokens to generate (default: `256`).                     |
| `--skeleton_predictor_path`   | `str`  | Path to the skeleton predictor checkpoint directory.                       |
| `--is_filter_schema`          | `int`  | Whether to enable schema filtering (`1` for true, `0` for false).          |
| `--cot`                       | `int`  | Whether to use chain-of-thought prompting (`1` for true, `0` for false).   |
| `--skeleton`                  | `int`  | Whether to include skeleton guidance (`1` for true, `0` for false).        |
| `--test_plsql_skeletons_path` | `str`  | Path to the test PLSQL skeletons file.                                     |
| `--host`                      | `str`  | Oracle database host address (default: `localhost`).                      |
| `--port`                      | `int`  | Oracle database port number (default: `1521`).                            |

### Oracle Few-Shot Text-to-PL/SQL

To execute the script `oracle_text2plsql_few_shot.py`, use the following command format:

```bash
python oracle_text2plsql_few_shot.py \
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
  --similarity_w <similarity_weight> \
  --test_question_skeletons_path <path_to_test_question_skeletons> \
  --test_plsql_skeletons_path <path_to_test_plsql_skeletons> \
  --plsql_skeleton_similarity_path <path_to_plsql_skeleton_similarity> \
  --host <oracle_host> \
  --port <oracle_port>
```

#### Parameter Descriptions

| Parameter                          | Type    | Description                                                                 |
|------------------------------------|---------|-----------------------------------------------------------------------------|
| `--llm_path`                       | `str`   | Path to the pretrained language model directory.                           |
| `--sic_path`                       | `str`   | Path to the semantic information component model.                          |
| `--table_num`                      | `int`   | Number of tables to include in the schema (default: `5`).                  |
| `--column_num`                     | `int`   | Number of columns per table (default: `6`).                                |
| `--dataset_path`                   | `str`   | Path to the evaluation or generation dataset.                              |
| `--demonstration_set_path`         | `str`   | Path to the demonstration (few-shot) example set.                          |
| `--num_of_demonstrations`          | `int`   | Number of demonstrations to include in the prompt.                         |
| `--max_tokens`                     | `int`   | Maximum number of input tokens to the model.                               |
| `--max_new_tokens`                 | `int`   | Maximum number of tokens the model is allowed to generate.                 |
| `--skeleton_predictor_path`        | `str`   | Path to the skeleton predictor checkpoint directory.                       |
| `--is_filter_schema`               | `int`   | Whether to enable schema filtering (`1` = yes, `0` = no).                  |
| `--cot`                            | `int`   | Whether to use chain-of-thought prompting (`1` = yes, `0` = no).           |
| `--skeleton`                       | `int`   | Whether to use skeleton guidance (`1` = yes, `0` = no).                    |
| `--new_similarity`                 | `int`   | Whether to use the new similarity calculation (`1` = yes, `0` = no).       |
| `--similarity_w`                   | `float` | Weight for similarity scoring in demonstration selection.                  |
| `--test_question_skeletons_path`   | `str`   | Path to the test question skeletons cache file.                           |
| `--test_plsql_skeletons_path`      | `str`   | Path to the test PLSQL skeletons cache file.                              |
| `--plsql_skeleton_similarity_path` | `str`   | Path to the PLSQL skeleton similarity cache file.                         |
| `--host`                           | `str`   | Oracle database host address (default: `localhost`).                      |
| `--port`                           | `int`   | Oracle database port number (default: `1521`).                            |
