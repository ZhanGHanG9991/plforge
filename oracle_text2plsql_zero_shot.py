import argparse
import os
import torch
import json
import time
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.load_sft_dataset import SFTPLSQLGenerationDataset
from utils.oracle_db_utils import check_plsql_executability, compare_plsql, is_exact_match

def parse_option():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--llm_path', type = str)
    parser.add_argument('--sic_path', type = str)
    parser.add_argument('--table_num', type = int, default = 6)
    parser.add_argument('--column_num', type = int, default = 10)

    parser.add_argument('--dataset_path', type = str)

    parser.add_argument('--max_tokens', type = int, default = 4096)
    parser.add_argument('--max_new_tokens', type = int, default = 256)

    parser.add_argument('--skeleton_predictor_path', type = str)

    parser.add_argument('--is_filter_schema', type = int)
    parser.add_argument('--cot', type = int)
    parser.add_argument('--skeleton', type = int)

    parser.add_argument('--test_plsql_skeletons_path', type = str)
    
    # 添加Oracle连接参数
    parser.add_argument('--host', type = str, default = 'localhost')
    parser.add_argument('--port', type = int, default = 1521)

    opt = parser.parse_args()

    return opt

def text2plsql_func(model, inputs, tokenizer, max_new_tokens):
    input_length = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            num_beams = 4,
            num_return_sequences = 4
        )

    generated_plsqls = tokenizer.batch_decode(generate_ids[:, input_length:], skip_special_tokens = True, clean_up_tokenization_spaces = False)

    return generated_plsqls

def post_process(schema, plsql):

    def tokenize_schema(schema):
        token_list = []
        for schema_item in schema["schema_items"]:
            token_list.append(schema_item["table_name"])
            token_list.extend(schema_item["column_names"])
        return token_list

    def tokenize_plsql_statement(plsql):
        # 使用正则表达式匹配单词、数字和特殊符号
        token_list = re.findall(r'\w+|[^\w\s]', plsql)
        return token_list

    # end with END;
    index = plsql.find("END;")
    if index != -1:
        plsql = plsql[:index + 4]
    
    # add "" for oracle
    plsql_token_list = tokenize_plsql_statement(plsql)
    schema_token_list = tokenize_schema(schema)

    final_token_list = []

    for token in plsql_token_list:
        if token in schema_token_list:
            token = f"\"{token}\""
        final_token_list.append(token)

    # adapt call sp
    for i in range(1, len(final_token_list) - 1):
        left = final_token_list[i-1].lower()
        right = final_token_list[i+1].lower()
        if (left == "procedure" or left == "function") and right == "(":
            final_token_list[i] = "sp"
    
    # deal with "" in (procedure parameters)
    monitoring = False
    meet_zuo_kuo = False
    for i in range(len(final_token_list)):
        if final_token_list[i] == "sp":
            monitoring = True
        if final_token_list[i] == "(":
            meet_zuo_kuo = True
        if final_token_list[i] == ")":
            monitoring = False
            meet_zuo_kuo = False
        if monitoring and meet_zuo_kuo:
            final_token_list[i] = final_token_list[i].replace("\"", "")

    # avoid "> =" and "< =" and "! =" and ": =" and "| |"
    for i in range(len(final_token_list) - 2):
        if (final_token_list[i] == "=" or final_token_list[i] == "<" or final_token_list[i] == ">") and final_token_list[i+1] == "=":
            final_token_list[i] = final_token_list[i] + "="
            final_token_list[i+1] = ""
        elif final_token_list[i] == "<" and final_token_list[i+1] == ">":
            final_token_list[i] = final_token_list[i] + ">"
            final_token_list[i+1] = ""
        elif final_token_list[i] == "!" and final_token_list[i+1] == "=":
            final_token_list[i] = final_token_list[i] + "="
            final_token_list[i+1] = ""
        elif final_token_list[i] == ":" and final_token_list[i+1] == "=":
            final_token_list[i] = final_token_list[i] + "="
            final_token_list[i+1] = ""
        elif final_token_list[i] == "|" and final_token_list[i+1] == "|":
            final_token_list[i] = final_token_list[i] + "|"
            final_token_list[i+1] = ""
        elif final_token_list[i] == "'" and final_token_list[i+1] == "%" and final_token_list[i+2] == "'":
            final_token_list[i] = final_token_list[i] + "%'"
            final_token_list[i+1] = ""
            final_token_list[i+2] = ""

    # avoid " " Something " "
    post_plsql = re.sub(r'" +"', '"', " ".join(final_token_list))

    # 修正小数格式，例如 "0 . 5" -> "0.5"
    post_plsql = re.sub(r'(\d+)\s*\.\s*(\d+)', r'\1.\2', post_plsql)

    # avoid " Something "
    post_plsql = re.sub(r'"\s*(.*?)\s*"', r'"\1"', post_plsql)
    
    return post_plsql


if __name__ == "__main__":
    opt = parse_option()
    print(opt)
    max_tokens = opt.max_tokens
    max_new_tokens = opt.max_new_tokens

    tokenizer = AutoTokenizer.from_pretrained(opt.llm_path)
    raw_dataset = json.load(open(opt.dataset_path))

    eval_set = SFTPLSQLGenerationDataset(
        opt.dataset_path,
        tokenizer,
        max_tokens - max_new_tokens,
        "eval",
        opt.table_num,
        opt.column_num,
        opt.sic_path,
        opt.skeleton_predictor_path,
        opt.is_filter_schema,
        opt.cot,
        opt.skeleton,
        opt.test_plsql_skeletons_path
    )

    dataloader = DataLoader(eval_set, batch_size = 1)
    model = AutoModelForCausalLM.from_pretrained(opt.llm_path, device_map = "auto", torch_dtype = torch.float16)

    model.eval()
    start_time = time.time()
    predicted_plsqls = []

    no_error_cnt = 0

    for raw_data, batch_data in tqdm(zip(raw_dataset, dataloader)):
        for key in batch_data:
            batch_data[key] = batch_data[key].to(model.device)
        generated_plsqls = text2plsql_func(model, batch_data, tokenizer, max_new_tokens)
        # TODO: postprocess
        generated_plsqls = [post_process(raw_data["schema"], generated_plsql) for generated_plsql in generated_plsqls]

        final_generated_plsql = None
        for generated_plsql in generated_plsqls:
            execution_error = check_plsql_executability(generated_plsql, raw_data["call"], raw_data["database"].lower(), opt.host, opt.port)
            if execution_error is None: # the generated sql has no execution errors, we will return it as the final generated sql
                final_generated_plsql = generated_plsql
                no_error_cnt += 1
                break
        if final_generated_plsql is None:
            if generated_plsqls[0].strip() != "":
                final_generated_plsql = generated_plsqls[0]
            else:
                final_generated_plsql = "PLSQL placeholder"

        print(final_generated_plsql)
        predicted_plsqls.append(final_generated_plsql)
    
    end_time = time.time()

    accurate_plsql_cnt = 0
    exact_match_cnt = 0

    total_insert_cnt = 0
    total_update_cnt = 0
    total_delete_cnt = 0
    total_if_cnt = 0
    total_loop_cnt = 0

    accurate_insert_cnt = 0
    accurate_update_cnt = 0
    accurate_delete_cnt = 0
    accurate_if_cnt = 0
    accurate_loop_cnt = 0

    # Evaluation
    for idx, (raw_data, predicted_plsql) in enumerate(zip(raw_dataset, predicted_plsqls)):
        print("------------------------------------------------------------------------------")
        print("------------------------------------------------------------------------------")
        print("idx: ", idx)
        print("gold plsql: ", raw_data["plsql"])
        print("predicted plsql: ", predicted_plsql)

        total_insert_cnt += raw_data["insert"]
        total_update_cnt += raw_data["update"]
        total_delete_cnt += raw_data["delete"]
        total_if_cnt += raw_data["if"]
        total_loop_cnt += raw_data["loop"]

        if is_exact_match(raw_data["plsql"], predicted_plsql):
            exact_match_cnt += 1

        if compare_plsql(raw_data["database"], raw_data["plsql"], predicted_plsql, raw_data["call"], True, opt.host, opt.port):
            accurate_plsql_cnt += 1

            accurate_insert_cnt += raw_data["insert"]
            accurate_update_cnt += raw_data["update"]
            accurate_delete_cnt += raw_data["delete"]
            accurate_if_cnt += raw_data["if"]
            accurate_loop_cnt += raw_data["loop"]

            print("Success")
        else:
            print("Failed")

    total_insert_cnt = 1 if total_insert_cnt == 0 else total_insert_cnt
    total_update_cnt = 1 if total_update_cnt == 0 else total_update_cnt
    total_delete_cnt = 1 if total_delete_cnt == 0 else total_delete_cnt
    total_if_cnt = 1 if total_if_cnt == 0 else total_if_cnt
    total_loop_cnt = 1 if total_loop_cnt == 0 else total_loop_cnt
    
    print("LLM name: {} | Total time: {}s | Average time: {}s | Example number: {} | No Error predict number: {} | Accurate plsql number: {} | EX: {} | EM: {} | INSERT ACC: {} | UPDATE ACC: {} | DELETE ACC: {} | IF ACC: {} | LOOP ACC: {}".format(
        opt.llm_path, 
        end_time - start_time,
        (end_time - start_time) / len(raw_dataset),
        len(raw_dataset),
        no_error_cnt,
        accurate_plsql_cnt,
        accurate_plsql_cnt / len(raw_dataset),
        exact_match_cnt / len(raw_dataset),
        accurate_insert_cnt / total_insert_cnt,
        accurate_update_cnt / total_update_cnt,
        accurate_delete_cnt / total_delete_cnt,
        accurate_if_cnt / total_if_cnt,
        accurate_loop_cnt / total_loop_cnt
        )
    )