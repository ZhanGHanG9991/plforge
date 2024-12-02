import argparse
from utils.db_utils import check_plsql_executability, compare_plsql, get_db_schema_sequence
from plsql_skeleton_similarity import get_plsql_skeleton_similarity
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import json
import nltk
import time
import re
import random
import numpy as np
import torch
from tqdm import tqdm
from simcse import SimCSE
from transformers.trainer_utils import set_seed


from schema_item_filter import SchemaItemClassifierInference, filter_schema

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_path', type = str)
    parser.add_argument('--sic_path', type = str)
    parser.add_argument('--table_num', type = int, default = 5)
    parser.add_argument('--column_num', type = int, default = 6)
    
    parser.add_argument('--dataset_path', type = str)
    parser.add_argument('--demonstration_set_path', type = str)
    parser.add_argument('--num_of_demonstrations', type = int)

    parser.add_argument('--max_tokens', type = int)
    parser.add_argument('--max_new_tokens', type = int)

    parser.add_argument('--skeleton_predictor_path', type = str)

    parser.add_argument('--is_filter_schema', type = int)
    parser.add_argument('--cot', type = int)
    parser.add_argument('--skeleton', type = int)
    parser.add_argument('--new_similarity', type = int)
    parser.add_argument('--similarity_w', type = float)


    opt = parser.parse_args()

    return opt

def prepare_cross_domain_input_seq(opt, eval_data, demonstration_set, similarity):
    top_k_indices = sorted(range(len(similarity)), key = lambda x: similarity[x], reverse = True)[:opt.num_of_demonstrations]
    print(top_k_indices)
    print(similarity[top_k_indices])

    input_seq = ""
    for idx in top_k_indices:
        demonstration_plsql = demonstration_set[idx]["plsql"]
        input_seq += demonstration_set[idx]["text"] + "\n" + demonstration_plsql + "\n\n"

    input_seq += eval_data["schema_sequence"] + "\n"
    if opt.cot == 1:
        input_seq += (get_CoT() + "\n")
    if opt.skeleton == 1:
        input_seq += ("Skeleton: " + eval_data["skeleton"] + "\n")
    input_seq += (eval_data["text"] + "\n")

    return input_seq

def prepare_input_ids_and_attention_mask(tokenizer, input_seq, max_input_length, device):
    input_ids = tokenizer(input_seq , truncation = False)["input_ids"]

    if len(input_ids) <= max_input_length:
        input_ids = input_ids
        attention_mask = [1] * len(input_ids)
    else:
        if tokenizer.name_or_path == "THUDM/codegeex2-6b":
            input_ids = [64790, 64792] + input_ids[-(max_input_length-2):]
        else:
            input_ids = [tokenizer.bos_token_id] + input_ids[-(max_input_length-1):]

        attention_mask = [1] * max_input_length
    
    print("len(input_ids):", len(input_ids))
 
    return {
        "input_ids": torch.tensor([input_ids]).to(device), # torch.int64
        "attention_mask": torch.tensor([attention_mask]).to(device) # torch.int64
    }

# TODO eos_token_id
def text2plsql_func(model, text2plsql_input_seq, tokenizer, max_tokens, max_new_tokens):
    inputs = prepare_input_ids_and_attention_mask(
        tokenizer, 
        text2plsql_input_seq, 
        max_tokens - max_new_tokens,
        model.device
    )

    input_length = inputs["input_ids"].shape[1]

    # check_tokenizer(tokenizer, inputs["input_ids"])

    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            num_beams = 4,
            num_return_sequences = 4,
            use_cache = True
        )

    generated_plsqls = tokenizer.batch_decode(generate_ids[:, input_length:], skip_special_tokens = True, clean_up_tokenization_spaces = False)

    return generated_plsqls

# extract the skeleton of the input text
def extract_skeleton(text):
    tokens_and_tags = nltk.pos_tag(nltk.word_tokenize(text))

    output_tokens = []
    for token, tag in tokens_and_tags:
        if tag in ['NN', 'NNP', 'NNS', 'NNPS', 'CD', 'SYM', 'FW', 'IN']:
            output_tokens.append("_")
        elif token in ['$', "''", '(', ')', ',', '--', '.', ':']:
            pass
        else:
            output_tokens.append(token)
    
    text_skeleton = " ".join(output_tokens)
    text_skeleton = text_skeleton.replace("_ 's", "_")
    text_skeleton = text_skeleton.replace(" 's", "'s")

    while("_ _" in text_skeleton):
        text_skeleton = text_skeleton.replace("_ _", "_")
    while("_ , _" in text_skeleton):
        text_skeleton = text_skeleton.replace("_ , _", "_")
    
    if text_skeleton.startswith("_ "):
        text_skeleton = text_skeleton[2:]
    
    return text_skeleton

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

    # end with $$;
    index = plsql.find("$$;")
    if index != -1:
        plsql = plsql[:index + 3]
    
    # add "" for postgres
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

    # avoid "$ $"
    for i in range(len(final_token_list) - 1):
        if final_token_list[i] == "$" and final_token_list[i+1] == "$":
            final_token_list[i] = "$$"

    final_token_list = [token for token in final_token_list if token != "$"]

    # avoid "> =" and "< ="
    for i in range(len(final_token_list) - 1):
        if (final_token_list[i] == "=" or final_token_list[i] == "<" or final_token_list[i] == ">") and final_token_list[i+1] == "=":
            final_token_list[i] = final_token_list[i] + "="
            final_token_list[i+1] = ""
        elif final_token_list[i] == "<" and final_token_list[i+1] == ">":
            final_token_list[i] = final_token_list[i] + ">"
            final_token_list[i+1] = ""

    # avoid " " Something " "
    post_plsql = re.sub(r'" +"', '"', " ".join(final_token_list))

    # avoid " Something "
    post_plsql = re.sub(r'"\s*(.*?)\s*"', r'"\1"', post_plsql)
    
    return post_plsql

def is_exact_match(plsql1, plsql2):
    format_plsql1 = ''.join(plsql1.split())
    format_plsql2 = ''.join(plsql2.split())
    return format_plsql1 == format_plsql2

def get_CoT():
    return "Let's think step by step. 1. Begin with CREATE OR REPLACE PROCEDURE sp(); 2. Consider the parameter number and types; 3. Declare LANGUAGE plpgsql; 4. Use BEGIN and END; 5. Consider code logic, the given skeleton and whether to use IF or LOOP. Finally, output the PLpgSQL."

def generate_skeletons(skeleton_predictor_path, plsql_texts):
    # 确保模型和tokenizer已经加载
    model_dir = skeleton_predictor_path
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    skeletons = []

    for plsql_text in plsql_texts:
        # 将输入文本转换为模型的输入格式
        input_encoding = tokenizer(
            plsql_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        ).to(device)

        # 使用模型生成输出
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_encoding['input_ids'],
                attention_mask=input_encoding['attention_mask'],
                max_length=512,
                min_length=50,
                num_beams=2,
                early_stopping=True
            )

        # 解码并添加到skeletons列表
        skeleton = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        skeletons.append(skeleton)

    return skeletons

def normalize_similarities(similarities):
    # 二维 numpy 数组
    min_val = np.min(similarities)
    max_val = np.max(similarities)

    normalized_similarities = (similarities - min_val) / (max_val - min_val)

    return normalized_similarities


if __name__ == "__main__":
    set_seed(42)
    opt = parse_option()
    print(opt)

    # load the evaluation set
    eval_set = json.load(open(opt.dataset_path))
    eval_set_questions = [data["text"] for data in eval_set]
    eval_set_question_skeletons = [extract_skeleton(question) for question in eval_set_questions]
    eval_set_plsql_skeletons = generate_skeletons(opt.skeleton_predictor_path, eval_set_questions)

    print("length of evaluation set:", len(eval_set))

     # load the demonstration pool
    demonstration_set = json.load(open(opt.demonstration_set_path))
    demonstration_set_questions = [data["text"] for data in demonstration_set]
    demonstration_set_question_skeletons = [extract_skeleton(question) for question in demonstration_set_questions]
    demonstration_set_plsql_skeletons = [data["skeleton"] for data in demonstration_set]

    print("length of demonstration set:", len(demonstration_set))

    # Filter schema
    if opt.is_filter_schema == 1:
        demonstration_set = filter_schema(demonstration_set, "train", None, opt.table_num, opt.column_num)
        sic = SchemaItemClassifierInference(opt.sic_path)
        eval_set = filter_schema(eval_set, "eval", sic, opt.table_num, opt.column_num)
        del sic
        torch.cuda.empty_cache()

    # TODO Add database metadata
    # for demonstration_sample in demonstration_set:
    #     demonstration_sample["schema_sequence"] = get_db_schema_sequence(demonstration_sample["database"])
    for eval_sample in eval_set:
        eval_sample["schema_sequence"] = get_db_schema_sequence(eval_sample["schema"])

    # compute similarities between questions in the evaluation set and the demonstration pool
    simsce_model = SimCSE("/home/zhanghang/opt/models/princeton-nlp_sup-simcse-roberta-base")
    question_similarities = simsce_model.similarity(eval_set_questions, demonstration_set_questions)
    question_skeleton_similarities = simsce_model.similarity(eval_set_question_skeletons, demonstration_set_question_skeletons)

    if opt.new_similarity == 1:
        plsql_skeleton_similarities = get_plsql_skeleton_similarity(eval_set_plsql_skeletons, demonstration_set_plsql_skeletons)

        normalized_question_skeleton_similarities = normalize_similarities(question_skeleton_similarities)
        normalized_plsql_skeleton_similarities = normalize_similarities(np.array(plsql_skeleton_similarities))
        
        w = opt.similarity_w
        similarities = w * normalized_question_skeleton_similarities + (1 - w) * normalized_plsql_skeleton_similarities
    
    elif opt.new_similarity == 0:
        similarities = np.maximum(question_similarities, question_skeleton_similarities)
    
    del simsce_model

    tokenizer = AutoTokenizer.from_pretrained(opt.llm_path)
    model = AutoModelForCausalLM.from_pretrained(opt.llm_path, device_map = "auto", torch_dtype = torch.float16)
    model.eval()
    print(model.dtype)

    # TODO 修改模型生成的终止条件，生成$$;则终止

    max_tokens = opt.max_tokens
    max_new_tokens = opt.max_new_tokens

    print("max_tokens:", max_tokens)
    print("max_new_tokens:", max_new_tokens)

    start_time = time.time()

    predicted_plsqls = []
    no_error_cnt = 0

    for eval_data_idx, eval_data in tqdm(enumerate(eval_set)):
        input_seq = prepare_cross_domain_input_seq(opt, eval_data, demonstration_set, similarities[eval_data_idx])

        if eval_data_idx < 2:
            print(input_seq)

        generated_plsqls = text2plsql_func(model, input_seq, tokenizer, max_tokens, max_new_tokens)
        # TODO postprocess
        generated_plsqls = [post_process(eval_data["schema"], generated_plsql) for generated_plsql in generated_plsqls]

        final_generated_plsql = None
        for generated_plsql in generated_plsqls:
            execution_error = check_plsql_executability(generated_plsql, eval_data["call"], eval_data["database"].lower())
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

        if eval_data_idx % 10 == 0:
            torch.cuda.empty_cache()

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
    for idx, (eval_data, predicted_plsql) in enumerate(zip(eval_set, predicted_plsqls)):
        print("------------------------------------------------------------------------------")
        print("------------------------------------------------------------------------------")
        print("idx: ", idx)
        print("gold plsql: ", eval_data["plsql"])
        print("predicted plsql: ", predicted_plsql)

        total_insert_cnt += eval_data["insert"]
        total_update_cnt += eval_data["update"]
        total_delete_cnt += eval_data["delete"]
        total_if_cnt += eval_data["if"]
        total_loop_cnt += eval_data["loop"]

        if is_exact_match(eval_data["plsql"], predicted_plsql):
            exact_match_cnt += 1

        if compare_plsql(eval_data["database"], eval_data["table"], eval_data["plsql"], predicted_plsql, eval_data["call"]):
            accurate_plsql_cnt += 1

            accurate_insert_cnt += eval_data["insert"]
            accurate_update_cnt += eval_data["update"]
            accurate_delete_cnt += eval_data["delete"]
            accurate_if_cnt += eval_data["if"]
            accurate_loop_cnt += eval_data["loop"]

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
        (end_time - start_time) / len(eval_set),
        len(eval_set),
        no_error_cnt,
        accurate_plsql_cnt,
        accurate_plsql_cnt / len(eval_set),
        exact_match_cnt / len(eval_set),
        accurate_insert_cnt / total_insert_cnt,
        accurate_update_cnt / total_update_cnt,
        accurate_delete_cnt / total_delete_cnt,
        accurate_if_cnt / total_if_cnt,
        accurate_loop_cnt / total_loop_cnt
        )
    )