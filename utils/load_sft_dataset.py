import json
import torch
import gc
import os
import pickle

from datasets import Dataset
from torch.utils.data import Dataset
from schema_item_filter import SchemaItemClassifierInference, filter_schema
from transformers import T5Tokenizer, T5ForConditionalGeneration
from utils.db_utils import get_db_schema_sequence

def prepare_text2plsql_prefix_sequence(data, cot, skeleton):

    prefix_seq = ""

    if cot == 1:
        prefix_seq += (get_CoT(data["text"]) + "\n")
    prefix_seq += data["schema_sequence"] + "\n"
    if skeleton == 1:
        prefix_seq += ("Skeleton: " + data["predict_skeleton"] + "\n")
    prefix_seq += (data["text"] + "\n")

    prefix_seq += "Output the PL/SQL procedure. Begin with CREATE OR REPLACE PROCEDURE sp();\n"

    return prefix_seq

def prepare_inputs_and_labels(prefix_seq, target_seq, tokenizer, max_tokens):
    prefix_ids = [tokenizer.bos_token_id] + tokenizer(prefix_seq , truncation = False)["input_ids"]
    target_ids = tokenizer(target_seq, truncation = False)["input_ids"] + [tokenizer.eos_token_id]

    seq_length = len(prefix_ids) + len(target_ids)
    if seq_length <= max_tokens: # pad inputs with pad_token_id
        pad_length = max_tokens - seq_length
        input_ids = prefix_ids + target_ids + [tokenizer.pad_token_id] * pad_length
        # tell the model to ignore the padding tokens when performing (masked) self-attention 
        attention_mask = [1] * seq_length + [0] * pad_length
        # only target_ids produces gradients
        labels = [-100] * len(prefix_ids) + target_ids + [-100] * pad_length
    else: # no padding
        print("the current input sequence exceeds the max_tokens, we will truncate it.")
        input_ids = prefix_ids + target_ids
        # pre-truncate input ids
        input_ids = [tokenizer.bos_token_id] + input_ids[-(max_tokens-1):]
        attention_mask = [1] * max_tokens
        # only target_ids produces gradients
        labels = [-100] * len(prefix_ids) + target_ids
        # pre-truncate labels
        labels = labels[-max_tokens:]

    return {
        "input_ids": torch.tensor(input_ids, dtype = torch.int64), 
        "attention_mask": torch.tensor(attention_mask, dtype = torch.int64), 
        "labels": torch.tensor(labels, dtype = torch.int64)
    }

def prepare_inputs(prefix_seq, tokenizer, max_prefix_length):
    input_ids = [tokenizer.bos_token_id] + tokenizer(prefix_seq , truncation = False)["input_ids"]

    if len(input_ids) > max_prefix_length:
        print("the current input sequence exceeds the max_tokens, we will truncate it.")
        input_ids = [tokenizer.bos_token_id] + input_ids[-(max_prefix_length-1):]
    
    attention_mask = [1] * len(input_ids)
    
    return {
        "input_ids": torch.tensor(input_ids, dtype = torch.int64),
        "attention_mask": torch.tensor(attention_mask, dtype = torch.int64)
    }

def get_CoT(text):
    if "oracle" in text.lower():
        return "To generate the PL/SQL code for the procedure: Start with the procedure header using CREATE OR REPLACE PROCEDURE sp(); Specify the required parameters with correct Oracle types like VARCHAR2 or NUMBER. After the header, declare the procedure body with IS, and add any needed variable or cursor declarations. Declare local variables used within the procedure body, including appropriate types and optional default values. Begin the logic with a BEGIN block. Consider what control structures are needed—do you need a LOOP, IF statement, or a cursor with FETCH and WHERE CURRENT OF? Use proper OPEN, FETCH, and CLOSE for cursors, and IF...THEN for conditional logic. Finish with END;, ensuring the logic handles relevant conditions and updates data as expected."
    return "To generate the PLpgSQL code for the procedure: Start with the procedure header using CREATE OR REPLACE PROCEDURE sp(); Specify the required parameters with correct PostgreSQL types like TEXT, INTEGER, or NUMERIC. After the header, declare the language using LANGUAGE plpgsql;. Then, declare the procedure body using a DECLARE block if needed, and add any variable or cursor declarations. Declare local variables used within the procedure body, including appropriate types and optional default values. Begin the logic with a BEGIN block. Consider what control structures are needed—do you need a LOOP, IF statement, or a cursor with FETCH and WHERE CURRENT OF? Use proper OPEN, FETCH, and CLOSE for cursors, and IF...THEN for conditional logic. Finish with END;, ensuring the logic handles relevant conditions and updates data as expected."

def generate_skeletons(skeleton_predictor_path, plsql_texts, cache_file):

    # 检查缓存文件是否存在
    if os.path.exists(cache_file):
        print(f"Loading plsql skeletons from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

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

    # 保存缓存
    if cache_file:
        print(f"Saving plsql skeletons to cache: {cache_file}")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(skeletons, f)

    return skeletons


class SFTPLSQLGenerationDataset(Dataset):
    def __init__(self, text2plsql_data_dir, tokenizer, max_tokens, mode, table_num, column_num, sic_path, skeleton_predictor_path, is_filter_schema, cot, skeleton, test_plsql_skeletons_path=None):
        super().__init__()
        dataset = json.load(open(text2plsql_data_dir))

        print("apply filtering strategies...")
        if mode == "train":
            dataset = filter_schema(dataset, "train", None, table_num, column_num)
            for i, data in enumerate(dataset):
                data["predict_skeleton"] = data["skeleton"]

        elif mode == "eval":
            if is_filter_schema == 1:
                sic = SchemaItemClassifierInference(sic_path)
                dataset = filter_schema(dataset, "eval", sic, table_num, column_num)
                del sic
                torch.cuda.empty_cache()
            if skeleton == 1:
                text_list = [data["text"] for data in dataset]
                predict_skeletons = generate_skeletons(skeleton_predictor_path, text_list, test_plsql_skeletons_path)
                for i, data in enumerate(dataset):
                    data["predict_skeleton"] = predict_skeletons[i]

        # prepare schema sequence and content sequence
        for data in dataset:
            data["schema_sequence"] = get_db_schema_sequence(data["schema"])

        self.mode = mode
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.cot = cot
        self.skeleton = skeleton

        if mode == "train":
            self.cot = 1
            self.skeleton = 1
            
        
    def __getitem__(self, index):
        data = self.dataset[index]
        prefix_seq = prepare_text2plsql_prefix_sequence(data, self.cot, self.skeleton)
        if index < 2:
            print(prefix_seq)

        if self.mode == "train":
            target_seq = data["plsql"]
            return prepare_inputs_and_labels(prefix_seq, target_seq, self.tokenizer, self.max_tokens)
        elif self.mode == "eval":
            return prepare_inputs(prefix_seq, self.tokenizer, self.max_tokens)

    def __len__(self):
        return len(self.dataset)