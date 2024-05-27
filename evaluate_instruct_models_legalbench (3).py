# !pip install datasets transformers accelerate trl bitsandbytes wandb peft --quiet
# !git clone https://github.com/HazyResearch/legalbench.git
# Imports
import os
import gc
import datasets
from tqdm import tqdm
import pandas as pd
import datasets
import numpy as np
import os
import numpy as np
import random
import torch
import wandb
import argparse
import json

from legalbench.tasks import TASKS
from legalbench.utils import generate_prompts
from transformers import GenerationConfig, pipeline
from huggingface_hub import login
from transformers import set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DataCollatorForCompletionOnlyLM
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

run = wandb.init(project='exp_4', name='llama3_70b')
set_seed(42)

max_len = 8000
## CSV
# eval_df['len'] = [len(prompt) for prompt in eval_df['prompt']]
# eval_df = eval_df[eval_df['len'] <= max_len]
# print('shape of eval dataset after limiting max length')
# print(eval_df.shape)
# all_legelbench_prompts_dataset = datasets.Dataset.from_pandas(eval_df)

## Dataset
all_legelbench_prompts_dataset = datasets.load_dataset("", split="train")
all_legelbench_prompts_dataset = all_legelbench_prompts_dataset.filter(lambda row: len(row['text_no_answer']) <= max_len)

print(all_legelbench_prompts_dataset)

config = {
    "loading_config": {
        "pretrained_model_name_or_path": "meta-llama/Meta-Llama-3-70B",
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        "quantization_config": {
            "load_in_4bit": True,
            "bnb_4bit_use_double_quant": False,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype":torch.float16
        }
    }
}


loading_config = config['loading_config']

# Get model
# actual_model = AutoModelForCausalLM.from_pretrained(**loading_config)
# tokenizer = AutoTokenizer.from_pretrained(loading_config['pretrained_model_name_or_path'])
 
## PEFT model
peft_model_id = ""
 
# Load Model with PEFT adapter
actual_model = AutoPeftModelForCausalLM.from_pretrained(
  peft_model_id,
  torch_dtype=torch.float16,
  quantization_config= {"load_in_4bit": True},
  device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "left"


gen_config = GenerationConfig.from_pretrained(loading_config['pretrained_model_name_or_path'],max_new_tokens=10,low_memory=True)

def generate_output(prompt):
    device = 'cuda'
    tokenized_prompt = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    with torch.inference_mode():
        output = actual_model.generate(
            tokenized_prompt,
            generation_config=gen_config,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=1,
        )
    return tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=True)

model_outputs = []
for row in tqdm(all_legelbench_prompts_dataset):
    decoded_output = generate_output(row['text_no_answer'])
    model_outputs.append(decoded_output.split("\n")[0].strip())

print("Outputs:", model_outputs[:10])

all_legelbench_prompts_dataset = all_legelbench_prompts_dataset.add_column("generation", model_outputs)
all_legelbench_prompts_dataset.push_to_hub('')

all_legelbench_prompts = all_legelbench_prompts_dataset.to_pandas()
all_legelbench_prompts.to_csv("exp_4.csv", index=False)

test_predictions = wandb.Artifact("predictions", type="predictions")
eval_table = wandb.Table(dataframe=all_legelbench_prompts)
test_predictions.add(eval_table, "sampled")
run.log_artifact(test_predictions)

## Submission
from legalbench.evaluation import evaluate as evaluatelb

def get_task_type(data_df):
    from legalbench.tasks import TASKS, ISSUE_TASKS, RULE_TASKS, CONCLUSION_TASKS, INTERPRETATION_TASKS, RHETORIC_TASKS
    mapper = {
        'issue_spotting': ISSUE_TASKS,
        'rule_recall': RULE_TASKS,
        'rule_conclusion': CONCLUSION_TASKS,
        'interpretation': INTERPRETATION_TASKS,
        'rhetoric_understanding': RHETORIC_TASKS,
    }
    reverse_mapper = {}
    for key in mapper:
        for subkey in mapper[key]:
            reverse_mapper[subkey] = key

    data_df['task_type'] = data_df['task'].apply(lambda x: reverse_mapper[x])
    return data_df

def evaluate(data_df, target_column):
    def wrapper(df):
        return evaluatelb(
            task=df.name,
            generations=df[target_column].tolist(),
            answers=df['answer'].tolist(),
        )
    results = data_df.groupby('task').apply(wrapper)
    return results

def get_dfs(results, data_df):
    results_df = pd.DataFrame()
    results_df['example_count'] = data_df.groupby('task')['answer'].count()
    results_df['task_type'] = data_df.groupby('task')['task_type'].apply(lambda x: x.iloc[0])
    results_df['score'] = results
    results_df.groupby('task_type')['score'].aggregate(['mean'])
    results_df.groupby('task_type')['example_count'].aggregate(['count', 'sum'])
    final_results_df = pd.DataFrame()
    final_results_df['example_count'] = results_df.groupby('task_type')['example_count'].aggregate(['sum'])
    final_results_df['score'] = results_df.groupby('task_type')['score'].aggregate(['mean'])
    final_results_df['task_count'] = results_df.groupby('task_type')['example_count'].aggregate(['count'])
    return results_df, final_results_df

def create_results_report(data_df, target_column='generation'):
    data_df = data_df[data_df['task'] != 'rule_qa']
    data_df = get_task_type(data_df)
    results = evaluate(data_df, target_column=target_column)
    per_task_results_df, per_group_results_df = get_dfs(results, data_df)
    return per_task_results_df, per_group_results_df, data_df


# Evaluate
per_task_results_df, per_group_results_df, data_df = create_results_report(all_legelbench_prompts)
print('per_task_results_df')
print(per_task_results_df)
print('per_group_results_df')
print(per_group_results_df)

