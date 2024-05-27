## pip install wandb pandas --quiet
## # Install Pytorch for FSDP and FA/SDPA
# %pip install "torch==2.2.2" tensorboard --quiet
 
# Install Hugging Face libraries
# %pip install  --upgrade "transformers==4.40.0" "datasets==2.18.0" "accelerate==0.29.3" "evaluate==0.4.1" "bitsandbytes==0.43.1" "huggingface_hub==0.22.2" "trl==0.8.6" "peft==0.10.0"
## wandb.login()

import os
import pandas as pd
from datasets import load_dataset, Dataset

# Load dataset
sample_dataset = load_dataset(
        "csv",
        data_files="examples.csv",
    split="train"
).train_test_split(test_size=0.1)

print("Dataset", sample_dataset)
# save datasets to disk
sample_dataset["train"].to_json("train_dataset.json", orient="records", force_ascii=False)
sample_dataset["test"].to_json("test_dataset.json", orient="records", force_ascii=False)

# !ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=1 run_fsdp_qlora.py --config llama_3_70b_fsdp_qlora.yaml