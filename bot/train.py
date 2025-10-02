import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
DATA_PATH = "whatsapp_training.jsonl"
OUTPUT_DIR = "./fine_tuned_phi3_bot"


dataset = load_dataset("json", data_files=DATA_PATH, split="train")


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def format_chat_template(example):

    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["response"]},
    ]

    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": formatted_text}

dataset = dataset.map(format_chat_template)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)


model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,  
    lora_alpha=32, 
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",    
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512, 
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names
)


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2, 
    gradient_accumulation_steps=4, 
    warmup_steps=20,
    num_train_epochs=3,
    learning_rate=2e-4, 
    fp16=True,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch", 
    optim="paged_adamw_32bit"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)


trainer.train()


model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Model fine-tuned and saved at {OUTPUT_DIR}")