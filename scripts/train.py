import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

def main():
    # Setup Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "dataset.jsonl")
    output_dir = os.path.join(base_dir, "models", "SDQ-LLM-LoRA")

    print(f"Loading dataset from {data_path}")
    dataset = load_dataset("json", data_files=data_path, split="train")

    # Use a small Korean model, e.g., skt/kogpt2-base-v2
    model_name = "skt/kogpt2-base-v2"
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, bos_token='</s>', eos_token='</s>', pad_token='<pad>')
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # If pad_token is missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # Format the Prompt
    def generate_prompt(data_point):
        if data_point["input"]:
            return f"명령어: {data_point['instruction']}\n입력: {data_point['input']}\n답변: {data_point['output']} </s>"
        else:
            return f"명령어: {data_point['instruction']}\n답변: {data_point['output']} </s>"

    def tokenize_function(examples):
        prompts = [generate_prompt(ex) for ex in (dict(zip(examples, t)) for t in zip(*examples.values()))]
        return tokenizer(prompts, truncation=True, padding="max_length", max_length=128)

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(lambda x: tokenizer(generate_prompt(x), truncation=True, padding="max_length", max_length=256), num_proc=1)

    # LoRA Configuration
    print("Configuring LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        # KoGPT2 uses standard attention names
    )
    
    # get_peft_model will apply LoRA to the linear layers
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Disable strictly requiring gradients for non-trainable params
    model.config.use_cache = False 

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,                     # Reduced from 3 to 1 for faster execution
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
        logging_steps=5,
        save_steps=50,
        eval_strategy="no",
        save_strategy="epoch",
        fp16=False, # Set to True if using CUDA and compatible GPU
        report_to="none"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("Starting Training...")
    trainer.train()

    print(f"Saving PEFT model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training Complete!")

if __name__ == "__main__":
    main()
