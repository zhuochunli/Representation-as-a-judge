import argparse
import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
import torch
from sklearn.model_selection import train_test_split
import os
import wandb


parser = argparse.ArgumentParser(description='Tune LLMs with LoRA for baselines.')
parser.add_argument('--model_path', type=str, default="Qwen/Qwen3-0.6B", help='Model name or path')
parser.add_argument('--data_path', type=str, default="Meta-Llama-3-8B-Instruct_alpaca2_roscoe5dim_probing.json", help='Path to filtered JSON data')
parser.add_argument('--aspect', default='semantic_consistency', choices=['semantic_consistency', 'logicality', 'informativeness','fluency','factuality'], help="Aspects for evaluation")
parser.add_argument('--classes', default='multi', choices=['multi','binary'], help="Classes for evaluation")
parser.add_argument('--output_dir', type=str, default="checkpoints/Qwen3-0.6B_gpqa_semantic_consistency_multi", help='Directory to save the trained model')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=8, help='Number of epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--max_length', type=int, default=1024, help='Max sequence length')
parser.add_argument('--select', type=str, choices=['all', 'top', 'random', 'score'], default='all', help='Data selection mode')
parser.add_argument('--percent', type=float, default=0.2, help='Fraction of top data to keep (e.g., 0.2 for 20%)')
parser.add_argument('--scores', type=str, default='[0,1,2,3,4,5]', help='which score data to keep')    
parser.add_argument('--seed', type=int, default=731, help='Random seed for reproducibility')
# LoRA specific arguments
parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha parameter')
parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
args = parser.parse_args()

# python tuning_lora.py --data_path "Meta-Llama-3-8B-Instruct_gsm8k_roscoe5dim_probing.json" --class "multi" --aspect "semantic_consistency" --output_dir "checkpoints/Qwen3-0.6B_gsm8k_multi_semantic_consistency"

os.environ["WANDB_PROJECT"] = "pingan-evaluation"
wandb.login(key="0f3dc9485e521a91e13546fa2d28f64b2f5aa8e9")
wandb.init(name=args.output_dir)

class QADataset(torch.utils.data.Dataset):
    def __init__(self, samples, tokenizer, max_length=1024):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        sample = self.samples[idx]
        if args.classes=="multi":
            messages = [
                {"role": "user", "content": sample['eval_prompt']},
                {"role": "assistant", "content": str(sample['score'])},
            ]
        elif args.classes=="binary":
            messages = [
                {"role": "user", "content": sample['eval_prompt']},
                {"role": "assistant", "content": "1" if sample['score']>3 else "0"},
            ]
        # Get the prompt as a string
        prompt_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        # Tokenize the prompt string
        enc = self.tokenizer(
            prompt_str,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )
        # Squeeze batch dimension
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc["labels"] = enc["input_ids"].clone()
        return enc

def select_data(data, mode, percent, scores, seed=731):
    # Accepts: list of (score, sample) or just sample dicts
    if mode == 'all':
        return data
    elif mode == 'top':
        sorted_data = sorted(data, key=lambda x: x['total_score'], reverse=True)
        n_keep = max(1, int(len(sorted_data) * percent))
        return sorted_data[:n_keep]
    elif mode == 'random':
        random.seed(seed)
        n_keep = max(1, int(len(data) * percent))
        return random.sample(data, n_keep)
    elif mode == 'score':
        keep_scores = set(json.loads(scores))
        return [s for s in data if s['total_score'] in keep_scores]

def main():
    # Set random seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Load data
    with open(args.data_path, 'r') as f:
        data = json.load(f)[args.aspect]
    # Select data
    train_samples = select_data(data, args.select, args.percent, args.scores, args.seed)
    train_samples, val_samples = train_test_split(train_samples, test_size=0.2, random_state=args.seed)
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Load tokenizer/model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        device_map="auto",
        torch_dtype=torch.float16,  # Use half precision for memory efficiency
        )

    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    model.train() 
    model.print_trainable_parameters()

    # Build dataset
    train_dataset = QADataset(train_samples, tokenizer, max_length=args.max_length)
    val_dataset = QADataset(val_samples, tokenizer, max_length=args.max_length)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training args - optimized for Llama-2-7B student model
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        learning_rate=args.lr, 
        save_strategy='epoch',
        eval_strategy="epoch",
        logging_steps=10,
        report_to=["wandb"],
        run_name=args.output_dir,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),  # Use bf16 if available
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),  # Fallback to fp16
        remove_unused_columns=False,
        save_total_limit=1,
        load_best_model_at_end=True,    # save the best model with the lowest eval_loss                    
        metric_for_best_model="eval_loss",             
        greater_is_better=False,
        warmup_steps=100,  # Add warmup steps
        weight_decay=0.01,  # Add weight decay
        dataloader_pin_memory=False,  # Can help with memory usage
        gradient_checkpointing=False,  # Save memory during training
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        eval_dataset=val_dataset,
    )

    trainer.train()
    
    # Save the LoRA adapters
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapters saved to {args.output_dir}")
    
    # Optionally save the merged model (base model + LoRA weights)
    # merged_model = model.merge_and_unload()
    # merged_output_dir = args.output_dir + "_merged"
    # merged_model.save_pretrained(merged_output_dir)
    # tokenizer.save_pretrained(merged_output_dir)
    # print(f"Merged model saved to {merged_output_dir}")


if __name__ == '__main__':
    main()