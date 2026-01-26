import argparse
import json
import os
import tempfile
import shutil
from tqdm import tqdm
from vllm import LLM, SamplingParams
from utils import split_sample, my_load_dataset, verify_preds, build_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def create_chat_messages(sample, options, dataset, demo_examples, n_shot):
    """Create chat messages based on dataset type with few-shot examples"""
    messages = []
    # Add few-shot examples
    for ex in demo_examples[:n_shot]:
        messages.append({"role": "user", "content": ex["question"]})
        messages.append({"role": "assistant", "content": ex["answer"]})
    messages.append({"role": "user", "content": build_prompt(sample["question"], options, dataset)})
    return messages


def main():
    parser = argparse.ArgumentParser(description="vLLM Batch Inference with Few-Shot Support")
    parser.add_argument('--model_path', type=str, default="meta-llama/Llama-2-7b-chat-hf", help='Path to trained model')
    parser.add_argument('--lora_path', type=str, default=None, help='Path to LoRA adapters (if using LoRA model)')
    parser.add_argument('--dataset', default='gsm8k', choices=['gsm8k', 'math', 'math500', 'gpqa'], help="Dataset for evaluation")
    parser.add_argument('--max_new_tokens', type=int, default=512, help='Max new tokens to generate')
    parser.add_argument('--n_shot', type=int, default=0, help='Number of few-shot examples to prepend to each prompt')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Number of GPUs to use for tensor parallelism')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help='GPU memory utilization ratio')
    args = parser.parse_args()

    # python test_vllm.py --model_path "meta-llama/Llama-2-7b-chat-hf" --lora_path "checkpoints/Llama-2-7b-chat-hf_gsm8k_Qwen3-1.7B_top10" --dataset 'gsm8k' --tensor_parallel_size 4
    
    if not args.lora_path:
        vllm_path = args.model_path
        output_dir = None
    else:   # vllm only accepts merged models
        base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype="auto",
        device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, args.lora_path)
        merged_model = model.merge_and_unload()
        output_dir = tempfile.mkdtemp(prefix="merged_model_")
        merged_model.save_pretrained(output_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        tokenizer.save_pretrained(output_dir)
        vllm_path = output_dir

    # Initialize vLLM model
    llm = LLM(
        model=vllm_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="float16",
        trust_remote_code=True,
    )
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
    )

    # Load dataset
    train_data, test_data = my_load_dataset(args.dataset)
    print(f'Dataset size: {len(test_data)}')

    # Build n-shot demo examples
    demo_examples = []
    if args.n_shot > 0:
        for i in range(args.n_shot):
            demo_examples.append({
                "question": train_data[i]["question"],
                "answer": train_data[i]["answer"]
            })

    # Preprocess all samples
    samples = []
    chat_messages = []
    golds = []
    
    for i, sample in enumerate(test_data):
        if args.dataset == 'gpqa':
            ques, options, final_ans = split_sample(sample, args.dataset)
            processed_sample = {
                "id": i,
                "question": ques,
                "options": options,
                "final_answer": final_ans,
                }
        else:
            ques, ration, final_ans = split_sample(sample, args.dataset)
            options = None
            processed_sample = {
                "id": i,
                "question": ques,
                "gold_answer": ration,
                "final_answer": final_ans
            }
        
        samples.append(processed_sample)
        golds.append(final_ans)
        chat_messages.append(create_chat_messages(processed_sample, options, args.dataset, demo_examples, args.n_shot))

    print("Starting batch inference with vLLM...")
    
    # Perform batch inference using vLLM chat method
    outputs = llm.chat(
        messages=chat_messages,
        sampling_params=sampling_params,
        use_tqdm=True
    )

    # Process results and calculate accuracy
    correct = 0
    total = 0
    all_predictions = []
    
    for i, (sample, output, gold) in enumerate(zip(samples, outputs, golds)):
        # Extract the generated text from the output
        pred_ans = output.outputs[0].text.strip()
        
        # Verify prediction
        is_correct = verify_preds(pred_ans, gold, args.dataset)
        correct += int(is_correct)
        total += 1
        all_predictions.append(pred_ans)

    acc = correct / total
    print(f"Accuracy: {acc:.4f} ({correct}/{total})")

    # Save results
    model_name = os.path.basename(os.path.normpath(args.model_path))
    if not args.lora_path:
        model_name = args.model_path.split('/')[1] if '/' in args.model_path else model_name
    else:
        model_name = args.lora_path.split('/')[1] if '/' in args.lora_path else model_name
    
    file_name = f'{model_name}_{args.dataset}_{args.n_shot}-shot_results.json'
    
    # Prepare results
    results = {
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "model": args.model_path,
        "dataset": args.dataset,
        "n_shot": args.n_shot,
        "predictions": []
    }
    
    for sample, pred, gold in zip(samples, all_predictions, golds):
        results["predictions"].append({
            "id": sample["id"],
            "question": sample["question"],
            "prediction": pred,
            "gold_answer": gold,
        })
    
    # Create results directory if it doesn't exist
    os.makedirs('test_results', exist_ok=True)
    output_path = f'test_results/{file_name}'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_path}")

    # delete tmp merged lora model path
    if output_dir and os.path.exists(output_dir):
        print(f"Cleaning up temporary directory: {output_dir}")
        shutil.rmtree(output_dir)

if __name__ == '__main__':
    main()