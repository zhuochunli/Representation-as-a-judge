import json
import os
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams
from utils import split_sample, my_load_dataset, build_prompt

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"

parser = argparse.ArgumentParser(description='vLLM Batch Inference',
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--model_path', '-m', default='meta-llama/Meta-Llama-3-8B-Instruct',
                    help='Base model path')
parser.add_argument('--dataset', default='gsm8k', choices=['gsm8k', 'math', 'math500', 'gpqa', 'gsm8k-hard'],
                    help="Dataset for evaluation")
parser.add_argument('--max_new_tokens', type=int, default=512,
                    help='Maximum new tokens for generation')
parser.add_argument('--tensor_parallel_size', type=int, default=2,
                    help='Number of GPUs to use for tensor parallelism')
parser.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                    help='GPU memory utilization ratio')
args = parser.parse_args()


def predict(args):
    # Initialize vLLM model
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # download_dir="/jet/home/zli26/tmp_ondemand_ocean_cis240025p_symlink/zli26/connection_proj/Llama3",
        dtype="float16",
        trust_remote_code=True,
    )
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
        # stop_token_ids are automatically handled by vLLM for chat models
    )

    # Load dataset
    dataset, test_dataset = my_load_dataset(args.dataset)
    # dataset = [dataset[i] for i in range(16)]    # demo dataset for debug
    print(f'Dataset size: {len(dataset)}')

    # Preprocess all samples
    samples = []
    chat_messages = []
    
    for i, sample in enumerate(dataset):
        if 'gsm8k' in args.dataset or args.dataset == 'math500':
            ques, ration, final_ans = split_sample(sample, args.dataset)
            processed_sample = {
                "id": i,
                "question": ques,
                "gold_answer": ration,
                "final_answer": final_ans
            }
        elif args.dataset == 'math':
            ques, ration, final_ans = split_sample(sample, args.dataset)
            processed_sample = {
                "id": i,
                "question": ques,
                "gold_answer": ration,
            }
        elif args.dataset == 'commonsenseQA':
            ques, options, final_ans = split_sample(sample, args.dataset)
            processed_sample = {
                "id": i,
                "question": ques,
                "options": options,
                "final_answer": final_ans,
            }
        elif args.dataset == 'gpqa':
            ques, options, final_ans = split_sample(sample, args.dataset)
            processed_sample = {
                "id": i,
                "question": ques,
                "options": options,
                "final_answer": final_ans,
            }
        
        samples.append(processed_sample)
        cur_options = processed_sample["options"] if "options" in processed_sample else None
        cur_prompt = build_prompt(processed_sample["question"], cur_options, args.dataset)
        if len(cur_prompt) < 10000:
            chat_messages.append([{"role": "user", "content": cur_prompt}])

    print("Starting batch inference with vLLM...")
    
    # Perform batch inference using vLLM chat method
    # vLLM automatically handles batching and optimization
    outputs = llm.chat(
        messages=chat_messages,
        sampling_params=sampling_params,
        use_tqdm=True  # Show progress bar
    )

    # Process results
    res = []
    for i, (sample, output) in enumerate(zip(samples, outputs)):
        # Extract the generated text from the output
        generated_text = output.outputs[0].text
        
        result = sample.copy()
        result['prediction'] = generated_text
        res.append(result)

    # Save results
    model_name = os.path.basename(os.path.normpath(args.model_path))
    output_file = f"{model_name}_{args.dataset}_results.json"
    
    # Sort results by id to maintain order
    res = sorted(res, key=lambda x: int(x['id']))
    
    with open(output_file, 'w') as f:
        json.dump(res, f, indent=4)
    
    print(f"Results saved to {output_file}")


if __name__ == '__main__':
    predict(args)