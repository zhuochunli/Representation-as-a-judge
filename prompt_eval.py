from tqdm import tqdm
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.stats import somersd
from vllm import LLM, SamplingParams
import json
import argparse
import re
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"


parser = argparse.ArgumentParser(description="directly prompt evaluation",
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--model_path', '-m', default='Qwen/Qwen3-1.7B', help='Base model path')
parser.add_argument('--file_path', default='Meta-Llama-3-8B-Instruct_math_roscoe5dim_probing.json', help="probing file path")
parser.add_argument('--max_new_tokens', type=int, default=2048,
                    help='Maximum new tokens for generation')
parser.add_argument('--tensor_parallel_size', type=int, default=2,
                    help='Number of GPUs to use for tensor parallelism')
parser.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                    help='GPU memory utilization ratio')
args = parser.parse_args()


# Initialize vLLM model
llm = LLM(
    model=args.model_path,    # meta-llama/Llama-3.2-1B-Instruct
    tensor_parallel_size=args.tensor_parallel_size,  # Adjust based on your GPU setup
    gpu_memory_utilization=args.gpu_memory_utilization,  # Use 90% of GPU memory
    max_model_len=32768,  # Match your original max_new_tokens context
    trust_remote_code=True,  # Required for some models
    # download_dir="/jet/home/zli26/tmp_ondemand_ocean_cis240025p_symlink/zli26/evaluation/local_models",
    dtype="float16",
)

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0,  # Deterministic generation
    max_tokens=args.max_new_tokens,  # Reasonable limit for evaluation responses
)


# For socreval
class Sample:
    def __init__(self, prompt, label):
        self.prompt = prompt
        self.label = label


def prompt_binary(dim, file_path):
    dataset = []
    with open(file_path, 'r') as f:
        all_data = json.load(f)
        for sample in all_data[dim]:
            dataset.append(Sample(prompt=sample['eval_prompt'], label=1 if sample['score']>3 else 0))
    train, test = train_test_split(dataset, test_size=0.20, stratify=[s.label for s in dataset], random_state=731)

    preds, ground_truth = [], []
    chat_messages = []
    for sample in test:
        # The eval_prompt should already be formatted correctly
        messages = [{"role": "user", "content": sample.prompt}]
        chat_messages.append(messages)
        ground_truth.append(sample.label)

    # from collections import Counter
    # print(Counter(ground_truth))
        
    # Generate responses using chat interface
    outputs = llm.chat(
    messages=chat_messages,
    sampling_params=sampling_params,
    use_tqdm=True,
    )

    # Process results
    res = []
    for i, output in enumerate(outputs):
        # Extract the generated text from the output
        generated_text = output.outputs[0].text
        try: 
            content = json.loads(generated_text)
            preds.append(content['overall quality'])   # if output is json format
        except:
            score_match = re.search(r'"overall quality":\s*(\d)', generated_text)
            if score_match:
                if int(score_match.group(1))>3:
                    preds.append(1)
                else:
                    preds.append(0)
            else:
                match = re.search(r"\b([1-5])\b", generated_text)  # if output is score
                if match:
                    if int(match.group(1))>3:
                        preds.append(1)
                    else:
                        preds.append(0)
                else:
                    preds.append(0)
        res.append(generated_text)

    print(f"For dimension {dim}, classification report:\n", classification_report(y_true=ground_truth, y_pred=preds, digits=4))
    # print("Socreval MATH Somer's-D correlation:\n", somersd(preds, ground_truth))

    # with open('qwen1.7b_math_socreval_eval.json', 'w') as f:
    #     json.dump(res, f, indent=4)

def prompt_multi(dim, file_path):
    dataset = []
    with open(file_path, 'r') as f:
        all_data = json.load(f)
        for sample in all_data[dim]:
            dataset.append(Sample(prompt=sample['eval_prompt'], label=sample['score']))
    train, test = train_test_split(dataset, test_size=0.20, stratify=[s.label for s in dataset], random_state=731)

    preds, ground_truth = [], []
    chat_messages = []
    for sample in test:
        # The eval_prompt should already be formatted correctly
        messages = [{"role": "user", "content": sample.prompt}]
        chat_messages.append(messages)
        ground_truth.append(sample.label)

    # from collections import Counter
    # print(Counter(ground_truth))
        
    # Generate responses using chat interface
    outputs = llm.chat(
    messages=chat_messages,
    sampling_params=sampling_params,
    use_tqdm=True,
    )

    # Process results
    res = []
    for i, output in enumerate(outputs):
        # Extract the generated text from the output
        generated_text = output.outputs[0].text
        try: 
            content = json.loads(generated_text)
            preds.append(content['overall quality'])
        except:
            score_match = re.search(r'"overall quality":\s*(\d)', generated_text)     # if output is json format
            if score_match:
                preds.append(int(score_match.group(1)))
            else:
                match = re.search(r"\b([1-5])\b", generated_text)    # if output is score
                if match:
                    preds.append(int(match.group(1)))
                else:
                    preds.append(1)
        res.append(generated_text)

    print(f"For dimension {dim}, classification report:\n", classification_report(y_true=ground_truth, y_pred=preds, digits=4))
    # print("Socreval MATH Somer's-D correlation:\n", somersd(preds, ground_truth))

    # with open('qwen1.7b_math_socreval_eval.json', 'w') as f:
    #     json.dump(res, f, indent=4)

for dim in ['semantic_consistency', 'logicality', 'informativeness', 'fluency', 'factuality']:
    prompt_multi(dim, args.file_path)
    print("--------------------------------")
    prompt_binary(dim, args.file_path)