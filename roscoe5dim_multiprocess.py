import os
import json
import argparse
from tqdm import tqdm
from openai import OpenAI
import concurrent.futures
import re
from utils import teacher_eval_prompt, build_prompt

parser = argparse.ArgumentParser(description='Get socreval scores from deepseek',
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--api', help='deepseek api')
parser.add_argument('--file_path', default='Meta-Llama-3-8B-Instruct_gsm8k_results.json', help='path to the file')
parser.add_argument('--dataset', default='gsm8k', choices=['gsm8k', 'math', 'gpqa', 'gsm8k-hard'], help="which dataset")
parser.add_argument('--batch_size', '-bs', default=32, type=int, help="batch size")
parser.add_argument('--temperature', default=0)
parser.add_argument('--max_tokens', default=2048)
args = parser.parse_args()

deepseek = OpenAI(api_key=args.api, base_url="https://api.deepseek.com")


def fix_backslashes(s):
    # Only escape backslashes not part of a valid JSON escape
    return re.sub(r'\\(?![\"\\/bfnrtu])', r'\\\\', s)

def llm_roscoe5dim(prompt):
    question, generated_response = prompt[0], prompt[1]
    max_num_per_question = 2
    res = {}
    for dim in ['semantic_consistency', 'logicality', 'informativeness', 'fluency', 'factuality']:
        res[dim] = {"score": None, "justification": None}
        combined_prompt = teacher_eval_prompt(question, generated_response, args.dataset, dim)
        for i in range(max_num_per_question):
            try:
                response = deepseek.chat.completions.create(
                    model="deepseek-chat",
                    temperature=args.temperature,
                    messages=[{"role": "user", "content": combined_prompt}],
                    response_format={'type': 'json_object'}
                )
                try:
                    ans = json.loads(response.choices[0].message.content)
                except json.JSONDecodeError as e:   # sometimes deepseek v3 doesn't return JSON
                    print("JSONDecodeError, trying to fix:", e)
                    fixed = fix_backslashes(response.choices[0].message.content)
                    try:
                        ans = json.loads(fixed)
                    except Exception as e2:
                        # print("Still invalid after fix:", e2)
                        # Return the raw string if all else fails
                        ans = response.choices[0].message.content
                if len(ans) > 0:   # already get the ans
                    res[dim] = ans
                    break
            except BaseException as E:
                print(E)
                continue
            
    return res


def collect_correct_rationale():
    # dataset_name = args.file_path.split('_')[1]
    output_file = args.file_path.replace('.json', '_roscoe5dim.jsonl')
    
    # check any already processed samples
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    if 'id' in sample:
                        processed_ids.add(sample['id'])
                except json.JSONDecodeError:
                    print(f"Skipping malformed line in {output_file}")
                    continue

    with open(args.file_path, 'r') as f:
        data = json.load(f)
        # data = data[:2000]    # the first 2000 samples

    unprocessed_samples = [sample for sample in data if sample.get('id') not in processed_ids]

    if not unprocessed_samples:
        print("All samples have been processed.")
        return

    prompts_to_process = []
    if 'gsm8k' in args.dataset or args.dataset == 'math':
        for sample in unprocessed_samples:
            prompts_to_process.append((sample['question'], sample['prediction']))
    elif args.dataset == 'gpqa':
        for sample in unprocessed_samples:
            question = build_prompt(sample["question"], sample['options'], 'gpqa')
            prompts_to_process.append((question, sample['prediction']))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.batch_size) as executor:
        # The results will be in the same order as the prompts
        socreval_results = list(tqdm(executor.map(llm_roscoe5dim, prompts_to_process), total=len(prompts_to_process)))

    with open(output_file, "a") as f:
        for i, sample in enumerate(unprocessed_samples):
            socreval_dict = socreval_results[i]
            sample['roscoe5dim_score'] = socreval_dict
            f.write(json.dumps(sample) + "\n")
            f.flush()

if __name__ == "__main__":
    collect_correct_rationale()