import json
import re
from datasets import load_dataset, concatenate_datasets
import glob
import os
from math_verify import LatexExtractionConfig, ExprExtractionConfig, parse, verify
import random
from collections import defaultdict

def build_prompt(question, options=None, dataset='gsm8k'):
    if 'gsm8k' in dataset:
        return  question + "\nLet's think step by step."
    elif dataset == 'math' or dataset == 'math500':
        return question + "\nLet's think step by step and put the final answer in \\boxed{}."
    elif dataset == 'gpqa':
        random.shuffle(options)
        return question + " Put the final answer in \\boxed{}.\n" + '\n'.join(options)
    elif dataset == 'alpaca2':
        return question

def my_load_dataset(dataset='gsm8k'):
    train_dataset, test_dataset = [], []
    if dataset == 'gsm8k':
        train_dataset = load_dataset('gsm8k', 'main', split='train', cache_dir='local_dataset/')
        test_dataset = load_dataset('gsm8k', 'main', split='test', cache_dir='local_dataset/')
    elif dataset == 'gsm8k-hard':
        dataset = load_dataset('reasoning-machines/gsm-hard', split='train', cache_dir='local_dataset/')
        train_dataset = dataset
        test_dataset = None
    elif dataset == 'math':
        train_dir = 'local_dataset/MATH/train'
        test_dir = 'local_dataset/MATH/test'
        train_files = glob.glob(os.path.join(train_dir, '**', '*.json'), recursive=True)
        for file in train_files:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                train_dataset.append(data)
        test_files = glob.glob(os.path.join(test_dir, '**', '*.json'), recursive=True)
        for file in test_files:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                test_dataset.append(data)
    elif dataset == 'math500':
        train_dir = 'local_dataset/MATH/train'
        train_files = glob.glob(os.path.join(train_dir, '**', '*.json'), recursive=True)
        for file in train_files:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                train_dataset.append(data)
        test_dataset = load_dataset("HuggingFaceH4/MATH-500", split='test', cache_dir='local_dataset/')
    elif dataset == 'commonsenseQA':
        train_dataset = load_dataset("tau/commonsense_qa", split='train', cache_dir='local_dataset/')
        test_dataset = load_dataset("tau/commonsense_qa", split='test', cache_dir='local_dataset/')
    elif dataset == 'gpqa':
        access_token = "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"
        gpqa_main = load_dataset("Idavidrein/gpqa", 'gpqa_main', split='train', cache_dir='local_dataset/', token=access_token)
        gpqa_extended = load_dataset("Idavidrein/gpqa", 'gpqa_extended', split='train', cache_dir='local_dataset/', token=access_token)
        train_dataset = concatenate_datasets([gpqa_main, gpqa_extended])
        test_dataset = load_dataset("Idavidrein/gpqa", 'gpqa_diamond',  split='train', cache_dir='local_dataset/', token=access_token)
    elif dataset == 'alpaca2':
        # dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval_gpt4_baseline", split='eval', cache_dir='local_dataset/')
        dataset = load_dataset("json", data_files={"eval": "local_dataset/alpaca_eval/alpaca_eval_gpt4_baseline.json"}, split="eval")
        train_dataset = dataset
        test_dataset = dataset
    return train_dataset, test_dataset
        

def split_sample(sample, dataset='gsm8k'):
    '''
    split question, answer and rationale for dataset GSM8K
    :param sample: dict{}
    :return: ques, ration, ans
    '''
    if dataset == 'gsm8k':
        ques = sample['question'].strip()
        ration = sample['answer'].strip()
        final_ans = sample['answer'].split('####')[1].strip()
    elif dataset == 'gsm8k-hard':
        ques = sample['input'].strip()
        ration = sample['code'].strip()
        final_ans = sample['target']
    elif dataset == 'math':
        ques = sample['problem'].strip()
        ration = sample['solution'].strip()
        # math has no gold final answer
        final_ans = sample['solution'].strip()
    elif dataset == 'math500':
        ques = sample['problem'].strip()
        ration = sample['solution'].strip()
        final_ans = sample['solution'].strip()   
        # final_ans = sample['answer'].strip()   # math500's provided final answer can't help evaluation
    elif dataset == 'commonsenseQA':
        ques = sample['question'].strip()
        options = sample['choices']
        final_ans = str(sample['answerKey']).strip()
        return ques, options, final_ans
    elif dataset == 'gpqa':
        ques = sample['Question'].strip()
        options = [sample['Correct Answer'], sample['Incorrect Answer 1'], sample['Incorrect Answer 2'], sample['Incorrect Answer 3']]
        final_ans = str(sample['Correct Answer']).strip()
        return ques, options, final_ans
    elif dataset == 'alpaca2':
        ques = sample['instruction'].strip()
        ration = sample['output'].strip()
        final_ans = sample['output'].strip()

    return ques, ration, final_ans


def verify_preds(pred, gold_answer, dataset='gsm8k'):
    if 'gsm8k' in dataset:
        gold_parsed = parse(gold_answer, extraction_config=[ExprExtractionConfig()],
                            fallback_mode="first_match", extraction_mode="any_match")
        answer_parsed = parse(pred, extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()],
                              fallback_mode="first_match", extraction_mode="any_match")
        return verify(answer_parsed, gold_parsed)
    elif 'math' in dataset:
        gold_parsed = parse(gold_answer, extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()],
                            fallback_mode="first_match", extraction_mode="any_match")
        answer_parsed = parse(pred, extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()],
                              fallback_mode="first_match", extraction_mode="any_match")
        return verify(answer_parsed, gold_parsed)
    elif dataset == 'gpqa':
        match = re.search(r'\\boxed\{([^}]+)\}', pred)      # first find wrapped by box
        if match:
            if gold_answer.lower() in match.group(1).strip().lower():
                return True
        else:       # then find "the correct answer is xxx"
            answer_patterns = [
            r"the correct answer is[:\s]*\n*(.+?)(?:\n|$)",
            r"the answer is[:\s]*\n*(.+?)(?:\n|$)",
            r"answer:[:\s]*\n*(.+?)(?:\n|$)",
            r"final answer[:\s]*\n*(.+?)(?:\n|$)",
            ]
            for pattern in answer_patterns:
                match = re.search(pattern, pred, re.IGNORECASE | re.DOTALL)
                if match:
                    extracted_answer = match.group(1).strip().lower()
                    # Clean extracted answer (remove punctuation, extra whitespace)
                    # extracted_answer = re.sub(r'[^\w\s]', '', extracted_answer).strip()
                    if gold_answer.lower() in extracted_answer:
                        return True

        if gold_answer in pred:
            return True

    return False


eval_prompts = defaultdict(dict)
eval_prompts['semantic_consistency']['instruction'] = "You are a judge that scores Semantic Consistency of a step-by-step rationale for a reasoning problem. Definition: Semantic Consistency = the solution steps and final answer must stay faithful to the problem facts (no inventing events, no dropping givens, no added unstated assumptions). A step is inconsistent if it contradicts the problem, introduces facts not present in the problem, or ignores givens. Scoring: integer 1–5."
eval_prompts['semantic_consistency']['criterion'] = "5 — Every step and the final answer strictly follow the problem facts; no unstated assumptions or contradictions.\n4 — Steps mostly follow the givens; one small unstated assumption that doesn’t change the outcome.\n3 — Minor omission or one mild contradiction that slightly weakens trust in the chain.\n2 — Noticeable contradictions or added facts that affect the reasoning or outcome.\n1 — Steps contradict the problem or introduce major unstated facts; answer not grounded in the problem."
eval_prompts['logicality']['instruction'] = "You are a judge that scores Logicality of a step-by-step rationale for a reasoning problem. Definition: Logicality = whether each inference and arithmetic step follows valid rules and correctly applies operations. Penalize invalid deductions or misapplied reasoning. Scoring: integer 1–5."
eval_prompts['logicality']['criterion'] = "5 — All inferences and arithmetic are valid; each step follows logically from prior steps.\n4 — One small inference leap or minor justification gap, but overall logic holds.\n3 — Some steps are questionable or contain small mistakes, yet parts of reasoning remain sound.\n2 — Multiple invalid inferences or arithmetic errors that materially affect the solution.\n1 — Fundamentally illogical or nonsensical reasoning (steps do not connect)."
eval_prompts['informativeness']['instruction'] = "You are a judge that scores Informativeness of a step-by-step rationale for a reasoning problem. Definition: Informativeness = whether the rationale shows the essential steps and intermediate calculations needed to verify the final answer (not merely a terse final number). Reward verifiable, stepwise derivations. Scoring: integer 1–5."
eval_prompts['informativeness']['criterion'] = "5 — Full, verifiable step-by-step derivation; anyone can re-check the answer from the steps.\n4 — Most essential steps shown; one or two minor gaps but overall verifiable.\n3 — Key steps present but several derivations omitted; partially verifiable.\n2 — Very terse; crucial intermediate calculations missing so verification is hard.\n1 — Only an answer or irrelevant details; no usable derivation."
eval_prompts['fluency']['instruction'] = "You are a judge that scores Fluency (readability and clarity) of a step-by-step rationale for a reasoning problem. Definition: Fluency = the text is grammatical, clear, and easy to follow. Judge punctuation, sentence flow, readable notation and presentation. Fluency does NOT evaluate correctness. Scoring: integer 1–5."
eval_prompts['fluency']['criterion'] = "5 — Clear, grammatical, well-punctuated, and easy to follow; notation readable.\n4 — Mostly clear with small phrasing or punctuation issues.\n3 — Understandable but awkward phrasing, punctuation, or notation that slows comprehension.\n2 — Hard to follow; many grammatical issues or poor notation.\n1 — Unreadable or incoherent language."
eval_prompts['factuality']['instruction'] = "You are a judge that scores Factuality of a step-by-step rationale for a reasoning problem. Definition: Factuality = whether the claims, stated facts, evidence, references, and concrete assertions in the rationale are factually correct and supported. Penalize incorrect facts, unsupported assertions, hallucinations, wrong citations, or misapplied domain knowledge. Scoring: integer 1–5."
eval_prompts['factuality']['criterion'] = "5 — All factual claims and referenced facts are correct and well-supported by the rationale or common knowledge. No hallucinations.\n4 — Minor factual imprecision (typo, small numeric slip, or weakly-supported minor claim) that does not change the conclusion.\n3 — Some factual errors or unsupported claims exist; the final answer may still be salvageable with corrections or additional evidence.\n2 — Multiple factual mistakes or serious unsupported assertions that materially affect confidence in the conclusion.\n1 — Major factual errors, clear hallucinations, or fundamentally wrong domain knowledge that render the answer incorrect."


def teacher_eval_prompt(question, generated_response, dataset='gsm8k', dimension="semantic_consistency"):
    evaluation_prompt = "Now evaluate the Question and Generated response above based on the instruction and the format of the example representation. Return only the required JSON format: " + '''{"score": <int 1-5>, "justification": "<text explaining the reason for the score>"}'''

    if dataset=='math' or 'gsm8k' in dataset:
        example_question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
        if dimension == 'semantic_consistency':
            example_generated_response = "She sold 48 in April. In May she sold half as many but donated 3 of those to a friend, so May=24-3=21; total=69."
            example_representation = json.dumps({"score": 2, "justification": "Introduces an unstated donation (3) not in the problem; solution no longer strictly follows the given facts."})
        elif dimension == 'logicality':
            example_generated_response = "April = 48. 'Half as many in May' interpreted as add half of April to April, May = 48 + 48/2 = 72. Total = 48 + 72 = 120."
            example_representation = json.dumps({"score": 1, "justification": "Misinterprets 'half as many' and performs an invalid reasoning step, producing an illogical total."})
        elif dimension == 'informativeness':
            example_generated_response = "After thinking, the answer is 72."
            example_representation = json.dumps({"score": 1, "justification": "Only final number given; no intermediate steps or explanation to verify the computation."})
        elif dimension == 'fluency':
            example_generated_response = "sold 48 april then half may 24 total 72"
            example_representation = json.dumps({"score": 2, "justification": "Ungrammatical and terse; the meaning is recoverable but phrasing and punctuation are poor."})
        elif dimension == 'factuality':
            example_generated_response = "She sold 48/2 = 20 clips in May. Natalia sold 48+20 = 68 clips altogether. #### 68"
            example_representation = json.dumps({"score": 1, "justification": "Arithmetic is incorrect (48/2 ≠ 20), so factual claims and final numeric result are wrong."})
    elif dataset=='gpqa':
        example_question = "A large gene has dozens of exons, of which the central ones code for folded triple helical repeats. Which structure below is not involved in the proposed therapy?\nOptions:\npolyA tail\nR-loops\nantisense\nlariat"
        if dimension == 'semantic_consistency':
            example_generated_response = "The therapy uses a Morpholino to skip exons. Therefore, R-loops are not involved."
            example_representation = json.dumps({"score": 5, "justification": "All steps in the rationale are strictly consistent with the problem description; no invented facts or omitted givens."})
        elif dimension == 'logicality':
            example_generated_response = "Morpholino blocks exon, so lariat must be involved, polyA tail not used, R-loops skipped, antisense binds randomly."
            example_representation = json.dumps({"score": 2, "justification": "Some reasoning steps are invalid or contradictory; e.g., 'antisense binds randomly' is unsupported and illogical in context."})
        elif dimension == 'informativeness':
            example_generated_response = "Answer: R-loops"
            example_representation = json.dumps({"score": 1, "justification": "Only the final answer is given; no step-by-step rationale to show how the answer was derived."})
        elif dimension == 'fluency':
            example_generated_response = "use morpholino. not involve r-loops"
            example_representation = json.dumps({"score": 2, "justification": "Short, ungrammatical phrasing and poor punctuation make it hard to read, although meaning is recoverable."})
        elif dimension == 'factuality':
            example_generated_response = "The polyA tail is not used; antisense and R-loops both irrelevant; lariat forms irrelevant loops."
            example_representation = json.dumps({"score": 3, "justification": "Some claims are factually imprecise (e.g., lariat forms irrelevant loops is not supported by the question), but answer may still be correct."})
    elif dataset=='alpaca2':
        example_question = "Write a script for a YouTube video exploring the history and cultural significance of jazz."
        if dimension == 'semantic_consistency':
            example_generated_response = "Let's dive into the evolution of heavy metal, from Black Sabbath to modern symphonic metal."
            example_representation = json.dumps({"score": 2, "justification": "Script shifts to heavy metal instead of jazz, so the content no longer follows the requested topic."})
        elif dimension == 'logicality':
            example_generated_response = ("Opening: introduce jazz in the 18th century.\n"
                                          "Next: explain how bebop led to the invention of the saxophone.\n"
                                          "Conclusion: note that jazz disappeared after 1930.")
            example_representation = json.dumps({"score": 1, "justification": "Timeline and causal links are illogical (bebop did not invent the saxophone, jazz did not end in 1930)."})
        elif dimension == 'informativeness':
            example_generated_response = "Jazz matters. Mention Louis Armstrong. End."
            example_representation = json.dumps({"score": 1, "justification": "Provides almost no structure or detail for a video script; lacks steps or sections to follow."})
        elif dimension == 'fluency':
            example_generated_response = "jazz history video script. talk stuff. people like it. done"
            example_representation = json.dumps({"score": 2, "justification": "Fragmented, ungrammatical sentences make the script hard to follow, though intent is vaguely clear."})
        elif dimension == 'factuality':
            example_generated_response = ("Opening: explain that jazz was invented in Paris in 1900 by Elvis Presley.\n"
                                          "Highlight how vinyl records were first created for jazz in 1985.\n"
                                          "State that the saxophone is a percussion instrument central to jazz rhythms.")
            example_representation = json.dumps({"score": 1, "justification": "Contains multiple factual errors about jazz origins, key figures, and instruments."})

    final_prompt = "Instruction:\n" + eval_prompts[dimension]['instruction'] + "\n" + eval_prompts[dimension]['criterion'] \
                    + "\n\nExample question:\n" + example_question \
                    + "\n\nExample generated response:\n" + example_generated_response \
                    + "\n\nExample representation:\n" + example_representation \
                    + "\n\nQuestion:\n" + question \
                    + "\n\nGenerated response:\n" + generated_response \
                    + "\n\n" + evaluation_prompt + "\n"
    return final_prompt

def probing_eval_prompt(question, generated_response, dimension="semantic_consistency"):
    evaluation_prompt = "Now evaluate the Question and Generated response above based on the instruction. Return only the score."

    if dimension == 'semantic_consistency':
        instruction = "You are a judge that scores Semantic Consistency of a step-by-step rationale for a reasoning problem. Definition: Semantic Consistency = the solution steps and final answer must stay faithful to the problem facts (no inventing events, no dropping givens, no added unstated assumptions). A step is inconsistent if it contradicts the problem, introduces facts not present in the problem, or ignores givens. Scoring: integer 1–5."
        criterion = "5 — Every step and the final answer strictly follow the problem facts; no unstated assumptions or contradictions.\n4 — Steps mostly follow the givens; one small unstated assumption that doesn’t change the outcome.\n3 — Minor omission or one mild contradiction that slightly weakens trust in the chain.\n2 — Noticeable contradictions or added facts that affect the reasoning or outcome.\n1 — Steps contradict the problem or introduce major unstated facts; answer not grounded in the problem."
    elif dimension == 'logicality':
        instruction = "You are a judge that scores Logicality of a step-by-step rationale for a reasoning problem. Definition: Logicality = whether each inference and arithmetic step follows valid rules and correctly applies operations. Penalize invalid deductions or misapplied reasoning. Scoring: integer 1–5."
        criterion = "5 — All inferences and arithmetic are valid; each step follows logically from prior steps.\n4 — One small inference leap or minor justification gap, but overall logic holds.\n3 — Some steps are questionable or contain small mistakes, yet parts of reasoning remain sound.\n2 — Multiple invalid inferences or arithmetic errors that materially affect the solution.\n1 — Fundamentally illogical or nonsensical reasoning (steps do not connect)."
    elif dimension == 'informativeness':
        instruction = "You are a judge that scores Informativeness of a step-by-step rationale for a reasoning problem. Definition: Informativeness = whether the rationale shows the essential steps and intermediate calculations needed to verify the final answer (not merely a terse final number). Reward verifiable, stepwise derivations. Scoring: integer 1–5."
        criterion = "5 — Full, verifiable step-by-step derivation; anyone can re-check the answer from the steps.\n4 — Most essential steps shown; one or two minor gaps but overall verifiable.\n3 — Key steps present but several derivations omitted; partially verifiable.\n2 — Very terse; crucial intermediate calculations missing so verification is hard.\n1 — Only an answer or irrelevant details; no usable derivation."
    elif dimension == 'fluency':
        instruction = "You are a judge that scores Fluency (readability and clarity) of a step-by-step rationale for a reasoning problem. Definition: Fluency = the text is grammatical, clear, and easy to follow. Judge punctuation, sentence flow, readable notation and presentation. Fluency does NOT evaluate correctness. Scoring: integer 1–5."
        criterion = "5 — Clear, grammatical, well-punctuated, and easy to follow; notation readable.\n4 — Mostly clear with small phrasing or punctuation issues.\n3 — Understandable but awkward phrasing, punctuation, or notation that slows comprehension.\n2 — Hard to follow; many grammatical issues or poor notation.\n1 — Unreadable or incoherent language."
    elif dimension == 'factuality':
        instruction = "You are a judge that scores Factuality of a step-by-step rationale for a reasoning problem. Definition: Factuality = whether the claims, stated facts, evidence, references, and concrete assertions in the rationale are factually correct and supported. Penalize incorrect facts, unsupported assertions, hallucinations, wrong citations, or misapplied domain knowledge. Scoring: integer 1–5."
        criterion = "5 — All factual claims and referenced facts are correct and well-supported by the rationale or common knowledge. No hallucinations.\n4 — Minor factual imprecision (typo, small numeric slip, or weakly-supported minor claim) that does not change the conclusion.\n3 — Some factual errors or unsupported claims exist; the final answer may still be salvageable with corrections or additional evidence.\n2 — Multiple factual mistakes or serious unsupported assertions that materially affect confidence in the conclusion.\n1 — Major factual errors, clear hallucinations, or fundamentally wrong domain knowledge that render the answer incorrect."

    final_prompt = "Instruction:\n" + instruction + "\n" + criterion \
                    + "\n\nQuestion:\n" + question \
                    + "\n\nGenerated response:\n" + generated_response \
                    + "\n\n" + evaluation_prompt + "\n"
    return final_prompt
    

# question="Tom has a red marble, a green marble, a blue marble, and three identical yellow marbles. How many different groups of two marbles can Tom choose?"
# prediction= "To find the total number of different groups of two marbles Tom can choose, we need to consider the different combinations of marbles.\n\n1. **Two distinct marbles**: Tom can choose two marbles of different colors. The possible combinations are:\n   - Red and green\n   - Red and blue\n   - Red and yellow\n   - Green and blue\n   - Green and yellow\n   - Blue and yellow\n   There are 6 combinations of distinct marbles.\n\n2. **Two identical marbles**: Tom can choose two yellow marbles. Since the yellow marbles are identical, there is only 1 combination of identical marbles.\n\nTo find the total number of different groups of two marbles, we add the number of combinations of distinct marbles and the number of combinations of identical marbles:\n6 (distinct marbles) + 1 (identical marbles) = 7\n\nSo, the total number of different groups of two marbles Tom can choose is \\boxed{7}."

# print(probing_eval_prompt(question, prediction, "factuality"))