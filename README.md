# <u>IN</u>ternal <u>S</u>ignal <u>P</u>robing and <u>E</u>valua<u>T</u>ion <u>O</u>f <u>R</u>epresentations (INSPECTOR)

Code for the paper: "[Rethinking LLM-as-a-Judge: Representation-as-a-Judge with Small Language Models via Semantic Capacity Asymmetry](https://openreview.net/pdf?id=VAISvCsrvG)", accepted by ICLR 2026.
![Overview of our FAIR method.](assets/inspector_v2.png)

## Before You Start
This repository aims to provide a plug-and-play reliable evaluators for reasoning questions with predictions (reference-free), and to reproduce our paper's work. Since our work is based on public models, datasets and HuggingFace, it is **easy to scale and adapt**. Feel free to scale out for your own tasks!

- Baselines: RoBERTa
- Small Language models: Qwen3-0.6B, Qwen3-1.7B, Llama-3.2-1B-Instruct, Llama-3.1-8B-Instruct
- Large Language models: DeepSeek-V3 
- Datasets: [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [MATH](https://github.com/hendrycks/math), [GPQA](https://huggingface.co/datasets/Idavidrein/gpqa), [AlpacaEval2](https://huggingface.co/datasets/tatsu-lab/alpaca_eval/blob/main/alpaca_eval_annotations_alpaca_eval_gpt4.json)  

Directory explanation:
- `/local_dataset`: The `cache_dir` of `load_dataset()` for all downloaded datasets.
- `/{gsm8k/math/gpqa}_{multi/binary}_clfs`: The ready to use trained multiclass or binary classifier on different datasets, based on optimial representations for best evaluation performance. They are based on the LLM Qwen3-1.7B and classifiers in scikit-learn.
- `/checkpoints`: The `output_dir` for fine-tuned models as baselines.

## Plug-and-Play
### 1. Requirements
```
pip install -r requirements.txt
```

### 2. Evaluate your question and prediction pairs on reasoning tasks
As we show above, we already uploaded trained classifiers for the datasets used in our paper. However, you can also use them to evaluate similar tasks, such as evaluate SVAMP tasks using our trained GSM8K/MATH classifiers.

The classifiers are based on Qwen3-1.7B and scikit-learn, please make sure your resource is compatible with the inference. It supports evaluate the reasoning pairs across 5 aspects: 

Semantic Consistency, Logicality, Informativeness, Fluency, Factuality

For each pair, multiclass classifier will rate 1-5 score for each aspect, and binary classifier will rate 0/1 (low quality/high quality) score for each aspect.

The evaluated results will add scores of these 5 aspects in the dict, and key `"total_score"` to sum up all scores. You can also specify `--top_percent` (defualt=1) to filter data based on the top perecent quality data based on the `"total_score"`.

We support 2 input formats (but the dict must contain keys `"question"` and `"prediction"`):

-  The input format is a list containing question and prediction pairs:
```
from quick_eval import evaluate_samples

# Your input list
samples = [
    {"question": "What is 2+2?", "prediction": "4"},
    {"question": "What is 3+3?", "prediction": "6"}
]

# Evaluate and get results
results = evaluate_samples(samples, clf_root='gsm8k_multi_clfs', batch_size=16, top_percent=1.0)

# Returns: [
#     {"question": "...", "prediction": "...", 
#      "semantic_consistency": 5, "logicality": 4, 
#      "informativeness": 1, "fluency": 1, "factuality": 5, 
#      "total_score": 16},
#     ...
# ]
```

- The input format is a json file containing similar pairs as above, e.g. `Meta-Llama-3-8B-Instruct_gsm8k_results.json`. This will produce a evaluated json file, e.g. `Meta-Llama-3-8B-Instruct_gsm8k_results_Qwen3-1.7B_multi_evaled.json`
```
python quick_eval.py --clf_root 'gsm8k_multi_clfs' --batch_size 16 --file_path 'Meta-Llama-3-8B-Instruct_math_results.json' --top_percent 1.0
```



## Citation
If you find this work helpful, we would appreciate it if you could cite it!
```bibtex
@inproceedings{
anonymous2026rethinking,
title={Rethinking {LLM}-as-a-Judge: Representation-as-a-Judge with Small Language Models via Semantic Capacity Asymmetry},
author={Anonymous},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=VAISvCsrvG}
}
```

