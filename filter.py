import argparse
import json
import pickle
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from utils import probing_eval_prompt
import pickle


parser = argparse.ArgumentParser(description="Filter data using a trained classifier.")
parser.add_argument('--clf_root', type=str, required=True, help='Path to classifier pickle root')
parser.add_argument('--batch_size', '-bs', default=16, type=int, help="batch size")
parser.add_argument('--file_path', type=str, required=True, help='Path to results JSON file')
# parser.add_argument('--output_path', type=str, required=True, help='Path to save filtered results')
parser.add_argument('--top_percent', type=float, default=1.0, help='Fraction of top data to keep (e.g., 0.2 for 20%)')
parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-1.7B", help='Huggingface model name for feature extraction')
args = parser.parse_args()
# python filter.py --clf_root "gsm8k_binary_clfs" --file_path "Meta-Llama-3-8B-Instruct_math_results.json"


# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"

def load_model_pickle(clf_path="model.pkl"):
    """
    Load back the model and metadata from pickle.
    """
    with open(clf_path, "rb") as f:
        package = pickle.load(f)
    clf = package["model"]
    metadata = package["metadata"]
    # print("Loaded metadata:", metadata)
    return clf, metadata


def extract_batch_reps(prompt_list, pool, tokenizer, model):
    inputs = tokenizer(
        prompt_list,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    # Send inputs to the same device as the model embeddings
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(
            **inputs, 
            output_hidden_states=True, 
            output_attentions=True
        )
    
    hidden = out.hidden_states
    attentions = out.attentions
    reps = {}
    attn_entropy = {}
    
    for layer_idx, h in enumerate(hidden[1:]):
        if pool == "mean":
            res_pooled = h.mean(dim=1).cpu().detach().numpy()
        elif pool == "last":
            seq_lens = (inputs['attention_mask'].sum(dim=1) - 1)
            res_pooled = torch.stack([h[i, seq_lens[i], :] for i in range(h.size(0))]).cpu().detach().numpy()
        elif pool == "min":
            res_pooled = h.min(dim=1)[0].cpu().detach().numpy()
        elif pool == "max":
            res_pooled = h.max(dim=1)[0].cpu().detach().numpy()
        reps[layer_idx] = {pool: res_pooled}

        # Attention entropy
        A = attentions[layer_idx]
        mask = A > 0
        log_A = torch.where(mask, torch.log(A), torch.zeros_like(A))
        ent = - (A * log_A).sum(dim=-1).mean(dim=-1)
        attn_entropy[layer_idx] = ent.cpu().detach().numpy()
    
    return reps, attn_entropy


def extract_features(reps, attn_entropy, layers, pool, use_attn=False):
    features = []
    for l in layers:
        h = reps[l][pool]  # (B, H)
        if use_attn:
            attn = attn_entropy[l]  # (B, n_heads)
            layer_feat = np.hstack([h, attn])  # (B, target_dim_per_layer + n_heads)
        else:
            layer_feat = h
        features.append(layer_feat)
    features_final = np.hstack(features)  # (B, ...)
    return features_final


def main():
    # Load model/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        output_hidden_states=True,
        output_attentions=True,
        dtype=torch.float16 if torch.cuda.is_available() or torch.backends.mps.is_available() else torch.float32,
        device_map="auto",
        attn_implementation="eager",
        # cache_dir='local_models/',
    )
    model.eval()

    # Load results
    with open(args.file_path, 'r') as f:
        samples = json.load(f)
        # samples = samples[:100]

    # Score each sample
    res = []
    for i in tqdm(range(0, len(samples), args.batch_size)):
        batch = samples[i:i+args.batch_size]
        tmp_res = batch.copy()
        for dim in ['semantic_consistency', 'logicality', 'informativeness', 'fluency', 'factuality']:
            # Load classifier
            clf_path = args.clf_root+f'/{dim}.pkl'       # 'gsm8k_clfs/semantic_consistency.pkl'
            clf, metadata = load_model_pickle(clf_path)
            pool = metadata["pool"]
            use_attn = metadata["use_attn"]
            layers = metadata["layers"]
            classification = metadata["classification"]

            prompts = [probing_eval_prompt(sample['question'], sample['prediction'], dim) for sample in batch]
            reps, attn_entropy = extract_batch_reps(prompts, pool, tokenizer, model)
            features = extract_features(reps, attn_entropy, layers, pool, use_attn)
            del reps, attn_entropy
            torch.cuda.empty_cache()
            batch_scores = clf.predict(features)
            for j, sample in enumerate(batch):
                score = batch_scores[j]
                if hasattr(score, "item"):
                    tmp_res[j][dim] = score.item()

        for s in tmp_res:   # total scores
            s['total_score'] = sum([s[dim] for dim in ['semantic_consistency', 'logicality', 'informativeness', 'fluency', 'factuality']])
        res.extend(tmp_res)
    # Sort and keep top X%
    if args.top_percent != 1.0:
        res.sort(reverse=True, key=lambda x: x['total_score'])
        n_keep = int(len(res) * args.top_percent)
        res = res[:n_keep]

    # Save
    model_id = args.model_name.split('/')[1]
    output_path = args.file_path.replace('.json', f'_{model_id}_{classification}_filtered_ood.json')
    with open(output_path, 'w') as f:
        json.dump(res, f, indent=4)
    print(f"Saved {len(res)} samples to {output_path}")


if __name__ == "__main__":
    main()
