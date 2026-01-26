import argparse
import json
import pickle
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from utils import probing_eval_prompt


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


def evaluate_samples(samples, clf_root='gsm8k_multi_clfs', batch_size=16, 
                     model_name="Qwen/Qwen3-1.7B", top_percent=1.0):
    """
    Evaluate a list of samples and return evaluated results.
    
    Args:
        samples: List of dictionaries, each must have 'question' and 'prediction' keys
        clf_root: Path to classifier pickle root directory
        batch_size: Batch size for processing
        model_name: Huggingface model name for feature extraction
        top_percent: Fraction of top data to keep (1.0 = keep all)
    
    Returns:
        List of dictionaries with original fields plus evaluation scores:
        - semantic_consistency
        - logicality
        - informativeness
        - fluency
        - factuality
        - total_score
    """
    # Load model/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_hidden_states=True,
        output_attentions=True,
        dtype=torch.float16 if torch.cuda.is_available() or torch.backends.mps.is_available() else torch.float32,
        device_map="auto",
        attn_implementation="eager",
        # cache_dir='local_models/',
    )
    model.eval()

    # Score each sample
    res = []
    classification = None  # Will be set from metadata
    for i in tqdm(range(0, len(samples), batch_size)):
        batch = samples[i:i+batch_size]
        tmp_res = batch.copy()
        for dim in ['semantic_consistency', 'logicality', 'informativeness', 'fluency', 'factuality']:
            # Load classifier
            clf_path = clf_root+f'/{dim}.pkl'       # 'gsm8k_multi_clfs/semantic_consistency.pkl'
            clf, metadata = load_model_pickle(clf_path)
            pool = metadata["pool"]
            use_attn = metadata["use_attn"]
            layers = metadata["layers"]
            if classification is None:
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
                else:
                    tmp_res[j][dim] = int(score)

        for s in tmp_res:   # total scores
            s['total_score'] = sum([s[dim] for dim in ['semantic_consistency', 'logicality', 'informativeness', 'fluency', 'factuality']])
        res.extend(tmp_res)
    
    # Sort and keep top X%
    if top_percent != 1.0:
        res.sort(reverse=True, key=lambda x: x['total_score'])
        n_keep = int(len(res) * top_percent)
        res = res[:n_keep]
    
    return res


def main():
    parser = argparse.ArgumentParser(description="Evaluate samples using trained classifiers.")
    parser.add_argument('--clf_root', type=str, default='gsm8k_multi_clfs', help='Path to classifier pickle root')
    parser.add_argument('--batch_size', '-bs', default=16, type=int, help="batch size")
    parser.add_argument('--file_path', type=str, default=None, help='Path to results JSON file (optional)')
    parser.add_argument('--top_percent', type=float, default=1.0, help='Fraction of top data to keep (e.g., 0.2 for 20%)')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-1.7B", help='Huggingface model name for feature extraction')
    parser.add_argument('--output_path', type=str, default=None, help='Output file path (optional, auto-generated if not provided)')
    args = parser.parse_args()
    
    # Load results from file if provided
    if args.file_path:
        with open(args.file_path, 'r') as f:
            samples = json.load(f)
    else:
        print("Error: --file_path is required when running as a script.")
        print("For programmatic use, import evaluate_samples() function and call it with a list.")
        return

    # Evaluate samples
    res = evaluate_samples(
        samples=samples,
        clf_root=args.clf_root,
        batch_size=args.batch_size,
        model_name=args.model_name,
        top_percent=args.top_percent
    )

    # Save
    if args.output_path:
        output_path = args.output_path
    else:
        model_id = args.model_name.split('/')[-1]
        # Get classification from first classifier metadata, multi/binary
        clf_path = args.clf_root + '/semantic_consistency.pkl'
        _, metadata = load_model_pickle(clf_path)
        classification = metadata["classification"]
        output_path = args.file_path.replace('.json', f'_{model_id}_{classification}_evaled.json')
    
    with open(output_path, 'w') as f:
        json.dump(res, f, indent=4)
    print(f"Saved {len(res)} samples to {output_path}")


if __name__ == "__main__":
    main()
