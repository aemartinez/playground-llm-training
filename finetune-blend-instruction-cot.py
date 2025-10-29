import os
from pathlib import Path
from typing import Dict, Tuple
import random
import math

from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict
from transformers import GPT2TokenizerFast
import argparse
import json
import pickle

# -----------------------------
# CONFIG
# -----------------------------
OUT_DIR = Path("sft_merged")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# change these counts to control the number drawn from each dataset
# TARGET_COUNTS: Dict[str, int] = {
#     # instruction datasets
#     "tatsu-lab/alpaca": 25_000,
#     "databricks/databricks-dolly-15k": 10_000,
#     "teknium/OpenHermes-2.5": 15_000,     # may be large; will sample
#     # reasoning / CoT
#     "thoughtsource/thoughtsource": 20_000,   # ThoughtSource is a collection - we'll pick subsets
#     "open-r1/OpenThoughts-114k": 20_000,
#     "gsm8k": 5_000,
#     "isaiahbjork/chain-of-thought": 5_000,
# }

# Dry run with little data
TARGET_COUNTS: Dict[str, int] = {
    "tatsu-lab/alpaca": 1000,
    "databricks/databricks-dolly-15k": 1000,
    "teknium/OpenHermes-2.5": 1000,
    "thoughtsource/thoughtsource": 1000,
    "open-r1/OpenThoughts-114k": 1000,
    "gsm8k": 1000,
    "isaiahbjork/chain-of-thought": 1000,
}

MAX_SEQ_LENGTH = 2048
RANDOM_SEED = 42
TOKENIZED_CACHE = OUT_DIR / "tokenized_sft_dataset.pkl"
RAW_MERGED_JSON = OUT_DIR / "merged_raw_examples.jsonl"

# -----------------------------
# tokenizer (matching your training)
# -----------------------------
tok = GPT2TokenizerFast.from_pretrained("gpt2")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.model_max_length = MAX_SEQ_LENGTH

# -----------------------------
# helpers: normalization functions
# Each returns list of {"prompt":..., "response":...}
# -----------------------------
def normalize_alpaca(ds):
    # dataset fields: instruction, input, output
    out = []
    for rec in ds:
        instr = rec.get("instruction","")
        inp = rec.get("input","")
        out_text = rec.get("output","").strip()
        prompt = instr if not inp else instr + ("\n\n" + inp)
        out.append({"prompt": prompt.strip(), "response": out_text})
    return out

def normalize_dolly(ds):
    # Dolly format: "instruction" and "response" or "text"
    out = []
    for rec in ds:
        # databricks/dolly-15k has "instruction","context","response"
        instr = rec.get("instruction") or rec.get("prompt") or ""
        ctx = rec.get("context","")
        resp = rec.get("response") or rec.get("output") or rec.get("text") or ""
        prompt = instr if not ctx else instr + ("\n\n" + ctx)
        out.append({"prompt": prompt.strip(), "response": resp.strip()})
    return out

def normalize_openhermes(ds):
    # OpenHermes variants vary; try common fields
    out = []
    for rec in ds:
        instr = rec.get("instruction") or rec.get("user_query") or rec.get("input","")
        resp = rec.get("output") or rec.get("response") or rec.get("assistant","")
        if not instr and "question" in rec:
            instr = rec["question"]
        out.append({"prompt": str(instr).strip(), "response": str(resp).strip()})
    return out

def normalize_thoughtsource(ds):
    # ThoughtSource is multi-dataset; look for fields like "question", "answer", "explanation"
    out = []
    for rec in ds:
        q = rec.get("question") or rec.get("query") or rec.get("input","")
        # explanation may be "explanation", "chain_of_thought", "rationale"
        exp = rec.get("explanation") or rec.get("chain_of_thought") or rec.get("rationale") or ""
        ans = rec.get("answer") or rec.get("final_answer") or ""
        # Build response: explanation (if present) then final answer
        resp = ""
        if exp:
            resp = exp.strip()
            if ans:
                resp += "\n\nFinal answer: " + str(ans).strip()
        else:
            resp = str(ans).strip()
        if not q or not resp:
            continue
        out.append({"prompt": str(q).strip(), "response": resp.strip()})
    return out

def normalize_gsm8k(ds):
    # GSM8K often has "question" and "answer" where answer contains "###" or explanation
    out = []
    for rec in ds:
        q = rec.get("question") or rec.get("problem") or ""
        a = rec.get("answer") or rec.get("final_answer") or ""
        # sometimes answer is "#### 42\n\nExplanation: ...", keep full trace
        out.append({"prompt": str(q).strip(), "response": str(a).strip()})
    return out

def normalize_generic(ds):
    # fallback; try to find q/a-like fields
    out = []
    for rec in ds:
        # a few heuristics
        q = rec.get("prompt") or rec.get("question") or rec.get("input") or rec.get("instruction") or ""
        a = rec.get("response") or rec.get("output") or rec.get("answers") or rec.get("answer") or ""
        out.append({"prompt": str(q).strip(), "response": str(a).strip()})
    return out

# mapping dataset id -> normalizer function and split selection
NORMALIZERS = {
    "tatsu-lab/alpaca": normalize_alpaca,
    "databricks/databricks-dolly-15k": normalize_dolly,
    "teknium/OpenHermes-2.5": normalize_openhermes,
    "thoughtsource/thoughtsource": normalize_thoughtsource,
    "open-r1/OpenThoughts-114k": normalize_thoughtsource,
    "gsm8k": normalize_gsm8k,
    "isaiahbjork/chain-of-thought": normalize_generic,
}

# -----------------------------
# utility: fetch and normalize dataset (with simple caching)
# -----------------------------
def fetch_and_normalize(dataset_id: str, target_count: int, seed=RANDOM_SEED):
    print(f"\n=== Loading {dataset_id} (target {target_count}) ===")
    normalizer = NORMALIZERS.get(dataset_id, normalize_generic)

    # Prefer streaming sample to avoid loading entire dataset
    recs = None
    try:
        # Try explicit train split in streaming mode
        ds_stream = load_dataset(dataset_id, split="train", streaming=True)
    except Exception:
        try:
            # Try without split, then pick a preferred split
            tmp = load_dataset(dataset_id, streaming=True)
            if isinstance(tmp, dict):
                for pref in ("train", "validation", "test"):
                    if pref in tmp:
                        ds_stream = tmp[pref]
                        break
                else:
                    ds_stream = list(tmp.values())[0]
            else:
                ds_stream = tmp
        except Exception:
            ds_stream = None

    if ds_stream is not None:
        print(" - streaming: unknown total size; sampling deterministically...")
        sampled = []
        for i, ex in enumerate(ds_stream.shuffle(seed=seed)):
            sampled.append(ex)
            if i + 1 >= target_count:
                break
        recs = sampled
    else:
        # Fallback: load in-memory and then sample deterministically
        ds = None
        try:
            ds = load_dataset(dataset_id, split="train")
        except Exception:
            try:
                ds = load_dataset(dataset_id)
                if isinstance(ds, dict):
                    for pref in ("train","validation","test"):
                        if pref in ds:
                            ds = ds[pref]
                            break
                    else:
                        ds = list(ds.values())[0]
            except Exception as e:
                print(f"Failed to load {dataset_id}: {e}")
                return []
        print(f" - raw size: {len(ds)} examples")

        total = len(ds)
        if target_count >= total:
            raw_sample = ds
        else:
            rng = random.Random(seed)
            idxs = list(range(total))
            rng.shuffle(idxs)
            sel = idxs[:target_count]
            raw_sample = ds.select(sel)
        recs = list(raw_sample)

    normalized = normalizer(recs)
    # filter empty prompts/responses and too-long prompt+response combos
    filtered = []
    for r in normalized:
        p = r.get("prompt","").strip()
        a = r.get("response","").strip()
        if not p or not a:
            continue
        # quick length check (characters) — tokens will be handled later
        if len(p) + len(a) > 20000:  # sanity upper bound
            continue
        filtered.append({"prompt": p, "response": a})
    print(f" - normalized & filtered -> {len(filtered)} examples")
    return filtered

# -----------------------------
# main: fetch all sources, merge, shuffle
# -----------------------------
def build_merged_dataset(target_counts: Dict[str,int], out_jsonl: Path):
    merged = []
    for ds_id, cnt in target_counts.items():
        examples = fetch_and_normalize(ds_id, cnt, seed=RANDOM_SEED)
        # if the dataset normalizer produced fewer than requested, accept fewer
        if len(examples) == 0:
            print(f"Warning: no examples for {ds_id}; skipping.")
            continue
        # If more than target (due to inconsistent dataset splits), trim deterministically
        if len(examples) > cnt:
            examples = examples[:cnt]
        merged.extend(examples)

    print(f"\nTotal merged examples before shuffle: {len(merged)}")
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(merged)

    # save raw merged as jsonl for audit
    print(f"Writing merged raw examples to {out_jsonl}")
    with out_jsonl.open("w", encoding="utf-8") as f:
        for ex in merged:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # convert to HF Dataset
    ds = Dataset.from_list(merged)
    return ds

# -----------------------------
# Format into single text: "User: <prompt>\nAssistant: <response>"
# -----------------------------
def format_prompt_response(example):
    prompt = example["prompt"].strip()
    response = example["response"].strip()

    # If response already contains "Let's think" or "Step", keep it.
    # Standard wrapper encourages CoT style when present.
    full = f"User: {prompt}\nAssistant: {response}"
    return {"text": full}

# -----------------------------
# Tokenize and (optionally) group into blocks
# -----------------------------
def tokenize_and_group(dataset: Dataset, tokenizer: GPT2TokenizerFast, max_length=MAX_SEQ_LENGTH, group_blocks=True):
    print("Tokenizing dataset (this may take some time)...")

    def to_text(ex):
        return {"text": format_prompt_response(ex)["text"]}

    # map to text column
    text_ds = dataset.map(lambda ex: {"text": [format_prompt_response(e)["text"] for e in ex]}, batched=True, remove_columns=dataset.column_names)
    # tokenization function
    def tok_batch(batch):
        out = tokenizer(batch["text"], truncation=True, max_length=max_length, padding=False)
        return out

    tokenized = text_ds.map(tok_batch, batched=True, batch_size=256, remove_columns=["text"])
    print(f" - tokenized examples: {len(tokenized)}")

    if not group_blocks:
        # prepare labels = input_ids for teacher forcing
        tokenized = tokenized.map(lambda ex: {"labels": ex["input_ids"]}, batched=True)
        return tokenized

    # Group into contiguous blocks of max_length (like language modeling)
    print("Grouping into blocks of max_length...")
    def group_texts(examples):
        all_ids = sum(examples["input_ids"], [])
        total_len = (len(all_ids) // max_length) * max_length
        all_ids = all_ids[:total_len]
        blocks = [all_ids[i:i+max_length] for i in range(0, total_len, max_length)]
        return {"input_ids": blocks,
                "attention_mask": [[1]*max_length for _ in blocks],
                "labels": [list(b) for b in blocks]}

    lm_dataset = tokenized.map(group_texts, batched=True, batch_size=1000, remove_columns=tokenized.column_names)
    # drop possible empty rows
    lm_dataset = lm_dataset.filter(lambda ex: len(ex["input_ids"])>0)
    print(f" - final LM blocks: {len(lm_dataset)}")
    return lm_dataset

# -----------------------------
# entrypoint
# -----------------------------
def main():
    merged_ds = build_merged_dataset(TARGET_COUNTS, RAW_MERGED_JSON)
    print("Sample merged item:", merged_ds[0])

    # Tokenize & group
    tokenized = tokenize_and_group(merged_ds, tok, max_length=MAX_SEQ_LENGTH, group_blocks=False)
    # group_blocks=False here because for SFT we want prompt+response pairs; you can enable grouping for LM-style training instead.
    print("Tokenized preview:", tokenized[0])

    # Save tokenized dataset (as HF arrow) and a pickle cache for quick reload
    tokenized_dataset_path = OUT_DIR / "sft_tokenized_dataset"
    tokenized.save_to_disk(str(tokenized_dataset_path))
    print(f"Saved tokenized dataset to {tokenized_dataset_path}")

    # Optionally also save a small pickle cache for fast loading in existing pipeline
    with TOKENIZED_CACHE.open("wb") as f:
        pickle.dump(tokenized, f)
    print(f"Saved tokenized cache to {TOKENIZED_CACHE}")

if __name__ == "__main__":
    main()
