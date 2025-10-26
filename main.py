# %%
import subprocess
import sys

# %%
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# %%
try:
    import datasets
    import transformers
    import torch
    import 
    print("✅ datasets, transformers, and torch already installed")
except ImportError:
    print("📦 Installing datasets...")
    install_package("datasets")
    install_package("transformers")
    install_package("torch")
# %%
from datasets import load_dataset

# %%
ds = load_dataset("wikitext", "wikitext-103-raw-v1")
train_ds = ds['train']

# %% 
# keep only 10% of the dataset
seed = 42
train_ds = train_ds.shuffle(seed=seed).select(range(int(len(train_ds) * 0.1)))
# %%
cleaned_train_ds = train_ds.filter(lambda x: x["text"].strip() != "" and 20 < len(x["text"]) < 20000)
# %%
print(f"Original train size: {len(train_ds)}")
print(f"Filtered train size: {len(cleaned_train_ds)}")
print(f"Kept: {len(cleaned_train_ds) / len(train_ds) * 100:.1f}%")

# %%
from transformers import GPT2TokenizerFast
tok = GPT2TokenizerFast.from_pretrained("gpt2")
# %%
def tokenize(batch):
    return tok(batch["text"], truncation=False)

# %%
cleaned_train_ds = cleaned_train_ds.map(tokenize, batched=True, remove_columns=cleaned_train_ds.column_names)
# %%
example = cleaned_train_ds[0]
print(example)

print(tok.decode(example["input_ids"]))
# %%
block_size = 1024
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

def group_texts(examples):
    all_ids = sum(examples["input_ids"], [])           # concatenate lists
    total_len = (len(all_ids) // block_size) * block_size
    all_ids = all_ids[:total_len]
    blocks = [all_ids[i:i+block_size] for i in range(0, total_len, block_size)]
    return {"input_ids": blocks, "attention_mask": [[1]*block_size for _ in blocks],
            "labels": [list(b) for b in blocks]}
# %%
lm_dataset = cleaned_train_ds.map(group_texts, batched=True, batch_size=1000, remove_columns=cleaned_train_ds.column_names)

# %%
print(len(lm_dataset))
print(lm_dataset[0]["input_ids"][:10])
# %%
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
# %%
from transformers import GPT2Config, GPT2LMHeadModel
cfg = GPT2Config(
    vocab_size=len(tok),
    n_positions=block_size,
    n_ctx=block_size,
    n_embd=768,
    n_layer=12,
    n_head=12,
)
model = GPT2LMHeadModel(cfg)

# %%
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="out/gpt2-100M",
    overwrite_output_dir=False,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=5e-4,
    weight_decay=0.1,
    warmup_steps=2000,
    fp16=True,                 # set False if you don't have a CUDA device
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=2000,
    save_total_limit=5,
    eval_strategy="no",        # older alias; your signature shows `eval_strategy` is supported
    dataloader_num_workers=2,
    run_name="gpt2-100M-exp"
)
# %%
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, DataCollatorForLanguageModeling

# small subset
train_small = lm_dataset.select(range(min(8, len(lm_dataset))))

# small-ish model for quick smoke (or use your real cfg)
cfg = GPT2Config(vocab_size=len(tok), n_positions=block_size, n_ctx=block_size,
                 n_embd=512, n_layer=6, n_head=8)
model = GPT2LMHeadModel(cfg)

data_collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

# override training_args for smoke (run only 2 steps)
smoke_args = training_args.__class__(**{**training_args.to_sanitized_dict(), **{
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 1,
    "max_steps": 2,
    "fp16": False,    # set False to avoid fp16 issues while debugging
    "save_strategy": "no",
    "logging_steps": 1,
}})

trainer = Trainer(model=model, args=smoke_args, train_dataset=train_small, data_collator=data_collator)
res = trainer.train()
print("Smoke test metrics:", res.metrics)
