# %%
import os
import subprocess
import sys

# %%
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# %%
install_package("datasets")
install_package("transformers[torch]")
install_package("torch")
install_package("torchvision")
install_package("torchaudio")
install_package("hf_transfer")
install_package("accelerate>=0.26.0")
install_package("huggingface_hub")
# %%
from datasets import load_dataset
from transformers import GPT2TokenizerFast
tok = GPT2TokenizerFast.from_pretrained("gpt2")
# %%
ds = load_dataset("wikitext", "wikitext-103-raw-v1")
train_ds = ds['train']

# %% 
seed = 42
train_ds = train_ds.shuffle(seed=seed)
# %%
cleaned_train_ds = train_ds.filter(lambda x: x["text"].strip() != "", batched=False)

# %%
print(f"Original train size: {len(train_ds)}")
print(f"Filtered train size: {len(cleaned_train_ds)}")
print(f"Kept: {len(cleaned_train_ds) / len(train_ds) * 100:.1f}%")

# %%
def tokenize(batch):
    return tok(batch["text"], truncation=False)

# %%
tokenized = cleaned_train_ds.map(tokenize, batched=True, remove_columns=cleaned_train_ds.column_names)
# %%
example = tokenized[0]
print(example)

print(tok.decode(example["input_ids"]))
# %%
# ensure pad token
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# shuffle and split BEFORE grouping to ensure disjoint train/eval
tokenized_shuf = tokenized.shuffle(seed=seed)
split = tokenized_shuf.train_test_split(test_size=0.02, seed=seed)  # 2% eval
tokenized_train = split["train"]
tokenized_eval  = split["test"]

block_size = 1024

def group_texts(examples):
    all_ids = sum(examples["input_ids"], [])
    total_len = (len(all_ids) // block_size) * block_size
    all_ids = all_ids[:total_len]
    blocks = [all_ids[i:i+block_size] for i in range(0, total_len, block_size)]
    return {"input_ids": blocks,
            "attention_mask": [[1]*block_size for _ in blocks],
            "labels": [list(b) for b in blocks]}

# apply grouping to train and eval separately
lm_train = tokenized_train.map(group_texts, batched=True, batch_size=1000, remove_columns=tokenized_train.column_names)
lm_eval  = tokenized_eval.map(group_texts,  batched=True, batch_size=1000, remove_columns=tokenized_eval.column_names)

# drop empty rows (if any)
lm_train = lm_train.filter(lambda ex: len(ex["input_ids"])>0)
lm_eval  = lm_eval.filter(lambda ex: len(ex["input_ids"])>0)

print("train blocks after re-chunk:", len(lm_train))
print("eval blocks after re-chunk :", len(lm_eval))

# keep lm_dataset variable name for downstream code compatibility
lm_dataset = lm_train
eval_ds = lm_eval

# %%
print(len(lm_dataset))
print(lm_dataset[0]["input_ids"][:10])
# %%
from transformers import DataCollatorForLanguageModeling, Trainer
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
model.gradient_checkpointing_enable()   # saves memory at cost of compute
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# %%
train_ds = lm_dataset

from transformers import TrainingArguments
import torch

# %%
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])

# %% 
import math

num_train_samples = len(train_ds)
per_device = 8   # choose based on GPU memory — A100-40GB can often do 8 for 100M model at 1024
grad_accum = 8
effective_batch = per_device * grad_accum

# estimate steps per epoch and total steps
steps_per_epoch = math.ceil(num_train_samples / per_device / grad_accum)
num_train_epochs = 3
total_training_steps = steps_per_epoch * num_train_epochs

# warmup: use fraction (e.g. 3% of total steps) but at least 100
warmup_steps = max(100, int(total_training_steps * 0.03))

# choose save/eval intervals relative to total steps
save_steps = max(500, total_training_steps // 10)
eval_steps = max(200, total_training_steps // 20)

print(f"num_train_samples={num_train_samples}, total_steps={total_training_steps}, warmup={warmup_steps}, save={save_steps}, eval={eval_steps}")


# %%
training_args = TrainingArguments(
    output_dir="out/gpt2-100M",
    overwrite_output_dir=False,
    per_device_train_batch_size=per_device,    # from above
    gradient_accumulation_steps=grad_accum,
    learning_rate=2e-4,                        # lower for stable from-scratch training
    weight_decay=0.01,
    num_train_epochs=num_train_epochs,
    warmup_steps=warmup_steps,
    fp16=torch.cuda.is_available(),
    logging_strategy="steps",
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=eval_steps,
    save_strategy="steps",
    save_steps=save_steps,
    save_total_limit=5,
    dataloader_num_workers=4,
    hub_model_id="aemartinez/gpt2-small-wiki",
    hub_strategy="every_save",
    push_to_hub=True, 
    remove_unused_columns=False,
)

print("args created")

# %%
# trainer = Trainer(model=model, args=smoke_args, train_dataset=train_small, eval_dataset=eval_ds, data_collator=data_collator)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
)

trainer.train()
# %%

# %%
login(token=os.environ["HF_TOKEN"])
trainer.push_to_hub(commit_message="Final checkpoint")

# %%
eval_res = trainer.evaluate()
print("eval_loss:", eval_res["eval_loss"])
print("eval_ppl:", math.exp(eval_res["eval_loss"]))