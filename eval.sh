#!/bin/bash
set -e

# Load HF_TOKEN from file
export HF_TOKEN=$(cat HF_TOKEN)

# Install lm-evaluation-harness and hf_transfer if not already installed
pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git
pip install hf_transfer

# Create tmux session and run evaluation with logging
tmux new-session -d -s eval 'lm_eval --model hf \
  --model_args pretrained=aemartinez/gpt2-small-wiki \
  --tasks mmlu \
  --num_fewshot 4 \
  --device cuda:0 \
  --batch_size 16 2>&1 | tee eval_output.log'

echo "Evaluation started in tmux session 'eval'"
echo "To view output: tmux attach -t eval"
echo "To detach: Ctrl+b then d"
echo "Log file: eval_output.log"
