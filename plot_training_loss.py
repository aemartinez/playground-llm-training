#!/usr/bin/env python3
import argparse
import ast
from typing import List, Tuple

import matplotlib.pyplot as plt


def parse_log_file(log_path: str, loss_key: str = "loss") -> List[Tuple[float, float]]:
    """
    Parse a log file where each line is a Python dict-like string containing
    at least 'epoch' and the specified loss key. Returns a list of (epoch, loss) tuples.
    Lines that cannot be parsed or that are missing fields are skipped.
    """
    epoch_loss_pairs: List[Tuple[float, float]] = []

    with open(log_path, "r", encoding="utf-8") as infile:
        for raw_line in infile:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = ast.literal_eval(line)
                if not isinstance(record, dict):
                    continue
                epoch = record.get("epoch")
                loss = record.get(loss_key)
                if epoch is None or loss is None:
                    continue
                epoch_loss_pairs.append((float(epoch), float(loss)))
            except Exception:
                # Silently skip malformed lines
                continue

    # Sort by epoch to ensure plotting is ordered
    epoch_loss_pairs.sort(key=lambda x: x[0])
    return epoch_loss_pairs


def plot_loss_vs_epoch(train_points: List[Tuple[float, float]], 
                      eval_points: List[Tuple[float, float]], 
                      output_path: str) -> None:
    if not train_points and not eval_points:
        raise ValueError("No valid (epoch, loss) data points parsed from log files.")

    plt.figure(figsize=(10, 6))
    
    # Plot training loss
    if train_points:
        train_epochs = [p[0] for p in train_points]
        train_losses = [p[1] for p in train_points]
        plt.plot(train_epochs, train_losses, label="Training Loss", color="#1f77b4", linewidth=1.8)
    
    # Plot evaluation loss
    if eval_points:
        eval_epochs = [p[0] for p in eval_points]
        eval_losses = [p[1] for p in eval_points]
        plt.plot(eval_epochs, eval_losses, label="Evaluation Loss", color="#ff7f0e", linewidth=1.8, linestyle="--")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Loss vs Epoch")
    plt.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training and evaluation loss vs epoch from tmux training log files.")
    parser.add_argument(
        "--train-log",
        default="/workspace/tmux_python_training.train.log",
        help="Path to training log file (default: /workspace/tmux_python_training.train.log)",
    )
    parser.add_argument(
        "--eval-log",
        default="/workspace/tmux_python_training.eval.log",
        help="Path to evaluation log file (default: /workspace/tmux_python_training.eval.log)",
    )
    parser.add_argument(
        "--output",
        default="/workspace/loss_vs_epoch.png",
        help="Path to output PNG file (default: /workspace/loss_vs_epoch.png)",
    )
    args = parser.parse_args()

    # Parse training log (uses 'loss' key)
    train_data = parse_log_file(args.train_log, "loss")
    print(f"Parsed {len(train_data)} training data points from {args.train_log}")
    
    # Parse evaluation log (uses 'eval_loss' key)
    eval_data = parse_log_file(args.eval_log, "eval_loss")
    print(f"Parsed {len(eval_data)} evaluation data points from {args.eval_log}")
    
    plot_loss_vs_epoch(train_data, eval_data, args.output)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()


