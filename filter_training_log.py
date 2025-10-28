#!/usr/bin/env python3
import argparse
import os
import re
import sys


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a tmux/python training log to extract training and eval metrics. "
            "Produces two files: one for training loss metrics and one for eval loss metrics."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="/workspace/tmux_python_training.log",
        help="Path to the input log file (default: /workspace/tmux_python_training.log)",
    )
    parser.add_argument(
        "--train-output",
        help="Path for training metrics output file (default: <input>.train.log)",
    )
    parser.add_argument(
        "--eval-output", 
        help="Path for eval metrics output file (default: <input>.eval.log)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting existing output files. The input file is never modified.",
    )
    parser.add_argument(
        "--print",
        dest="to_stdout",
        action="store_true",
        help="Also print matched lines to stdout while writing the files.",
    )
    return parser


def determine_output_paths(input_path: str, train_output: str | None, eval_output: str | None) -> tuple[str, str]:
    base, ext = os.path.splitext(input_path)
    default_ext = ext or '.log'
    
    train_path = train_output or f"{base}.train{default_ext}"
    eval_path = eval_output or f"{base}.eval{default_ext}"
    
    return train_path, eval_path


def compile_patterns() -> tuple[re.Pattern[str], re.Pattern[str]]:
    # Training metrics pattern: {'loss': ..., 'grad_norm': ..., 'learning_rate': ..., 'epoch': ...}
    train_pattern = r"^\{\s*'loss':\s*[^,]+,\s*'grad_norm':\s*[^,]+,\s*'learning_rate':\s*[^,]+,\s*'epoch':\s*[^}]+\s*\}$"
    
    # Eval metrics pattern: {'eval_loss': ..., 'eval_runtime': ..., 'eval_samples_per_second': ..., 'eval_steps_per_second': ..., 'epoch': ...}
    eval_pattern = r"^\{\s*'eval_loss':\s*[^,]+,\s*'eval_runtime':\s*[^,]+,\s*'eval_samples_per_second':\s*[^,]+,\s*'eval_steps_per_second':\s*[^,]+,\s*'epoch':\s*[^}]+\s*\}$"
    
    return re.compile(train_pattern), re.compile(eval_pattern)


def filter_log(input_path: str, train_output: str, eval_output: str, to_stdout: bool = False) -> tuple[int, int]:
    train_matcher, eval_matcher = compile_patterns()

    try:
        with open(input_path, "r", encoding="utf-8", errors="replace") as fin, \
             open(train_output, "w", encoding="utf-8") as train_fout, \
             open(eval_output, "w", encoding="utf-8") as eval_fout:
            train_matches = 0
            eval_matches = 0
            
            for line in fin:
                stripped_line = line.rstrip("\n")
                # Keep original line endings consistent in output
                if train_matcher.match(stripped_line):
                    train_fout.write(line)
                    if to_stdout:
                        sys.stdout.write(f"[TRAIN] {line}")
                    train_matches += 1
                elif eval_matcher.match(stripped_line):
                    eval_fout.write(line)
                    if to_stdout:
                        sys.stdout.write(f"[EVAL] {line}")
                    eval_matches += 1
                    
            return train_matches, eval_matches
    except FileNotFoundError:
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1, 1


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_path = args.input
    train_output, eval_output = determine_output_paths(input_path, args.train_output, args.eval_output)

    # Safety: never allow writing back to the same path as the input
    if (os.path.abspath(train_output) == os.path.abspath(input_path) or 
        os.path.abspath(eval_output) == os.path.abspath(input_path)):
        print("Error: output paths must differ from input path to avoid modifying the original log.", file=sys.stderr)
        sys.exit(2)

    # Safety: avoid accidental overwrite unless --force is provided
    for output_path in [train_output, eval_output]:
        if os.path.exists(output_path) and not args.force:
            print(
                f"Error: output file already exists: {output_path}. Use --force to overwrite.",
                file=sys.stderr,
            )
            sys.exit(3)

    # Create output directories if needed
    for output_path in [train_output, eval_output]:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    train_matches, eval_matches = filter_log(input_path, train_output, eval_output, to_stdout=args.to_stdout)
    
    if isinstance(train_matches, int) and isinstance(eval_matches, int) and train_matches >= 0 and eval_matches >= 0:
        print(f"Wrote {train_matches} training metric lines to {train_output}")
        print(f"Wrote {eval_matches} eval metric lines to {eval_output}")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()


