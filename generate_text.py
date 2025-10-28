#!/usr/bin/env python3
"""
Text generation script using the trained model from Hugging Face.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

def load_hf_token():
    """Load HF token from file."""
    try:
        with open('HF_TOKEN', 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        print("Warning: HF_TOKEN file not found. Using environment variable or default.")
        return os.getenv('HF_TOKEN', None)

def load_model_and_tokenizer(model_name, device='cuda'):
    """Load model and tokenizer from Hugging Face."""
    print(f"Loading model: {model_name}")
    
    # Load HF token
    hf_token = load_hf_token()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True
    )
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device_map='auto' if device == 'cuda' else None,
        trust_remote_code=True
    )
    
    if device == 'cuda' and not next(model.parameters()).is_cuda:
        model = model.to(device)
    
    print(f"Model loaded successfully on {device}")
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7, top_p=0.9, device='cuda'):
    """Generate text from a prompt."""
    print(f"\nGenerating text for prompt: '{prompt}'")
    print("-" * 50)
    
    # Tokenize input
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    # Decode generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the new generated part (remove the original prompt)
    new_text = generated_text[len(prompt):].strip()
    
    print(f"Generated text:\n{new_text}")
    print("-" * 50)
    
    return generated_text, new_text

def main():
    parser = argparse.ArgumentParser(description='Generate text using trained model')
    parser.add_argument('--model', default='aemartinez/gpt2-small-wiki', 
                       help='Model name or path')
    parser.add_argument('--prompt', default='The future of artificial intelligence is',
                       help='Text prompt for generation')
    parser.add_argument('--max_length', type=int, default=100,
                       help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Top-p sampling parameter')
    parser.add_argument('--device', default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load model and tokenizer
    try:
        model, tokenizer = load_model_and_tokenizer(args.model, args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    if args.interactive:
        # Interactive mode
        print("\nInteractive text generation mode. Type 'quit' to exit.")
        print("=" * 60)
        
        while True:
            try:
                prompt = input("\nEnter prompt: ").strip()
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                if not prompt:
                    continue
                
                generate_text(model, tokenizer, prompt, 
                           args.max_length, args.temperature, args.top_p, args.device)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    else:
        # Single generation
        generate_text(model, tokenizer, args.prompt, 
                     args.max_length, args.temperature, args.top_p, args.device)

if __name__ == "__main__":
    main()