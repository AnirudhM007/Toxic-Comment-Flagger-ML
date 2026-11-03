#!/usr/bin/env python3
"""
Toxic Comment Flagger - Inference CLI & Module
Author: Anirudh Mohite (https://github.com/AnirudhM007)

Provides single-comment and context-aware toxicity prediction using
the trained deep learning models and serialized tokenizers.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Category labels
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

BASE_DIR = Path(__file__).resolve().parent

def load_tokenizer(tokenizer_path=None):
    """Load serialized Keras Tokenizer from JSON."""
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    
    if tokenizer_path is None:
        candidates = [
            BASE_DIR / 'base_tokenizer.json',
            BASE_DIR / 'tokenizer.json'
        ]
        for c in candidates:
            if c.exists():
                tokenizer_path = c
                break
    
    if not tokenizer_path or not Path(tokenizer_path).exists():
        raise FileNotFoundError(f"Tokenizer JSON file not found in {BASE_DIR}")
    
    print(f"[*] Loading tokenizer from: {tokenizer_path.name}")
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_json = f.read()
    return tokenizer_from_json(tokenizer_json)

def load_model(model_path=None):
    """Load trained Keras model."""
    import tensorflow as tf
    
    if model_path is None:
        candidates = [
            BASE_DIR / 'best_model.keras',
            BASE_DIR / 'base_toxic_model.h5',
            BASE_DIR / 'model.keras'
        ]
        for c in candidates:
            if c.exists():
                model_path = c
                break
                
    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found in {BASE_DIR}")
        
    print(f"[*] Loading deep learning model from: {model_path.name}")
    return tf.keras.models.load_model(model_path)

def load_config():
    """Load sequence lengths configuration."""
    cfg_path = BASE_DIR / 'config.json'
    if cfg_path.exists():
        with open(cfg_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"max_len_comment": 150, "max_len_context": 200}

def predict_single(comment, model, tokenizer, max_len=150):
    """Evaluate toxicity on a standalone comment."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    seq = tokenizer.texts_to_sequences([str(comment)])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    
    # Check if model expects dictionary input (multi-branch) or single tensor
    if isinstance(model.input, dict) or (isinstance(model.input, list) and len(model.input) > 1):
        # Fallback empty context for dual-branch models
        pad_ctx = pad_sequences([[]], maxlen=200, padding='post')
        preds = model.predict({'comment': padded, 'context': pad_ctx}, verbose=0)[0]
    else:
        preds = model.predict(padded, verbose=0)[0]
        
    return {label: float(round(p, 4)) for label, p in zip(LABELS, preds)}

def predict_with_context(comment, context, model, tokenizer, max_len_comment=150, max_len_context=200):
    """Evaluate toxicity incorporating preceding conversation context."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    seq_c = tokenizer.texts_to_sequences([str(comment)])
    seq_ctx = tokenizer.texts_to_sequences([str(context)])
    
    pad_c = pad_sequences(seq_c, maxlen=max_len_comment, padding='post', truncating='post')
    pad_ctx = pad_sequences(seq_ctx, maxlen=max_len_context, padding='post', truncating='post')
    
    try:
        preds = model.predict({'comment': pad_c, 'context': pad_ctx}, verbose=0)[0]
    except Exception:
        # Fallback to single comment branch if model is not dual-branch
        preds = model.predict(pad_c, verbose=0)[0]
        
    return {label: float(round(p, 4)) for label, p in zip(LABELS, preds)}

def print_results(comment, results, context=None, threshold=0.5):
    """Format and print prediction outputs with colored toxicity status."""
    print("\n" + "=" * 60)
    print(f"Comment: \"{comment}\"")
    if context:
        print(f"Context: \"{context}\"")
    print("-" * 60)
    print(f"{'Category':<16} | {'Probability':<12} | {'Flagged':<8}")
    print("-" * 60)
    
    flagged_any = False
    for label, prob in results.items():
        is_flagged = prob >= threshold
        if is_flagged:
            flagged_any = True
        status = "[FLAGGED]" if is_flagged else " Clean"
        bar = "#" * int(prob * 20)
        print(f"{label:<16} | {prob:<6.2%} {bar:<8} | {status}")
        
    print("-" * 60)
    verdict = "TOXIC CONTENT DETECTED" if flagged_any else "CONTENT APPROVED (SAFE)"
    print(f"Final Verdict: {verdict}")
    print("=" * 60 + "\n")

def interactive_mode(model, tokenizer, config):
    """Run interactive CLI evaluation loop."""
    print("\n[+] Entering Interactive Toxic Comment Evaluation Mode")
    print("[+] Type 'exit' or 'quit' to terminate.\n")
    
    while True:
        try:
            comment = input("Enter comment to analyze: ").strip()
            if not comment or comment.lower() in ('exit', 'quit'):
                print("Exiting.")
                break
                
            context = input("Enter preceding context (optional, press Enter to skip): ").strip()
            if context:
                res = predict_with_context(
                    comment, context, model, tokenizer,
                    max_len_comment=config.get('max_len_comment', 150),
                    max_len_context=config.get('max_len_context', 200)
                )
                print_results(comment, res, context=context)
            else:
                res = predict_single(
                    comment, model, tokenizer,
                    max_len=config.get('max_len_comment', 150)
                )
                print_results(comment, res)
        except KeyboardInterrupt:
            print("\nExiting.")
            break

def main():
    parser = argparse.ArgumentParser(description="Multi-Label Toxic Comment Flagger CLI")
    parser.add_argument("--text", "-t", type=str, help="Comment text to analyze")
    parser.add_argument("--context", "-c", type=str, default="", help="Preceding conversational context")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification probability threshold (default 0.5)")
    parser.add_argument("--model", type=str, default=None, help="Path to custom model file (.keras or .h5)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive evaluation session")
    args = parser.parse_args()

    config = load_config()
    tokenizer = load_tokenizer()
    model = load_model(args.model)

    if args.interactive or not args.text:
        interactive_mode(model, tokenizer, config)
    else:
        if args.context:
            res = predict_with_context(
                args.text, args.context, model, tokenizer,
                max_len_comment=config.get('max_len_comment', 150),
                max_len_context=config.get('max_len_context', 200)
            )
            print_results(args.text, res, context=args.context, threshold=args.threshold)
        else:
            res = predict_single(
                args.text, model, tokenizer,
                max_len=config.get('max_len_comment', 150)
            )
            print_results(args.text, res, threshold=args.threshold)

if __name__ == '__main__':
    main()
