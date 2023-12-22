# Copyright © 2023 Apple Inc.

import argparse
import json
from pathlib import Path

import numpy as np
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Mistral weights to MLX.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="mistral-7B-v0.1/",
        help="The path to the Mistral model. The MLX weights will also be saved there.",
    )
    args = parser.parse_args()

    try:
        import transformers
    except ImportError as e:
        print("The transformers package must be installed for this model conversion:")
        print("pip install transformers")
        import sys

        sys.exit(0)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(args.model_path)
    ).state_dict()
    config = transformers.AutoConfig.from_pretrained(args.model_path)

    # things to change
    # 1. there's no "model." in the weight names
    model = {k.replace("model.", ""): v for k, v in model.items()}

    # 2. mlp is called feed_forward
    model = {k.replace("mlp", "feed_forward"): v for k, v in model.items()}

    # 3. up_proj, down_proj, gate_proj
    model = {k.replace("down_proj", "w2"): v for k, v in model.items()}
    model = {k.replace("up_proj", "w3"): v for k, v in model.items()}
    model = {k.replace("gate_proj", "w1"): v for k, v in model.items()}

    # 4. layernorms
    model = {
        k.replace("input_layernorm", "attention_norm"): v for k, v in model.items()
    }
    model = {
        k.replace("post_attention_layernorm", "ffn_norm"): v for k, v in model.items()
    }

    # 5. lm head
    model = {k.replace("lm_head", "output"): v for k, v in model.items()}

    # 6. token emb
    model = {k.replace("embed_tokens", "tok_embeddings"): v for k, v in model.items()}

    # 7. attention
    model = {k.replace("self_attn", "attention"): v for k, v in model.items()}
    model = {k.replace("q_proj", "wq"): v for k, v in model.items()}
    model = {k.replace("k_proj", "wk"): v for k, v in model.items()}
    model = {k.replace("v_proj", "wv"): v for k, v in model.items()}
    model = {k.replace("o_proj", "wo"): v for k, v in model.items()}

    params = {}
    params["dim"] = config.hidden_size
    params["hidden_dim"] = config.intermediate_size
    params["n_heads"] = config.num_attention_heads
    if hasattr(config, "num_key_value_heads"):
        params["n_kv_heads"] = config.num_key_value_heads
    params["n_layers"] = config.num_hidden_layers
    params["vocab_size"] = config.vocab_size
    params["norm_eps"] = config.rms_norm_eps
    params["rope_traditional"] = False
    weights = {k: v.to(torch.float16).numpy() for k, v in model.items()}


    args.model_path="../../weights/"+args.model_path
    model_path = Path(args.model_path)
    model_path.mkdir(parents=True, exist_ok=True)
    np.savez(str(model_path / "weights.npz"), **weights)
    with open(model_path / "config.json", "w") as fid:
        json.dump(params, fid, indent=4)
