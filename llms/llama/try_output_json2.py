

# https://github.com/outlines-dev/outlines/tree/main

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Optional, Tuple, List
from sentencepiece import SentencePieceProcessor
import time
from transformers import AutoTokenizer

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from llama import load_model,toc

mx.random.seed(44)

"""
TinLlama >v0.2 uses a different prompt format
<|assistant|>
"""

# model_path="llama/TinyLlama/TinyLlama-1.1B-Chat-v0.6-converted"
model_path="../../weights/llama/Llama-2-7b-chat-mlx"
if not Path(model_path).exists():
    model_path=model_path.replace("../","")

tokenizer_path="../../weights/llama/Llama-2-7b-chat-mlx/tokenizer.model"
if not Path(tokenizer_path).exists():
    tokenizer_path=tokenizer_path.replace("../","")

class Tokenizer:
    def __init__(self, model_path: str):
        assert Path(model_path).exists(), model_path
        self._model = SentencePieceProcessor(model_file=model_path)
        self._sep = "▁"
        assert self._model.vocab_size() == self._model.get_piece_size()

    @property
    def eos_id(self) -> int:
        return self._model.eos_id()

    @property
    def pad_id(self) -> int:
        return self._model.pad_id()

    def encode(self, s: str) -> List[int]:
        return [self._model.bos_id(), *self._model.encode(s)]

    def decode(self, t: List[int]) -> str:
        out = self._model.decode(t)
        if t and self._model.id_to_piece(t[0])[0] == self._sep:
            return " " + out
        return out
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.encode(*args, **kwds)
    
# tokenizer = SentencePieceProcessor(model_file=tokenizer_path)
# tokenizer = Tokenizer(tokenizer_path)
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")

model = load_model(model_path)

temp=0.8
max_tokens=1000
write_every=2





# test lm-format-enforcer
from lmformatenforcer import (
    CharacterLevelParser,
    JsonSchemaParser,
    RegexParser,
    StringParser,
)
from lmformatenforcer.integrations.transformers import (
    build_token_enforcer_tokenizer_data,
    generate_enforced,
)

tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)
parser=JsonSchemaParser(None)
generate_enforced(model, tokenizer_data, parser)
# parser.



#### test outlines
from enum import Enum
from pydantic import BaseModel, constr

import outlines
import torch


class Weapon(str, Enum):
    sword = "sword"
    axe = "axe"
    mace = "mace"
    spear = "spear"
    bow = "bow"
    crossbow = "crossbow"


class Armor(str, Enum):
    leather = "leather"
    chainmail = "chainmail"
    plate = "plate"


class Character(BaseModel):
    name: constr(max_length=10)
    age: int
    armor: Armor
    weapon: Weapon
    strength: int


# model = outlines.models.transformers("mistralai/Mistral-7B-v0.1", device="cuda")

# Construct guided sequence generator
generator = outlines.generate.json(model, Character, max_tokens=100)

# Draw a sample
rng = torch.Generator(device="cuda")
rng.manual_seed(789001)

sequence = generator("Give me a character description", rng=rng)
print(sequence)