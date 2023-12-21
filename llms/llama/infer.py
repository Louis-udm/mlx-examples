
# python llama.py TinyLlama/TinyLlama-1.1B-Chat-v0.6-converted/ Llama-2-7b-chat-mlx/tokenizer.model --prompt "[INST] generate a markdown table and convert it to json. [/INST]"

# # Install mlx, mlx-examples, huggingface-cli
# pip install mlx
# pip install huggingface_hub hf_transfer
# git clone https://github.com/ml-explore/mlx-examples.git

# # Download model
# export HF_HUB_ENABLE_HF_TRANSFER=1
# huggingface-cli download --local-dir Llama-2-7b-chat-mlx mlx-llama/Llama-2-7b-chat-mlx

# # Run example
# python mlx-examples/llama/llama.py --prompt "My name is " Llama-2-7b-chat-mlx/ Llama-2-7b-chat-mlx/tokenizer.model

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional, Tuple, List
from sentencepiece import SentencePieceProcessor
import time

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
model_path="llama/Llama-2-7b-chat-mlx"
tokenizer_path="llama/Llama-2-7b-chat-mlx/tokenizer.model"

tokenizer = SentencePieceProcessor(model_file=tokenizer_path)

model = load_model(model_path)

# prompt="Tell me a joke.\nThis is a new line."
prompt="""
Extract the Request for Proposal requirements from the following text. Maintain the original content, but remove redundant blanks. Prefix each requirement block with ## at the beginning.

%%%%

| ID   | Rated Requirement Description                                                                                                                                                                                                                           | Rating Criteria                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Max Points    |
|------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| R1.1 | The solution should provide written a  RPA strategic enterprise    implementation plan which will be  evaluated based on the following  marking scheme:   1.1 E- SDC infrastructure requirements;                                                       | 1 •  Information is well  documented / clear and  relevant to the subject – full  points per bullet  •  Information is semi  documented / unclear /  contains unrelated  information – half points per  bullet  .1 - ESDC infrastructure  requirements evaluation criteria   - includes relevant infrastructure  diagrams (Max 10 points)  - includes relevant infrastructure  hardware and software  requirements(Max 10 points)  - includes relevant costings and  firewall rule requirements(Max 10  points) | Max 30 points |
| R1.2 | The solution should provide written a  RPA strategic enterprise    implementation plan which will be  evaluated based on the following  marking scheme:   1.2 - best practices for business and IT  RPA Centres of Expertise (CoE);                     | •  Information is well  documented / clear and  relevant to the subject – full  points per bullet  •  Information is semi  documented / unclear /  contains unrelated  information – half points per  bullet    1.2 -best practices for business  and IT RPA centres of expertise  (CoE);   - includes relevant best practices  for business RPA center of  excellence. (Max 10 points)  - includes relevant best practices  for IT RPA center of excellence.  (Max 10 points)                                  | Max 20 points |
| R1.3 | 1 The solution should provide written a  RPA strategic enterprise    implementation plan which will be  evaluated based on the following  marking scheme:   .3 - business process evaluation and  creation of inventories of potential  RPA candidates; | •  Information is well  documented / clear and  relevant to the subject – full  points per bullet  •  Information is semi  documented / unclear /  contains unrelated  information – half points per  bullet                                                                                                                                                                                                                                                                                                    | Max 60 points |

%%%%
"""

prompt2="""
Analyze the text contained by %%%% below. Within the text, locate the markdown table, provide the exact content found in the intersection of row 3 and column 2 in the table.

%%%%
# Animal employees
We are **DreamAI**, we have 4 animal employees.
## Detail infomations

SEQ | Kind    |Name    |   Age| City
----|---------|--------|------|----
A1  | Dog    |Fred    |   2 |   Montreal
A2  | Cat     |Jim     |   4 |   Toronto
B1  | Snake   |Harry   |   3 |   Vancouver
B2  | Bird   |Louis   |   5 |   Ottawa

Our employees are welcome for you.

## Brief
Our employees are not working in offfice, they work from home.

%%%%
"""

temp=0.8
max_tokens=1000
write_every=2

print("------")
print(prompt)
# print(tokenizer.encode(prompt))
x = mx.array([[tokenizer.bos_id()] + tokenizer.encode(prompt)])
skip = 0
prompt_processing = None
tokens = []
start = time.time()
for token in model.generate(x, temp):
    tokens.append(token)

    if len(tokens) == 1:
        # Actually perform the computation to measure the prompt processing time
        mx.eval(token)
        prompt_processing = toc("Prompt processing", start)

    if len(tokens) >= max_tokens:
        break

    elif (len(tokens) % write_every) == 0:
        # It is perfectly ok to eval things we have already eval-ed.
        mx.eval(tokens)
        s = tokenizer.decode([t.item() for t in tokens])
        print(s[skip:], end="", flush=True)
        skip = len(s)

mx.eval(tokens)
full_gen = toc("Full generation", start)
s = tokenizer.decode([t.item() for t in tokens])
print(s[skip:], flush=True)
print("------")
print(prompt_processing)
print(full_gen)

