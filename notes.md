https://pypi.org/project/poetry-add-requirements-txt/
 pip install poetry-add-requirements.txt
Poeareq   dir/reqirements.txt

poetry add hf_transfer

https://huggingface.co/mlx-community/Llama-2-7b-chat-mlx
huggingface-cli download --local-dir Llama-2-7b-chat-mlx mlx-llama/Llama-2-7b-chat-mlx
python llama.py Llama-2-7b-chat-mlx Llama-2-7b-chat-mlx/tokenizer.model --prompt "Tell me a Joke."

https://huggingface.co/mlx-community/CodeLlama-7b-Python-mlx
huggingface-cli download --local-dir CodeLlama-7b-Python-mlx mlx-llama/CodeLlama-7b-Python-mlx
python mlx-examples/llama/llama.py CodeLlama-7b-Python-mlx/ CodeLlama-7b-Python-mlx/tokenizer.model "def fibonacci("



python convert.py --model-path TinyLlama/TinyLlama-1.1B-Chat-v0.6 --model-name tiny_llama
python convert.py --model-path TTinyLlama/TinyLlama-1.1B-python-v0.1 --model-name tiny_llama


# # Download model
# export HF_HUB_ENABLE_HF_TRANSFER=1
# huggingface-cli download --local-dir weights/llama/Llama-2-7b-chat-mlx mlx-llama/Llama-2-7b-chat-mlx


python llama.py ../../weights/TinyLlama/TinyLlama-1.1B-Chat-v0.6-converted/ Llama-2-7b-chat-mlx/tokenizer.model --prompt "[INST] Generate a markdown table. [/INST]"

python llama.py ../../weights/llama/TinyLlama/TinyLlama-1.1B-Chat-v0.6-converted/ Llama-2-7b-chat-mlx/tokenizer.model --prompt "[INST] generate a markdown table and convert it to json. [/INST]"

python llama.py ../../weights/llama/TinyLlama/TinyLlama-1.1B-Chat-v0.6-converted/ ../../weights/llama/Llama-2-7b-chat-mlx/tokenizer.model --prompt "[INST] Generate a markdown table. [/INST]"