# https://github.com/noamgat/lm-format-enforcer/blob/main/samples/colab_llama2_enforcer.ipynb

from typing import Tuple, Optional, Union, List
import pandas as pd
from lmformatenforcer import (
    JsonSchemaParser,
    CharacterLevelParser,
    RegexParser,
    StringParser,
)
from lmformatenforcer.integrations.transformers import (
    generate_enforced,
    build_token_enforcer_tokenizer_data,
)

StringOrManyStrings = Union[str, List[str]]

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "NousResearch/Llama-2-7b-chat-hf"
device = "cuda"

if torch.cuda.is_available():
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
else:
    raise Exception("GPU not available")
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token_id is None:
    # Required for batching example
    tokenizer.pad_token_id = tokenizer.eos_token_id


def get_prompt(message: str, system_prompt: str) -> str:
    texts = [f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"]
    # The first user input is _not_ stripped
    do_strip = False
    message = message.strip() if do_strip else message
    texts.append(f"{message} [/INST]")
    return "".join(texts)


tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)


def run(
    message: StringOrManyStrings,
    system_prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 50,
    num_beams: int = 1,
    required_regex: Optional[str] = None,
    required_str: Optional[str] = None,
    required_json_schema: Optional[dict] = None,
    required_json_output: Optional[bool] = None,
) -> Tuple[StringOrManyStrings, Optional[pd.DataFrame]]:
    is_multi_message = isinstance(message, list)
    messages = message if is_multi_message else [message]
    prompts = [get_prompt(msg, system_prompt) for msg in messages]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        add_special_tokens=False,
        return_token_type_ids=False,
        padding=is_multi_message,
    ).to(device)

    generate_kwargs = dict(
        inputs,
        # streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=num_beams,
        output_scores=True,
        return_dict_in_generate=True,
    )

    parser: Optional[CharacterLevelParser] = None
    if required_regex:
        parser = RegexParser(required_regex)
    if required_str:
        parser = StringParser(required_str)
    if required_json_schema:
        parser = JsonSchemaParser(required_json_schema)
    if required_json_output:
        parser = JsonSchemaParser(None)

    if parser:
        output = generate_enforced(model, tokenizer_data, parser, **generate_kwargs)
    else:
        output = model.generate(**generate_kwargs)

    sequences = output["sequences"]
    # skip_prompt=True doesn't work consistenly, so we hack around it.
    string_outputs = [
        tokenizer.decode(sequence, skip_special_tokens=True) for sequence in sequences
    ]
    string_outputs = [
        string_output.replace(prompt[3:], "")
        for string_output, prompt in zip(string_outputs, prompts)
    ]
    if parser and not is_multi_message:
        enforced_scores_dict = output.enforced_scores
        enforced_scores = pd.DataFrame(enforced_scores_dict)
        pd.set_option("display.width", 1000)
        pd.set_option("display.max_columns", 10)
        pd.set_option("display.max_rows", 999)
        pd.set_option("display.float_format", " {:,.5f}".format)
    else:
        enforced_scores = None
    return string_outputs if is_multi_message else string_outputs[0], enforced_scores


from pydantic import BaseModel
from typing import List


DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""
DEFAULT_MAX_NEW_TOKENS = 100


class AnswerFormat(BaseModel):
    first_name: str
    last_name: str
    year_of_birth: int
    num_seasons_in_nba: int


question = "Please give me information about Michael Jordan. You MUST answer using the following json schema: "
question_with_schema = f"{question}{AnswerFormat.schema_json()}"

print("Question:")
print(question_with_schema)

print("Answer, With json mode enforcing (json output, schemaless):")
result, _ = run(
    question_with_schema,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    required_json_output=True,
)
print("----final result-----")
print(result)

print("Answer, With json schema enforcing:")
result, enforced_scores = run(
    question_with_schema,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    required_json_schema=AnswerFormat.schema(),
)
print("----final result-----")
print(result)

# print("Answer, Without json schema enforcing:")
# result, _ = run(
#     question_with_schema,
#     system_prompt=DEFAULT_SYSTEM_PROMPT,
#     max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
# )
# print(result)



# output
"""
Question:

Please give me information about Michael Jordan. You MUST answer using the following json schema: {"title": "AnswerFormat", "type": "object", "properties": {"first_name": {"title": "First Name", "type": "string"}, "last_name": {"title": "Last Name", "type": "string"}, "year_of_birth": {"title": "Year Of Birth", "type": "integer"}, "num_seasons_in_nba": {"title": "Num Seasons In Nba", "type": "integer"}}, "required": ["first_name", "last_name", "year_of_birth", "num_seasons_in_nba"]}
Answer, With json schema enforcing:

/home/noamgat/mambaforge/envs/commentranker/lib/python3.10/site-packages/transformers/generation/utils.py:1411: UserWarning: You have modified the pretrained model configuration to control generation. This is a deprecated strategy to control generation and will be removed soon, in a future version. Please use a generation configuration file (see https://huggingface.co/docs/transformers/main_classes/text_generation )
  warnings.warn(
  {
"first_name": "Michael",
"last_name": "Jordan",
"year_of_birth": 1963,
"num_seasons_in_nba": 15
}
Answer, Without json schema enforcing:

  Of course! I'd be happy to help you with your question about Michael Jordan. Here's the information you requested in the format you specified:
{
"title": "AnswerFormat",
"type": "object",
"properties": {
"first_name": {
"title": "First Name",
"type": "string"
},
"last_name": {
"title": "Last Name",
"type": "string"
Answer, With json mode enforcing (json output, schemaless):

  {
"title": "AnswerFormat",
"type": "object",
"properties": {
"first_name": {
"title": "First Name",
"type": "string"
},
"last_name": {
"title": "Last Name",
"type": "string"
},
"year_of_birth": {
"title": "Year Of Birth",
"type": "integer"
},
"""