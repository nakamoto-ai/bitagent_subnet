
from math import exp
from bitagent.tasks.tool_call_task import ToolCallTask
from bitagent.schemas.chat import messages_to_list
from bitagent.datasources import ToolDataset
from neurons.validator import Validator
from bitagent.protocol import QueryTask
from bitagent.criteria import (
    default_criteria,
    tool_call_criteria,
    irrelevant_tool_call_criteria,
)
import random
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import time
import torch
import os
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Generate and evaluate tool call tasks")
parser.add_argument(
    "--model",
    type=str,
    default="watt-ai/watt-tool-8B",
    help="The model name or path to use for generation",
)
parser.add_argument(
    "--examples",
    type=int,
    default=333, # how many bitagent runs
    help="Number of tasks to evaluate",
)
args = parser.parse_args()

response_gen_model = args.model
output_file = f'eval_results/{response_gen_model.split("/")[-1]}.json'
examples = args.examples

print(f"Using model: {response_gen_model}")
print(f"Will evaluate {examples} tasks")
print(f"Results will be saved to: {output_file}")

tokenizer = AutoTokenizer.from_pretrained(response_gen_model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    response_gen_model, torch_dtype="auto", device_map="auto"
)

system_prompt = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the function can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function call in tools call sections.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1="params_string_value1", params_name2=params_value2...), func_name2(params)]
Notice that any values that are strings must be put in quotes like this: "params_string_value1"
You SHOULD NOT include any other text in the response.
Here is a list of functions in JSON format that you can invoke.\n{functions}\n
"""


class MockedValidator(Validator):
    def __init__(self):
        self.tool_dataset = ToolDataset()

class MockedToolCallTask(ToolCallTask):
    def __init__(self, validator):
        self.validator = validator

dataset = load_dataset('BitAgent/tool_shuffle_small')

raw_df = pd.DataFrame(dataset['train'])

scores = []
for idx, row in tqdm(raw_df.iterrows(), total=raw_df.shape[0], desc="Processing rows"):
    conversation = row['conversation']
    tools = row['tools']

    print(f"tools ({type(tools)}): {tools}")
    formatted_system_prompt = system_prompt.format(functions=tools)

    print(f"conversation ({type(conversation)}): {conversation}")
    conversation_list = json.loads(conversation)

    user_query = conversation_list[0]['content']
    tool_call = conversation_list[1]['content']

    # Prepare input for model
    input = [
        {
            "role": "system",
            "content": formatted_system_prompt
        },
        {
            "role": "user",
            "content": user_query
        },
        {
            "role": "assistant",
            "content": function_call
        }
    ]

    # Generate response
    inputs = tokenizer.apply_chat_template(
        input, return_tensors="pt"
    ).to(model.device)
    attention_mask = torch.ones_like(inputs).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            inputs,
            max_new_tokens=512,
            do_sample=False,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=attention_mask,
        )
        output = tokenizer.decode(
            outputs[0][len(inputs[0]):], skip_special_tokens=True
        )


    val = MockedValidator()
    tool_call_task = MockedToolCallTask(validator=val)

    # Score response
    syn = tool_call_task.synapse
    syn.response = output

    # mocking like they do in the validator
    syn.dendrite.process_time = 5.0
    syn.dendrite.status_code = 200
    syn.axon.status_code = 200

    total_score, total_possible, results, correct_answer = tool_call_task.reward(validator=val, synapse=syn)

    scores.append(total_score/total_possible)
    accuracy = sum(scores)/len(scores)

    print(f"accuracy: {accuracy}")
