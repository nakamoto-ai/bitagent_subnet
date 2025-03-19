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


parser = argparse.ArgumentParser(description="Generate and evaluate tool call tasks")
parser.add_argument(
    "--model",
    type=str,
    default="watt-ai/watt-tool-8B",
    help="The model name or path to use for generation",
)
parser.add_argument(
    "--len",
    type=int,
    default=333, # how many bitagent runs
    help="Number of tasks to evaluate",
)
args = parser.parse_args()

response_gen_model = args.model
output_file = f'eval_results/{response_gen_model.split("/")[-1]}.json'
len = args.len

print(f"Using model: {response_gen_model}")
print(f"Will evaluate {len} tasks")
print(f"Results will be saved to: {output_file}")

tokenizer = AutoTokenizer.from_pretrained(response_gen_model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    response_gen_model, torch_dtype="auto", device_map="auto"
)

system_prompt = """You are an expert in composing functions.
You are given a question and a set of possible functions.
Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, RETURN A BLANK RESPONSE.
If the given question lacks the parameters required by any function,
You should only return the function call in tools call sections.

If you decide to invoke any of the function(s),
you MUST put it in the format of [func_name(params_name1=params_value1, params_name2=params_value2...)].
You SHOULD NOT include any other text in the response.
Here is a list of functions in JSON format that you can invoke:

{functions}"""

class MockedValidator(Validator):
    def __init__(self):
        self.tool_dataset = ToolDataset()

    def validate(self, task):
        return True


val = MockedValidator()

all_results = []
task_name = "tool_call"
scores = []
accuracy = 0.0

# Process batch of tasks
for i in range(len):
    try:
        print(f"Processing task {i+1}/{len}")

        # Create a tool call task
        tool_call_task = ToolCallTask(
            validator=val,
            name="Responds with correct function call",
            offline=True,
        )

        # Format tools and messages
        json_formatted_tools = [tool.__dict__ for tool in tool_call_task.synapse.tools]
        json_formatted_messages = [{"role": msg.role.value, "content": msg.content} for msg in tool_call_task.messages]

        # Prepare input for model
        input = [
            {
                "role": "system",
                "content": system_prompt.format(functions=json_formatted_tools),
            }
        ]
        input.extend(json_formatted_messages)

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

        # Store result details
        result = {
            "task_id": i,
            "response": output,
            "expected_tool_call": tool_call_task.expected_tool_call,
            "total_score": total_score,
            "total_possible": total_possible,
            "detailed_results": results,
        }
        all_results.append(result)

        # Print current result
        print(f"Response: {output}")
        print(f"Expected: {tool_call_task.expected_tool_call}")
        print(f"Score: {total_score}/{total_possible}")
        print(f"Current accuracy: {accuracy}")

    except Exception as e:
        print(f"Error processing task {i+1}: {e}")

# Prepare results summary
accuracy_results = {
    "model": response_gen_model,
    "accuracy": accuracy,
    "detailed_results": all_results
}

# Save results
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w") as f:
    json.dump(accuracy_results, f, indent=2)

print(f"Results saved to {output_file}")
