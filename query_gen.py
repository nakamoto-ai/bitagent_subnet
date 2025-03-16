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
import boto3
import torch
import os


parser = argparse.ArgumentParser(description="Generate and evaluate tool call tasks")
parser.add_argument(
    "--model",
    type=str,
    default="BitAgent/BitAgent-8B",
    help="The model name or path to use for generation",
)
args = parser.parse_args()

response_gen_model = args.model
output_base_name = response_gen_model.split("/")[-1].lower()

print(f"Using model: {response_gen_model}")
print(f"Output files will be named: {output_base_name}_N.json")

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

TASK_FREQUENCY = {
    "tool_call": 1,
}

TASK_WEIGHTS = {
    "tool_call": 0.05,
}

task_names = list(TASK_FREQUENCY.keys())
task_frequencies = list(TASK_FREQUENCY.values())
choice = random.choices(task_names, weights=task_frequencies)[0]


class MockedValidator(Validator):
    def __init__(self):
        pass
        self.tool_dataset = ToolDataset()

    def validate(self, task):
        return True


val = MockedValidator()
syn = QueryTask()

s3_client = boto3.client("s3")


def upload_file_to_s3(file_path, bucket_name, object_name=None):
    try:
        if object_name is None:
            object_name = file_path.split("/")[-1]

        # Upload the file
        s3_client.upload_file(Filename=file_path, Bucket=bucket_name, Key=object_name)
        print(f"Successfully uploaded {file_path} to {bucket_name}/{object_name}")
    except Exception as e:
        print(f"Error uploading file: {e}")


while True:
    tasks = []
    task_datas = []
    task_rewards = []
    tasks_and_rewards = []

    batch_size = 1000
    for i in range(batch_size):
        try:
            match choice:
                case "tool_call":
                    print(f"Scoring task {i}/{batch_size}")
                    tool_call_task = ToolCallTask(
                        validator=val,
                        name="Responds with correct function call",
                        offline=True,
                    )
                    tasks.append(tool_call_task)

                    #print(f"Generated messages: {tool_call_task.messages}")
                    #print(f"Generated tools: {tool_call_task.synapse.tools}")

                    json_formatted_tools = [tool.__dict__ for tool in tool_call_task.synapse.tools]
                    json_formatted_messages = [msg.__dict__ for msg in tool_call_task.messages]

                    input = [
                        {
                            "role": "system",
                            "content": system_prompt.format(functions=json_formatted_tools),
                        }
                    ]
                    for msg in tool_call_task.messages:
                        input.append({
                            "role": msg.role,
                            "content": msg.content
                        })

                    print(f"Created input: {input}")

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
                            outputs[0][len(inputs[0]) :], skip_special_tokens=True
                        )
                    syn.response = output

                    print(f"response: {output}")

                    total_score, total_possible, results, correct_answer = tool_call_task.reward(validator=val, synapse=syn)

                    data_dict = {
                        "input": input,
                        "tools_json": json_formatted_tools,
                        "messages_json": json_formatted_messages,
                        "response": syn.response,
                        "expected_tool_call": tool_call_task.expected_tool_call,
                        "total_score": total_score,
                        "total_possible": total_possible,
                        "results": results,
                    }

                    print("Row contents:")
                    for key, value in data_dict.items():
                        print(f"  {key}: {value}\n")

                    tasks_and_rewards.append(data_dict)
                    print("\n\n")

        except Exception as e:
            # bt.logging.warning(f'Error getting task (name {choice}): ', e)
            # bt.logging.warning(traceback.format_exc())
            raise e
            print(f"Error getting task (name {choice}): ", e)

    # Write tasks_and_rewards to json file
    unix_timestamp = time.time()
    output_filename = f"{output_base_name}_{unix_timestamp}.json"
    output_path = f"output/{output_base_name}/{output_filename}"
    serializable_data = []

    for item in tasks_and_rewards:
        # Convert task to serializable format
        task_dict = {
            "name": item["task"].name,
            "type": item["task"].__class__.__name__,
            "messages": [msg.to_dict() for msg in item["task"].messages],
            "tools": [tool.to_dict() for tool in item["task"].synapse.tools],
        }

        # If task has additional attributes to save, add them here
        if hasattr(item["task"], "task_data"):
            task_dict["task_data"] = item["task"].task_data

        # Format reward data
        reward_data = {
            "value": float(item["reward"])
            if hasattr(item["reward"], "__float__")
            else item["reward"]
        }

        # Create serializable entry
        entry = {"task": task_dict, "response": item["response"], "reward": reward_data}

        serializable_data.append(entry)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(serializable_data, f, indent=2)

    print(f"Tasks and rewards data written to {output_path}")

    # upload file to s3
    bucket = "sn20"
    upload_file_to_s3(
        output_path, bucket, f"generated/{output_base_name}/{output_filename}"
    )
