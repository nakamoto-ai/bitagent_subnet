from bitagent.tasks.tool_call_task import ToolCallTask
from bitagent.datasources import ToolDataset
from neurons.validator import Validator
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
    default="watt-ai/watt-tool-8B",
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

system_prompt = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the function can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function call in tools call sections.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1="params_string_value1", params_name2=params_value2...), func_name2(params)]
Notice that any values that are strings must be put in quotes like this: "params_string_value1"
You SHOULD NOT include any other text in the response.
Here is a list of functions in JSON format that you can invoke.\n{functions}\n"""



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

def main():
    class MockedValidator(Validator):
        def __init__(self):
            self.tool_dataset = ToolDataset(task_dataset_flag=False)
            self.task_dataset = ToolDataset(task_dataset_flag=True)
            version = "1.0.8"
            version_split = version.split(".")
            spec_version = (
                (1000 * int(version_split[0]))
                + (10 * int(version_split[1]))
                + (1 * int(version_split[2]))
            )
            self.spec_version = spec_version
            self.seed = self.spec_version*1000000

        def validate(self, task):
            return True


    val = MockedValidator()


    tasks = []
    task_datas = []
    task_rewards = []
    tasks_and_rewards = []

    batch_size = 333

    for i in range(batch_size):
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
        json_formatted_messages = [{"role": msg.role.value, "content": msg.content} for msg in tool_call_task.messages]

        input = [
            {
                "role": "system",
                "content": system_prompt.format(functions=json_formatted_tools),
            }
        ]
        input.extend(json_formatted_messages)

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

        syn = tool_call_task.synapse
        syn.response = output

        # to get the score for it (also done in validator code)
        syn.dendrite.process_time = 5.0
        syn.dendrite.status_code = 200
        syn.axon.status_code = 200

        print(f"response: {output}")

        total_score, total_possible, results, correct_answer = tool_call_task.reward(validator=val, synapse=syn)

        data_dict = {
            "input": input,
            "tools_json": json_formatted_tools,
            "messages_json": json_formatted_messages,
            "response": output,
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

    # Write tasks_and_rewards to json file
    unix_timestamp = time.time()
    output_filename = f"{output_base_name}_{unix_timestamp}.json"
    output_path = f"output/{output_base_name}/{output_filename}"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(tasks_and_rewards, f, indent=2)

    print(f"Tasks and rewards data written to {output_path}")

    # upload file to s3
    bucket = "sn20"
    upload_file_to_s3(
        output_path, bucket, f"generated/{output_base_name}/{output_filename}"
    )

if __name__ == "__main__":
    while True:
        main()
