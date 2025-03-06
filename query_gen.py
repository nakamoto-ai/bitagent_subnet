from bitagent.tasks.tool_call_task import ToolCallTask
from bitagent.datasources import ToolDataset
from neurons.validator import Validator
from bitagent.protocol import QueryTask
import random
pass
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

response_gen_model = "watt-ai/watt-tool-8B"
tokenizer = AutoTokenizer.from_pretrained(response_gen_model)
model = AutoModelForCausalLM.from_pretrained(response_gen_model, torch_dtype='auto', device_map='auto')
system_prompt ="""You are an expert in composing functions. You are given a question, or just general problem statement, and a set of functions you can use. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
You should only return the function call in tools call sections.
You MUST put the function call in the format of func_name1(params_name1=params_value1, params_name2=params_value2...)
You SHOULD NOT include any other text in the response.
MAKE SURE to use the correct function names, and double check the values passed to the function call for logic errors.
Here is a list of functions in JSON format that you can invoke.\n{functions}\n
"""

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

tasks = []
task_datas = []
task_rewards = []

tasks_and_rewards = []

# 5 sets of 10000 records
for i in range(5):
    for _ in range(10000):
        try:
            match choice:
                case "tool_call":
                    tool_call_task = ToolCallTask(validator=val, name="Responds with correct function call", offline=False)
                    task_data = tool_call_task.generate_task_data()
                    tasks.append(tool_call_task)
                    task_datas.append(task_data)


                    user_query = tool_call_task.messages[0].content
                    tools = tool_call_task.synapse.tools

                    messages = [
                        {'role': 'system', 'content': system_prompt.format(functions=tools)},
                        {'role': 'user', 'content': user_query}
                    ]
                    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

                    outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
                    output = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
                    syn.response = output

                    # [total_score, total_possible, results, correct_answer]
                    task_reward = tool_call_task.reward(validator=val, synapse=syn)
                    task_rewards.append(task_reward)
                    print(f"scored response: {task_reward[0]}/{task_reward[1]}")      
                    tasks_and_rewards.append({"task": tool_call_task, "response": syn.response, "reward": task_reward})

        except Exception as e:
            #bt.logging.warning(f'Error getting task (name {choice}): ', e)
            #bt.logging.warning(traceback.format_exc())
            print(f'Error getting task (name {choice}): ', e)



    # Write tasks_and_rewards to json file
    output_file = f"tasks_and_rewards_{i}.json"
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
            "value": float(item["reward"]) if hasattr(item["reward"], "__float__") else item["reward"]
        }

        # Create serializable entry
        entry = {
            "task": task_dict,
            "response": item["response"],
            "reward": reward_data
        }

        serializable_data.append(entry)

    # Write to file
    with open(output_file, 'w') as f:
        json.dump(serializable_data, f, indent=2)

    print(f"Tasks and rewards data written to {output_file}")
