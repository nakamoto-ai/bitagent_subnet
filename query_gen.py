from bitagent.tasks.tool_call_task import ToolCallTask
from bitagent.datasources import ToolDataset
from neurons.validator import Validator
from bitagent.protocol import QueryTask
import random
pass
import json

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
for _ in range(100):
    try:
        match choice:
            case "tool_call":
                tool_call_task = ToolCallTask(validator=val, name="Responds with correct function call", offline=False)
                task_data = tool_call_task.generate_task_data()
                tasks.append(tool_call_task)
                task_datas.append(task_data)

                syn.response = 'print("balls")' # TODO: generate a response with an llm?

                task_reward = tool_call_task.reward(validator=val, synapse=syn)
                task_rewards.append(task_reward)
                tasks_and_rewards.append({"task": tool_call_task, "response": syn.response, "reward": task_reward})

    except Exception as e:
        #bt.logging.warning(f'Error getting task (name {choice}): ', e)
        #bt.logging.warning(traceback.format_exc())
        print(f'Error getting task (name {choice}): ', e)



# Write tasks_and_rewards to json file
output_file = "tasks_and_rewards.json"
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
