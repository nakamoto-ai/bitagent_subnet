from bitagent.tasks.tool_call_task import ToolCallTask
from bitagent.datasources import ToolDataset
from neurons.validator import Validator
import json
import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Evaluate tool call responses")
parser.add_argument(
    "--input_file",
    type=str,
    required=True,
    help="Path to the JSON file containing tool call responses to score"
)
parser.add_argument(
    "--output_file",
    type=str,
    default="accuracy_results.json",
    help="Path to output the scoring results"
)
args = parser.parse_args()

input_file = args.input_file
output_file = args.output_file

print(f"Loading responses from: {input_file}")
print(f"Results will be saved to: {output_file}")

class MockedValidator(Validator):
    def __init__(self):
        self.tool_dataset = ToolDataset()

    def validate(self, task):
        return True

val = MockedValidator()

# Load responses from input file
with open(input_file, 'r') as f:
    responses_data = json.load(f)

results = []
correct_count = 0
total_count = 0

# Process each response
for item in tqdm(responses_data, desc="Scoring responses"):
    try:
        # Create a tool call task
        tool_call_task = ToolCallTask(
            validator=val,
            name="Responds with correct function call",
            offline=True
        )

        # Set the task's tools and messages from the loaded data
        tool_call_task.synapse.tools = [type('Tool', (), t) for t in item["tools_json"]]

        # Set the expected tool call
        tool_call_task.expected_tool_call = item.get("expected_tool_call")

        # Set the response
        syn = tool_call_task.synapse
        syn.response = item["response"]

        # Score the response
        total_score, total_possible, detailed_results, correct_answer = tool_call_task.reward(validator=val, synapse=syn)

        # Determine if the response is correct (perfect score)
        is_correct = total_score == total_possible
        if is_correct:
            correct_count += 1
        total_count += 1

        # Store result
        result_item = {
            "response": item["response"],
            "expected_tool_call": tool_call_task.expected_tool_call,
            "total_score": total_score,
            "total_possible": total_possible,
            "is_correct": is_correct,
            "detailed_results": detailed_results
        }
        results.append(result_item)

    except Exception as e:
        print(f"Error scoring response: {e}")

# Calculate overall accuracy
accuracy = correct_count / total_count if total_count > 0 else 0
print(f"\nAccuracy: {accuracy:.2%} ({correct_count}/{total_count})")

# Save results
output_data = {
    "accuracy": accuracy,
    "correct_count": correct_count,
    "total_count": total_count,
    "detailed_results": results
}

os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"Results saved to {output_file}")
