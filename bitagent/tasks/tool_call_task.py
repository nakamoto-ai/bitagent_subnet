# The MIT License (MIT)
# Copyright © 2023 RogueTensor

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import ast
import json
import random
import datetime
import bittensor as bt
from huggingface_hub import dataset_info
from bitagent.protocol import QueryTask
from bitagent.tasks import Task
from bitagent.tasks import TASK_WEIGHTS
from bitagent.schemas.chat import messages_to_list
from bitagent.datasources.tools import ToolDataset
from bitagent.datasources.tools import ToolCallData
from bitagent.helpers.tool_parsing import validate_tool_call, find_msgs_before_tool_call, find_first_tool_call
from bitagent.criteria import default_criteria, tool_call_criteria, irrelevant_tool_call_criteria

REWRITE_TOOL_USER_PROMPT = """You rewrite questions to make sense when paired with a function call.
The rewritten question will need to be changed to match the argument parameters and values relative to the function name.
You should change the phrasing of the question to be different and keeping aligned with the function name and arguments.
The capitalization of your user prompt rephrasasl should match the exact case of what is expected in the function call.
Your response should be the rewritten question only.\n
Function call:\n`{tool_call}`\n
Question: {user}\n
Modified Question: """

class ToolCallTask(Task):
    def __init__(
        self,
        validator,
        name: str,
        desc: str = "",
        offline: bool = False,
    ):
        super().__init__(name=name, desc=desc)
        self.validator = validator
        self.timeout = 15.0
        self.name += " - Tool Call"
        self.weight = TASK_WEIGHTS["tool_call"]

        self.expected_tool_call = None

        if offline:
            self.mode = "offline"
        messages = None
        for _ in range(10):
            try:
                messages, tools, data = self.generate_task_data()
                expected_messages = messages_to_list(data.messages)
                expected_tool_call_messages = [em for em in expected_messages if em['role'] == 'tool call']
                if messages[0].role == 'system':
                    # try again - skip tasks with system prompts
                    continue
                if len(expected_tool_call_messages) > 0:
                    expected_tool_call_message = expected_tool_call_messages[0]['content']
                else:
                    #bt.logging.debug(f"Skipping - no tool call message found in expected messages: {expected_messages}")
                    continue

                if type(expected_tool_call_message) == str:
                    expected_tool_call = json.loads(expected_tool_call_message)
                else:
                    expected_tool_call = expected_tool_call_message

                self.expected_tool_call = expected_tool_call
                self.criteria = default_criteria + tool_call_criteria(expected_response=expected_tool_call)

                break

            except Exception as e:
                bt.logging.debug(f'Exception getting new task - {e} - you may need to CHECK YOUR vLLM docker instance')
                pass
        if not messages:
            raise Exception(f"Failed to generate task data 10 times")
        self.messages = messages
        self.synapse = QueryTask(messages=messages, tools=tools)



    def generate_task_data(self) -> ToolCallData:

        data: ToolCallData = next(self.validator.tool_dataset)
        random.seed(572343)
        tool_call = find_first_tool_call(data.messages)
        if not tool_call:
            # no tool call in the messages, so skip
            raise Exception(f"Skipping - no tool call in the messages: {data.messages}")

        # increase number of tools
        for _ in range(random.randint(2,4)):
            # filter out the tools by name that are already in the data.tools
            new_tools = [t for t in next(self.validator.tool_dataset).tools if t.name not in [dt.name for dt in data.tools]]
            data.tools = data.tools + new_tools

        # remove all the messages after the first tool call, keeping the assistant
        # this reduces the number of messages needing rewording
        messages = data.messages
        filtered_msgs = []
        seen_tool_call = False
        for msg in messages:
            filtered_msgs.append(msg)
            if seen_tool_call: # want to do break after to include the assistant response
                break
            if msg.role == 'tool call':
                seen_tool_call = True
        data.messages = filtered_msgs

        user = data.messages[0].content

        count = 0
        while count < 10:
            count += 1
            if find_first_tool_call(data.messages):
                tool_call = find_first_tool_call(data.messages).content
                try: # check that the tool call can be loaded, and that it's valid
                    try:
                        if isinstance(tool_call, str):
                            new_tool_call = json.dumps(json.loads(tool_call))
                            tool_call_dict = json.loads(new_tool_call)
                        elif isinstance(tool_call, dict):
                            new_tool_call = tool_call
                            tool_call_dict = tool_call
                        else:
                            raise Exception(f'tool call is not a string or dict: {tool_call}')

                    except Exception as e:
                        # this usually happens when the json is not valid (single vs double quotes)
                        new_tool_call = json.dumps(ast.literal_eval(tool_call))
                        tool_call_dict = ast.literal_eval(tool_call)

                except Exception as e:
                    bt.logging.error(f'An error occured while rewriting the tool call {e} - you may need to CHECK YOUR vLLM docker instance')
                    count = 11
                    continue


                data.messages[0].content = user

                data = ToolCallData(messages=data.messages, tools=data.tools)
                messages_before_call = find_msgs_before_tool_call(data.messages)

            else:
                # no tool call in the messages, so skip
                raise Exception(f"Skipping - guess there was no tool call in the messages: {data.messages}")

            all_tools = data.tools
            random.shuffle(all_tools)
            return messages_before_call, all_tools, data

        raise Exception("Skipping - while loop ended without a tool call task")
