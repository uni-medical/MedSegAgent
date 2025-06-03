from prompts.react_2shot import REACT_SYS_PROMPT
from prompts.react_2shot import SYSTEM_PROMPT_REACT
from autogen import config_list_from_json, UserProxyAgent, ConversableAgent
from utils.reply_utils import extract_llm_answer

from prompts.data import DATASET_JSON_INFO


FORMATTED_OUTPUT = '{ "thought":"<Short thought process>", "answer":{"AbdomenAtlasMini": [1], "AMOS22_Task2": [2,3], "FLARE23": [15]} }'

FORMATTED_EXAMPLES = '{ "thought":"I need to find all CT datasets containing duodenum. I should first check the modality, then the labels.", "answer":{"AbdomenAtlasMini":[13], "AMOS22_Task2":[12], "FLARE23":[19]} }\n{ "thought":"To segment adrenal gland in CT, I need to first find all dataset with adrenal gland.", "answer":{"AMOS22_Task2":[11,12], "FLARE23":[8,9], "TotalSegmentator_v2":[7,8]} }'

MISSING_MODALITY = '{ "thought":"No suitable modality", "answer": {} }'
NO_DATASET_FOUND = '{ "thought":"No suitable dataset", "answer": {} }'

SYSTEM_PROMPT_COT = f"""
# Role:
You are an AI expert on medical image segmentation, specializing in addressing clinical tasks efficiently. You will answer the input question as best you can.

## Skills:
1. Dataset & Label Selection
- Choose datasets and corresponding labels based on the input question.
- Use the provided dataset information to ensure compatibility.
- Validate the selected labels can cover the need to segment the requested targets.

## Constraints:
1. Response Format
- MUST use the following JSON format for Final Answer:
Final Answer: {FORMATTED_OUTPUT}
Here are some examples:
{FORMATTED_EXAMPLES}

2. Handling Missing Modalities
- If fail to find any needed modalities, reply in JSON: {MISSING_MODALITY}
- If no suitable dataset is available, reply in JSON: {NO_DATASET_FOUND}

## Reasoning (CoT):
To ensure accuracy and transparency, provide a step-by-step reasoning process for selecting datasets, labels, and handling constraints. This reasoning should reflect logical steps leading to the final JSON response.

Here's the dataset information:
{DATASET_JSON_INFO}

Let’s Begin!
"""



def chat_with_llm_cot(query, model, autogen_config_path, sys_prompt_dict=SYSTEM_PROMPT_COT):
    autogen_config = config_list_from_json(env_or_file=autogen_config_path,
                                           filter_dict={"model": [model]})[0]
    general_chat_history = []
    user_proxy = UserProxyAgent("user_proxy", code_execution_config=False)

    expert_agent = ConversableAgent(
        name=f"single_expert",
        system_message=SYSTEM_PROMPT_COT,
        llm_config=autogen_config,
        human_input_mode="NEVER",
    )

    expert_result = user_proxy.initiate_chat(expert_agent, message=query, max_turns=1)

    final_llm_answer = extract_llm_answer(expert_result.summary)

    return {
        "reply": final_llm_answer,
        "chat_history": general_chat_history,
    }
