from prompts.react_2shot import REACT_SYS_PROMPT
from autogen import config_list_from_json, UserProxyAgent
from agents.evo_utils import EvolvableAgent

import json


def factual_exp_filter(json_content):
    # json_content.pop("llm_raw_answer", None)
    json_content.pop("checklist", None)
    return json_content


factual_exp_file = "C:/Sync_Code/MedSegAgent/ckpt/ReAct_2shot_train_set_100_gpt-4o-2024-08-06_factual_exp_suggested.jsonl"
factual_exp_info = [factual_exp_filter(json.loads(line)) for line in open(factual_exp_file)]


# print(factual_exp_info)
def working_exp_filter(json_content):
    return json_content.get("checklist", None)


working_exp_info = [working_exp_filter(json.loads(line)) for line in open(factual_exp_file)]
working_exp_info = [we for we in working_exp_info if (we)]
# working_exp_info = working_exp_info[:3]
# print(working_exp_info)
# print(len(working_exp_info))


def chat_with_llm_evo_react(query, model, autogen_config_path, sys_prompt_dict=REACT_SYS_PROMPT):
    autogen_config = config_list_from_json(env_or_file=autogen_config_path,
                                           filter_dict={"model": [model]})[0]
    user_proxy = UserProxyAgent("user_proxy", code_execution_config=False)
    expert_agent = EvolvableAgent(name="expert",
                                  system_message=sys_prompt_dict["Expert"],
                                  llm_config=autogen_config,
                                  human_input_mode="NEVER",
                                  max_exp_size=1000)

    for i, fexp in enumerate(factual_exp_info):
        expert_agent.update_factual_exp(f"case-{i}", str(fexp))
    for i, wexp in enumerate(working_exp_info):
        expert_agent.update_working_exp(f"exp-{i}", str(wexp))

    expert_agent.update_profile_with_exp(silent=True)

    chat_result = user_proxy.initiate_chat(expert_agent, message=query, max_turns=1)

    return {
        "reply": chat_result.summary,
        "chat_history": chat_result.chat_history,
    }
