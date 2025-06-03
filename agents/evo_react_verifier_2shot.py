from prompts.react_2shot import REACT_SYS_PROMPT
from prompts.verifier import VERIFIER_SYS_PROMPT
from autogen import config_list_from_json, UserProxyAgent
from agents.evo_utils import EvolvableAgent
from prompts.data import retrieve_dataset_label_info
from utils.reply_utils import extract_llm_answer

import json


def factual_exp_filter(json_content):
    json_content.pop("checklist", None)
    return json_content


factual_exp_file = "C:/Sync_Code/MedSegAgent/ckpt/ReAct_2shot_train_set_100_gpt-4o-2024-08-06_factual_exp_suggested.jsonl"
factual_exp_info = [factual_exp_filter(json.loads(line)) for line in open(factual_exp_file)]


def working_exp_filter(json_content):
    return json_content.get("checklist", None)


working_exp_info = [working_exp_filter(json.loads(line)) for line in open(factual_exp_file)]
working_exp_info = [we for we in working_exp_info if (we)]

# def chat_with_llm_evo_react(query, model, autogen_config_path, sys_prompt_dict=REACT_SYS_PROMPT):
#     autogen_config = config_list_from_json(env_or_file=autogen_config_path,
#                                            filter_dict={"model": [model]})[0]
#     user_proxy = UserProxyAgent("user_proxy", code_execution_config=False)
#     expert_agent = EvolvableAgent(name="expert",
#                                   system_message=sys_prompt_dict["Expert"],
#                                   llm_config=autogen_config,
#                                   human_input_mode="NEVER",
#                                   max_exp_size=1000)

#     for i, fexp in enumerate(factual_exp_info):
#         expert_agent.update_factual_exp(f"case-{i}", str(fexp))
#     expert_agent.update_profile_with_exp(silent=True)

#     chat_result = user_proxy.initiate_chat(expert_agent, message=query, max_turns=1)

#     return {
#         "reply": chat_result.summary,
#         "chat_history": chat_result.chat_history,
#     }


def chat_with_llm_evo_react_verifier(query,
                                     model,
                                     autogen_config_path,
                                     sys_prompt_dict=REACT_SYS_PROMPT):
    autogen_config = config_list_from_json(env_or_file=autogen_config_path,
                                           filter_dict={"model": [model]})[0]
    general_chat_history = []
    user_proxy = UserProxyAgent("user_proxy", code_execution_config=False)
    expert_agent = EvolvableAgent(name="expert",
                                  system_message=sys_prompt_dict["Expert"],
                                  llm_config=autogen_config,
                                  human_input_mode="NEVER",
                                  max_exp_size=1000)
    verifier_agent = EvolvableAgent(name="verifier",
                                    system_message=VERIFIER_SYS_PROMPT,
                                    llm_config=autogen_config,
                                    human_input_mode="NEVER",
                                    max_exp_size=1000)
    for i, fexp in enumerate(factual_exp_info):
        expert_agent.update_factual_exp(f"case-{i}", str(fexp))
    expert_agent.update_profile_with_exp(silent=True)

    # user -> expert
    expert_result = user_proxy.initiate_chat(expert_agent, message=query, max_turns=1)
    general_chat_history.extend(expert_result.chat_history)
    llm_answer = extract_llm_answer(expert_result.summary)

    # user -> verifier
    verf_query = f"LLM Answer:`{llm_answer}`\nReference Information:```\n"
    for dataset, labels in llm_answer.items():
        label_info = retrieve_dataset_label_info(dataset, labels)
        verf_query += f"{dataset}:{label_info}\n"
    verf_query += "```\nPlease verify the answer and return in the constrainted format."
    verf_result = user_proxy.initiate_chat(verifier_agent, message=verf_query, max_turns=1)
    general_chat_history.extend(verf_result.chat_history)

    # verifier -> expert
    final_ans_query = f"Verifier Comments:`{verf_result.summary}`\n Question:`{query}`\nPlease provide your new answer."
    ans_result = user_proxy.initiate_chat(expert_agent, message=final_ans_query, max_turns=1)
    general_chat_history.extend(ans_result.chat_history)

    return {
        "reply": ans_result.summary,
        "chat_history": general_chat_history,
    }
