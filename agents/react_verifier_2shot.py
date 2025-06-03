from prompts.react_2shot import REACT_SYS_PROMPT
from prompts.verifier import VERIFIER_SYS_PROMPT
from autogen import config_list_from_json, UserProxyAgent, ConversableAgent
from prompts.data import retrieve_dataset_label_info
from utils.reply_utils import extract_llm_answer


def chat_with_llm_react_verifier(query,
                                 model,
                                 autogen_config_path,
                                 sys_prompt_dict=REACT_SYS_PROMPT):
    autogen_config = config_list_from_json(env_or_file=autogen_config_path,
                                           filter_dict={"model": [model]})[0]
    general_chat_history = []
    user_proxy = UserProxyAgent("user_proxy", code_execution_config=False)
    expert_agent = ConversableAgent(name="expert",
                                    system_message=sys_prompt_dict["Expert"],
                                    llm_config=autogen_config,
                                    human_input_mode="NEVER")
    verifier_agent = ConversableAgent(name="verifier",
                                      system_message=VERIFIER_SYS_PROMPT,
                                      llm_config=autogen_config,
                                      human_input_mode="NEVER")

    # user -

    # user -> expert
    expert_result = user_proxy.initiate_chat(expert_agent, message=query, max_turns=1)
    general_chat_history.extend(expert_result.chat_history)
    llm_answer = extract_llm_answer(expert_result.summary)

    # user -> verifier
    verf_query = f"Question: {query}\nLLM Answer:{llm_answer}\nReference Information:\n"
    for dataset, labels in llm_answer.items():
        label_info = retrieve_dataset_label_info(dataset, labels)
        verf_query += f"{dataset}:{label_info}\n"
    verf_query += "Please verify the answer and return in the constrainted format."
    verf_result = user_proxy.initiate_chat(verifier_agent, message=verf_query, max_turns=1)
    general_chat_history.extend(verf_result.chat_history)

    # verifier -> expert
    final_ans_query = f"Verifier Comments:{verf_result.summary}\n Question:{query}\nPlease provide your new answer."
    ans_result = user_proxy.initiate_chat(expert_agent, message=final_ans_query, max_turns=1)
    general_chat_history.extend(ans_result.chat_history)

    return {
        "reply": ans_result.summary,
        "chat_history": general_chat_history,
    }
