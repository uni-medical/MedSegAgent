from prompts.react_2shot import REACT_SYS_PROMPT
from prompts.react_2shot import SYSTEM_PROMPT_REACT
from autogen import config_list_from_json, UserProxyAgent, ConversableAgent
from utils.reply_utils import extract_llm_answer


def chat_with_llm_react_selfverify(query, model, autogen_config_path, sys_prompt_dict=REACT_SYS_PROMPT):
    autogen_config = config_list_from_json(env_or_file=autogen_config_path,
                                           filter_dict={"model": [model]})[0]
    general_chat_history = []
    user_proxy = UserProxyAgent("user_proxy", code_execution_config=False)

    expert_agent = ConversableAgent(
        name=f"single_expert",
        system_message=SYSTEM_PROMPT_REACT,
        llm_config=autogen_config,
        human_input_mode="NEVER",
    )

    # Get initial answer from expert
    expert_result = user_proxy.initiate_chat(expert_agent, message=query, max_turns=1)
    initial_llm_answer = extract_llm_answer(expert_result.summary)
    general_chat_history.extend(expert_result.chat_history)

    # Expert self-verifies their answer
    self_verify_query = (
        f"The question was: {query}\n"
        f"Your previous answer was: {initial_llm_answer}\n"
        "Please review your previous answer. Does it align with your knowledge and expertise?\n"
        "If your answer is correct, please confirm it. If there are any errors or omissions, please correct them in your reply."
    )
    
    self_verify_result = user_proxy.initiate_chat(expert_agent,
                                                  message=self_verify_query,
                                                  max_turns=1)
    final_llm_answer = extract_llm_answer(self_verify_result.summary)
    general_chat_history.extend(self_verify_result.chat_history)

    return {
        "reply": final_llm_answer,
        "chat_history": general_chat_history,
    }