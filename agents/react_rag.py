from prompts.react_2shot import REACT_SYS_PROMPT
from prompts.react_2shot import SYSTEM_PROMPT_REACT
from prompts.rag import RAG_HELPER_SYS_PROMPT
from prompts.data import retrieve_dataset_from_label_name
from autogen import config_list_from_json, UserProxyAgent, ConversableAgent
from utils.reply_utils import extract_llm_answer


def chat_with_llm_react_rag(query, model, autogen_config_path, sys_prompt_dict=REACT_SYS_PROMPT):
    autogen_config = config_list_from_json(env_or_file=autogen_config_path,
                                           filter_dict={"model": [model]})[0]
    general_chat_history = []
    user_proxy = UserProxyAgent("user_proxy", code_execution_config=False)

    rag_helper = ConversableAgent(
        name=f"rag_helper",
        system_message=RAG_HELPER_SYS_PROMPT,
        llm_config=autogen_config,
        human_input_mode="NEVER",
    )

    expert_agent = ConversableAgent(
        name=f"single_expert",
        system_message=SYSTEM_PROMPT_REACT,
        llm_config=autogen_config,
        human_input_mode="NEVER",
    )

    rag_helper_result = user_proxy.initiate_chat(
        rag_helper, message=f"Please give one keyword for this query: {query}", max_turns=1)
    rag_keyword = rag_helper_result.summary.strip()
    retrieved_info = retrieve_dataset_from_label_name(rag_keyword, backend="spacy")

    expert_result = user_proxy.initiate_chat(
        expert_agent,
        message=
        f"Here are some more relevant datasets, please pay special attention to them:{retrieved_info}\n{query}",
        max_turns=1)

    final_llm_answer = extract_llm_answer(expert_result.summary)

    return {
        "reply": final_llm_answer,
        "chat_history": general_chat_history,
    }
