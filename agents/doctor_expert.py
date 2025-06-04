from autogen import ConversableAgent, config_list_from_json
from prompts.basic_v2 import FORMATTED_OUTPUT, FORMATTED_EXAMPLES
from prompts.doctor_expert import DOCTOR_EXPERT_SYS_PROMPT


def chat_with_llm_multi_turn_doctor_expert(query,
                                           model,
                                           autogen_config_path,
                                           system_prompt_dict=DOCTOR_EXPERT_SYS_PROMPT):
    autogen_config = config_list_from_json(env_or_file=autogen_config_path,
                                           filter_dict={"model": [model]})[0]
    for agent_name in ["Doctor", "Expert"]:
        if agent_name not in system_prompt_dict:
            raise ValueError(f"Missing {agent_name} in system_prompt_dict")

    doctor_agent = ConversableAgent(
        name="Doctor",
        system_message=system_prompt_dict["Doctor"],
        llm_config=autogen_config,
    )
    expert_agent = ConversableAgent(
        name="Expert",
        system_message=system_prompt_dict["Expert"] + query,
        llm_config=autogen_config,
    )

    SUMMARY_PROMPT = f""" Summarize the final results of dataset and label selctions from the conversation.
    Please output using the following JSON format: ``` {FORMATTED_OUTPUT} ```
    Make sure that all suitable datasets and corresponding labels are chosen.
    Here are some examples:
    {FORMATTED_EXAMPLES}
    """
    # expert_agent (recipient) will summarize with the prompt
    chat_result = doctor_agent.initiate_chat(expert_agent,
                                             message=query,
                                             max_turns=2,
                                             summary_method="reflection_with_llm",
                                             summary_args={"summary_prompt": SUMMARY_PROMPT})

    return {
        "reply":
        chat_result.summary,
        "chat_history":
        chat_result.chat_history + [{
            "role": "Summarizer",
            "content": chat_result.summary
        }],
    }
