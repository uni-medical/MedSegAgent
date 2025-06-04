from autogen import config_list_from_json, ConversableAgent
from prompts.reflexion_2shot import REFLECTXION_SYS_PROMPT
from prompts.react_2shot import REACT_FORMATTED_EXAMPLES
from prompts.data import retrieve_dataset_label_info
from utils.reply_utils import extract_llm_answer

MAX_TRY = 2


def chat_with_llm_reflexion(query,
                            model,
                            autogen_config_path,
                            sys_prompt_dict=REFLECTXION_SYS_PROMPT):
    autogen_config = config_list_from_json(env_or_file=autogen_config_path,
                                           filter_dict={"model": [model]})[0]
    expert_agent = ConversableAgent(name="actor",
                                    system_message=sys_prompt_dict["Expert"],
                                    llm_config=autogen_config,
                                    human_input_mode="NEVER")

    eval_agent = ConversableAgent(name="evaluator",
                                  system_message=sys_prompt_dict["Evaluator"],
                                  llm_config=autogen_config,
                                  human_input_mode="NEVER")

    reflection_agent = ConversableAgent(name="self_reflection",
                                        system_message=sys_prompt_dict["Reflection"],
                                        llm_config=autogen_config,
                                        human_input_mode="NEVER")

    general_chat_history = []
    curr_query = query
    for i in range(MAX_TRY):
        # ask for first result
        actor_result = eval_agent.initiate_chat(expert_agent, message=curr_query, max_turns=1)
        general_chat_history += actor_result.chat_history
        actor_final_answer = actor_result.summary
        if ("Final Answer:" in actor_final_answer):
            actor_final_answer = actor_final_answer.split("Final Answer:")[-1].strip()

        # get eval
        llm_answer = extract_llm_answer(actor_final_answer)
        eval_query = f"Previous Answer:{actor_final_answer}\nVerified Labels:\n"
        for dataset, labels in llm_answer.items():
            eval_query += f"dataset:{dataset}," + "labels:{" + retrieve_dataset_label_info(
                dataset, labels) + "}\n"
        eval_query += "\nPlease evaluate."

        eval_result = expert_agent.initiate_chat(eval_agent, message=eval_query, max_turns=1)
        general_chat_history += eval_result.chat_history[1:]
        eval_final_result = eval_result.summary
        if ("Final Result: True" in eval_final_result):
            eval_final_result = "True"

        # eval think we get the right answer or upon the max try time
        if (eval_final_result == "True" or i == MAX_TRY - 1):
            return {
                "reply": actor_result.summary,
                "chat_history": general_chat_history,
            }

        # if not find the right answer
        # self-reflection for suggestion
        reflect_result = eval_agent.initiate_chat(reflection_agent,
                                                  message=f"result:{actor_result.summary}\n"
                                                  f"evaluation:{eval_result.summary}\n"
                                                  "Please give the suggestion.",
                                                  max_turns=1)
        general_chat_history += reflect_result.chat_history

        # generate new query
        suggestion = "No suggestion."
        if ("Final Suggestion:" in reflect_result.summary):
            suggestion = reflect_result.summary.split("Final Suggestion:")[-1].strip()
        curr_query = f"Task:{query}\nWrong Answer:{actor_final_answer}\nSuggestion for this task:'{suggestion}'\nHere's some examples: {REACT_FORMATTED_EXAMPLES}"

    return {
        "reply": actor_result.summary,
        "chat_history": general_chat_history,
    }
