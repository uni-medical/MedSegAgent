from prompts.assigner_experts import DEPARTMENT_EXPERTS_INFO, generate_expert_sys_prompt_dict, generate_doctor_expert_sys_prompt_dict
from prompts.data import DATASET_NAME_TO_PATH
from agents.single_turn import chat_with_llm_single_turn
from agents.doctor_expert import chat_with_llm_multi_turn_doctor_expert
from utils.reply_utils import extract_llm_answer
'''
The task is decouple into this workflow:
1. query - <Assigner> -> chosen <Expert>s
2. for each <Expert>, reuse the `doctor_expert` pattern to generate results
3. a general <Verifier> check the result with selected information (extract only related information from dataset.json) 
   and decided
'''


def chat_with_llm_assigner_expert_single_turn(query, model, autogen_config_path):
    profile_dataset_mapping = DEPARTMENT_EXPERTS_INFO
    ''' step1: assigner choose experts '''
    # skip now
    ''' step2: experts answer the question '''
    general_reply_list = []
    general_chat_history = []
    general_answer = dict()

    for profile, dataset_list in profile_dataset_mapping.items():
        sys_prompt_dict = generate_expert_sys_prompt_dict(profile, dataset_list)
        expert_res = chat_with_llm_single_turn(query,
                                               model,
                                               autogen_config_path,
                                               sys_prompt_dict=sys_prompt_dict)
        general_reply_list.append(expert_res["reply"])
        expert_res['chat_history'][-1]['role'] = profile.replace(".", "")
        general_chat_history.extend(expert_res['chat_history'])

        expert_answer = extract_llm_answer(expert_res["reply"])
        for dataset, labels in expert_answer.items():
            if (str(dataset) not in DATASET_NAME_TO_PATH):
                continue
            general_answer[dataset] = labels

    dataset_reply = ", ".join(list(general_answer.keys()))
    labels_reply = ", ".join([str(general_answer[gak]) for gak in list(general_answer.keys())])
    general_reply = "{\"dataset\": \"" + dataset_reply + "\",\"labels\": \"" + labels_reply + "\"}"
    general_chat_history.extend([
        {
            "role": "Summarizer",
            "content": general_reply
        },
    ])

    return {
        "reply": general_reply,
        "chat_history": general_chat_history,
    }


def chat_with_llm_assigner_expert_multi_turn_verifer(query, model, autogen_config_path):
    profile_dataset_mapping = DEPARTMENT_EXPERTS_INFO
    ''' step1: assigner choose experts '''
    # skip now
    ''' step2: experts answer the question '''
    general_reply_list = []
    general_chat_history = []
    general_answer = dict()

    from tenacity import retry, stop_after_attempt
    retry_decorator = retry(stop=stop_after_attempt(3))
    chat_func_with_retry = retry_decorator(chat_with_llm_multi_turn_doctor_expert)

    for profile, dataset_list in profile_dataset_mapping.items():
        sys_prompt_dict = generate_doctor_expert_sys_prompt_dict(profile, dataset_list)
        expert_res = chat_func_with_retry(query,
                                          model,
                                          autogen_config_path,
                                          system_prompt_dict=sys_prompt_dict)
        general_reply_list.append(expert_res["reply"])
        expert_res['chat_history'][-1]['role'] = profile.replace(".", "")
        general_chat_history.extend(expert_res['chat_history'])

        expert_answer = extract_llm_answer(expert_res["reply"])
        for dataset, labels in expert_answer.items():
            if (str(dataset) not in DATASET_NAME_TO_PATH):
                continue
            general_answer[dataset] = labels

    dataset_reply = ", ".join(list(general_answer.keys()))
    labels_reply = ", ".join([str(general_answer[gak]) for gak in list(general_answer.keys())])
    general_reply = "{\"dataset\": \"" + dataset_reply + "\",\"labels\": \"" + labels_reply + "\"}"
    general_chat_history.extend([
        {
            "role": "Summarizer",
            "content": general_reply
        },
    ])

    return {
        "reply": general_reply,
        "chat_history": general_chat_history,
    }
