import os.path as osp
from openai import OpenAI
from autogen import config_list_from_json
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
)
from agents import *
from agents.react_selfverify import chat_with_llm_react_selfverify
from agents.cot import chat_with_llm_cot
from agents.c2f_tree_search import chat_with_llm_c2f
from agents.c2f_tree_search2 import chat_with_llm_c2f2


AUTOGEN_CONFIG_PATH = osp.join(osp.dirname(osp.dirname(__file__)), "OAI_CONFIG_LIST")


def get_OAI_client_for_model(model, autogen_config_path=AUTOGEN_CONFIG_PATH):
    config = config_list_from_json(env_or_file=autogen_config_path,
                                   filter_dict={"model": [model, "default"]})[0]
    client = OpenAI(api_key=config['api_key'], base_url=config['base_url'])
    return client


def get_model_list():
    all_available_models = []
    for possible_model in ["default", "gpt-4o"]:
        client = get_OAI_client_for_model(possible_model)
        model_list = client.models.list()
        model_list = [model.id for model in model_list]
        all_available_models += model_list
    return all_available_models


def chat_with_llm(pattern, query, model="gpt-4o"):
    if pattern == "Native":
        res = chat_with_llm_native(query=query,
                                   model=model,
                                   autogen_config_path=AUTOGEN_CONFIG_PATH)
    elif pattern == "ReAct":
        res = chat_with_llm_react(query=query,
                                  model=model,
                                  autogen_config_path=AUTOGEN_CONFIG_PATH)
        
    elif pattern == "ReAct_SelfVerify":
        res = chat_with_llm_react_selfverify(query=query,
                                            model=model,
                                            autogen_config_path=AUTOGEN_CONFIG_PATH)

    elif pattern == "ReAct_MultiExpert":
        res = chat_with_llm_react_multiexpert(query=query,
                                              model=model,
                                              autogen_config_path=AUTOGEN_CONFIG_PATH)
    elif pattern == "ReAct_MultiExpert_Experience":
        res = chat_with_llm_react_multiexpert_experience(query=query,
                                                         model=model,
                                                         autogen_config_path=AUTOGEN_CONFIG_PATH)
    elif pattern == "ReAct_MultiExpert_SelfVerify":
        res = chat_with_llm_react_multiexpert_selfverify(query=query,
                                                         model=model,
                                                         autogen_config_path=AUTOGEN_CONFIG_PATH)
    elif pattern == "ReAct_MultiExpert_SelfVerify_Experience":
        res = chat_with_llm_react_multiexpert_selfverify_experience(
            query=query, model=model, autogen_config_path=AUTOGEN_CONFIG_PATH)
    elif pattern == "ReAct_MultiExpert_Experience":
        res = chat_with_llm_react_multiexpert_experience(query=query,
                                                         model=model,
                                                         autogen_config_path=AUTOGEN_CONFIG_PATH)
    elif pattern == "C2F":
        res = chat_with_llm_c2f(query=query, model=model, autogen_config_path=AUTOGEN_CONFIG_PATH)
    elif pattern == "C2F2":
        res = chat_with_llm_c2f2(query=query, model=model, autogen_config_path=AUTOGEN_CONFIG_PATH)
    elif pattern == "ReAct_RAG":
        res = chat_with_llm_react_rag(query=query,
                                      model=model,
                                      autogen_config_path=AUTOGEN_CONFIG_PATH)
    elif pattern == "CoT":
        res = chat_with_llm_cot(query=query, model=model, autogen_config_path=AUTOGEN_CONFIG_PATH)
    else:
        raise NotImplementedError(f"No pattern named '{pattern}'")
    return res
