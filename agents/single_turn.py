from prompts.basic_v2 import BASIC_V2_SYS_PROMPT
from openai import OpenAI
from autogen import config_list_from_json


def chat_with_llm_single_turn(query,
                              model,
                              autogen_config_path,
                              sys_prompt_dict=BASIC_V2_SYS_PROMPT):
    config = config_list_from_json(env_or_file=autogen_config_path,
                                   filter_dict={"model": [model, "default"]})[0]
    client = OpenAI(api_key=config['api_key'], base_url=config['base_url'])
    chat_history = [{"role": "user", "content": query}]
    completion = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "system",
            "content": sys_prompt_dict["Expert"]
        }, {
            "role": "user",
            "content": query
        }],
    )
    assistant_reply = completion.choices[0].message.content
    chat_history.append({"role": "assistant", "content": assistant_reply})
    return {
        "reply": assistant_reply,
        "chat_history": chat_history,
    }
