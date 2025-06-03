from openai import OpenAI
from autogen import config_list_from_json


def chat_with_agent_single_turn(query, model, autogen_config_path, sys_prompt):
    config = config_list_from_json(env_or_file=autogen_config_path,
                                   filter_dict={"model": [model, "default"]})[0]
    client = OpenAI(api_key=config['api_key'], base_url=config['base_url'])
    chat_history = [{"role": "user", "content": query}]
    completion = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "system",
            "content": sys_prompt
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


class Agent:

    def __init__(self, name, system_message, llm_config, *args):
        """
        Initialize the Agent with a name, system message, and LLM configuration.

        :param name: The name of the agent.
        :param system_message: The system message or prompt for the agent.
        :param llm_config: Configuration for the language model.
        """
        self.name = name
        self.system_message = system_message
        self.llm_config = llm_config
        self.model = self.llm_config["model"]
        self.client = OpenAI(api_key=self.llm_config['api_key'],
                             base_url=self.llm_config['base_url'])

    def send(self, recipient, message):
        """
        Send a query to another agent.

        :param query: The query to send.
        :param recipient: The target agent to send the query to.
        :return: The generated response.
        """
        return recipient.process_query(message)

    def process_query(self, query):
        """
        Process a query received from another agent.

        :param query: The query to process.
        :return: The generated response.
        """
        response = self._generate_response(query)
        return response

    def _generate_response(self, query):
        """
        Generate a response using the LLM.

        :param query: The query to generate a response for.
        :return: The generated response.
        """
        chat_history = [{"role": "user", "content": query}]
        import pdb
        pdb.set_trace()
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "system",
                "content": self.system_message
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
