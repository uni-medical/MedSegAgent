from autogen import ConversableAgent
from collections import OrderedDict


class EvolvableAgent(ConversableAgent):

    def __init__(self,
                 name=None,
                 system_message=None,
                 llm_config=None,
                 human_input_mode=None,
                 working_exp=None,
                 factual_exp=None,
                 max_exp_size=3,
                 *args,
                 **kwargs):
        super().__init__(name=name,
                         system_message=system_message,
                         llm_config=llm_config,
                         human_input_mode=human_input_mode,
                         *args,
                         **kwargs)
        self.max_exp_size = max_exp_size
        # Use OrderedDict to save both key-value pairs and maintain order
        # factual_exp: Example questions with corresponding answers
        # working_exp: Rules or guidelines for the agent to follow for better performance
        self.factual_exp = factual_exp or OrderedDict()
        self.working_exp = working_exp or OrderedDict()
        self.begin_flag = "Let’s Begin!"
        self.orig_sys_msg = system_message.replace(self.begin_flag, "")

    def _update_exp(self, exp, key, value):
        if len(exp) >= self.max_exp_size:
            # Remove the oldest entry to maintain the size limit
            oldest_key = next(iter(exp))
            del exp[oldest_key]
        exp[key] = value

    def update_factual_exp(self, key, value):
        self._update_exp(self.factual_exp, key, value)

    def update_working_exp(self, key, value):
        self._update_exp(self.working_exp, key, value)

    def update_profile_with_exp(self, silent=True):
        new_fact_exp = ""
        if self.factual_exp:
            factual_exp_str = "\n".join(
                [f"{i+1}. {value}" for i, (_, value) in enumerate(self.factual_exp.items())])
            new_fact_exp = f"Here are some examples of questions and answers based on your factual experience:\n{factual_exp_str}\n"

        new_work_exp = ""
        if self.working_exp:
            working_exp_str = "\n".join(
                [f"{i+1}. {value}" for i, (_, value) in enumerate(self.working_exp.items())])
            new_work_exp = f"Here are some rules or guidelines based on your working experience:\n{working_exp_str}\n"
        new_sys_msg = f"{self.orig_sys_msg}\n{new_fact_exp}{new_work_exp}\n{self.begin_flag}"

        # Update the system message with the new experience information
        self.update_system_message(new_sys_msg)
        if (not silent):
            print(f"System message updated with new experience information for agent {self.name}.")
            print(f"new System message:\n{self.system_message}")
