from prompts.data import DATASET_JSON_INFO
from prompts.basic_v2 import SYSTEM_PROMPT_AI_EXPERT_V2

DOCTOR_SYS_PROMPT = f""""
You are a doctor who request a AI expert to select the correct tools (datasets and labels) to the segment requested targets on the medical image with specific modality. 
After the expert's response, you will verify if the labels of the selected datasets meets your request, and give your feedback.

The dataset informations are here: 
{DATASET_JSON_INFO}. 

Your request:
"""

EXPERT_SYS_PROMPT = SYSTEM_PROMPT_AI_EXPERT_V2.replace(
    "Let’s Begin!",
    "When you are given the feedback, you will adjust your response and return it with the correct format. Remember, ALL suitable datasets and corresponding labels should be selected. \n Let’s Begin!"
)

DOCTOR_EXPERT_SYS_PROMPT = {
    "Doctor": DOCTOR_SYS_PROMPT,
    "Expert": EXPERT_SYS_PROMPT,
}

if __name__ == "__main__":
    print(DOCTOR_EXPERT_SYS_PROMPT)
