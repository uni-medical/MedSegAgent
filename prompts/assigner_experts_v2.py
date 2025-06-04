from prompts.data import DATASET_NAME_TO_PATH, get_info_from_json_path_list, DATASET_JSON_INFO
from prompts.react_2shot import SYSTEM_PROMPT_REACT
import json


# 加载专家配置文件
def load_expert_config(config_path):
    with open(config_path, 'r') as config_file:
        return json.load(config_file)


# 加载配置数据
config_data = load_expert_config('./prompts/expert_config.json')

# 定义 ASSIGNER_PROMPT 所需的格式化输出和示例
ASSIGNER_FORMATTED_OUTPUTS = '{"thought": "<Your brief justification>", "answer": ["WHOLEBODY", "EXPERT1", "EXPERT2"]}'

ASSIGNER_FORMATTED_EXAMPLES = '''
{"thought": "Liver segmentation requires both ABDOMINAL and WHOLEBODY expertise because the liver is an abdominal organ that may also appear in whole-body segmentation tasks.", "answer": ["WHOLEBODY", "ABDOMINAL"]}, 
{"thought": "Heart segmentation requires CARDIAC expertise because the heart is the primary focus of cardiac specialists. THORACIC expertise is necessary because the heart is located within the thoracic cavity and may involve surrounding thoracic structures. WHOLEBODY expertise is relevant since the heart can appear in whole-body scans.", "answer": ["WHOLEBODY", "CARDIAC", "THORACIC"]}, 
'''


# 获取选定数据集的信息
def get_selected_dataset_info(dataset_name_list):
    json_path_list = [DATASET_NAME_TO_PATH.get(dn, None) for dn in dataset_name_list]
    if None in json_path_list:
        missing_datasets = [
            dn for dn, path in zip(dataset_name_list, json_path_list) if path is None
        ]
        raise ValueError(f"Non-existent dataset names found: {missing_datasets}")
    return get_info_from_json_path_list(json_path_list)


# 创建一个格式化的共同经验字符串
common_experiences_section = "\n**Common Experiences**:\n" + "\n".join(
    f"- {exp}" for exp in config_data['common_experiences'])

# 更新 COMMON_EXPERT_PROMPT 以包含格式化的经验
COMMON_EXPERT_PROMPT = SYSTEM_PROMPT_REACT + common_experiences_section


# 创建每个专家的 prompt，包括他们的经验
def create_expert_prompt(dataset_name_list, experience_list, include_experiences=True):
    dataset_info = get_selected_dataset_info(dataset_name_list)
    if include_experiences:
        # 包含共同经验和专家独有经验
        complete_prompt = COMMON_EXPERT_PROMPT.replace(DATASET_JSON_INFO, dataset_info)
        if experience_list:
            experience_section = "\n**Expert Experience**:\n" + "\n".join(
                f"- {exp}" for exp in experience_list)
            complete_prompt += experience_section
    else:
        # 不包含任何经验
        complete_prompt = SYSTEM_PROMPT_REACT.replace(DATASET_JSON_INFO, dataset_info)
    return complete_prompt


# 根据配置数据生成每个专家的 prompt
expert_prompts_with_experience = {}
expert_prompts_no_experience = {}
for expert in config_data['experts']:
    expert_name = expert['name'].upper()
    dataset_list = expert['datasets']
    experience_list = expert.get('expert_experiences', [])
    # 生成包含经验的 prompt
    expert_prompts_with_experience[expert_name] = create_expert_prompt(dataset_list,
                                                                       experience_list,
                                                                       include_experiences=True)
    # 生成不包含经验的 prompt
    expert_prompts_no_experience[expert_name] = create_expert_prompt(dataset_list,
                                                                     experience_list,
                                                                     include_experiences=False)

# 为 ASSIGNER_PROMPT 准备专家的 profile
expert_profiles = []
for expert in config_data['experts']:
    expert_name = expert['name'].upper()
    profile = expert.get('profile', '')
    if profile:
        profile_text = f"**{expert_name}**: {profile}"
    else:
        profile_text = f"**{expert_name}**: No specific profile provided."
    expert_profiles.append(profile_text)

expert_profiles_text = "\n\n".join(expert_profiles)

# 更新 ASSIGNER_PROMPT，引用配置中的专家
expert_names = ', '.join([expert['name'].upper() for expert in config_data['experts']])

ASSIGNER_PROMPT = f"""
You are an AI expert assigner specializing in medical image segmentation tasks. Your responsibility is to select the most suitable specialist(s) based on anatomical regions and technical requirements. The specialists' areas of expertise include: {expert_names}.

**Experts' Profiles**:
{expert_profiles_text}

**Instructions**:

1. **Analyze Task Requirements**: Identify the specified anatomical region(s) (e.g., HEAD, ABDOMINAL) and required segmentation techniques or imaging modalities (e.g., MRI, CT). **If the anatomical region is not specified, assume that the task could involve any relevant regions where the structures might appear, and select all applicable specialists.**

2. **Match Expert Profiles**: Based on each expert's anatomical specialization, experience, and preferred modalities, select the most appropriate expert(s) for the task. **Always include WHOLEBODY expertise unless the task is exclusively limited to a specific region without any whole-body context.**

3. **Provide Brief Justification**: Briefly explain your choice, referencing each expert's specialization and limitations.

**Response Format**:
Use the following JSON format for Final Answer:
{ASSIGNER_FORMATTED_OUTPUTS}
Replace <Your brief justification> with your reasoning for selecting the expert(s).
Replace "WHOLEBODY", "EXPERT1", "EXPERT2" with the names of the assigned specialists.

Here are some examples:
{ASSIGNER_FORMATTED_EXAMPLES}
"""

# 使用包含经验的 expert_prompts_with_experience 字典
HEAD_EXPERT_PROMPT = expert_prompts_with_experience.get("HEAD", "")
CARDIAC_EXPERT_PROMPT = expert_prompts_with_experience.get("CARDIAC", "")
ABDOMINAL_EXPERT_PROMPT = expert_prompts_with_experience.get("ABDOMINAL", "")
THORACIC_EXPERT_PROMPT = expert_prompts_with_experience.get("THORACIC", "")
WHOLEBODY_EXPERT_PROMPT = expert_prompts_with_experience.get("WHOLEBODY", "")

# 使用不包含经验的 expert_prompts_no_experience 字典
HEAD_EXPERT_PROMPT_NO_EXPERIENCE = expert_prompts_no_experience.get("HEAD", "")
CARDIAC_EXPERT_PROMPT_NO_EXPERIENCE = expert_prompts_no_experience.get("CARDIAC", "")
ABDOMINAL_EXPERT_PROMPT_NO_EXPERIENCE = expert_prompts_no_experience.get("ABDOMINAL", "")
THORACIC_EXPERT_PROMPT_NO_EXPERIENCE = expert_prompts_no_experience.get("THORACIC", "")
WHOLEBODY_EXPERT_PROMPT_NO_EXPERIENCE = expert_prompts_no_experience.get("WHOLEBODY", "")
