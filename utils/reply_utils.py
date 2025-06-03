import json
import logging
import os
import re


def save_state(state, state_filename='task_state.json'):
    with open(state_filename, 'w') as f:
        json.dump(state, f)


def load_state(state_filename='task_state.json'):
    if os.path.exists(state_filename):
        with open(state_filename, 'r') as f:
            return json.load(f)
    return None


def clean_dataset_format(input_str):
    input_str = input_str.replace("'", "").replace("\"", "")
    if input_str.startswith("[") and input_str.endswith("]"):
        input_str = input_str[1:-1]
    return input_str


def clean_label_format(input_str):
    input_str = input_str.replace("'", "").replace("\"", "").strip()

    if input_str.startswith("'") and input_str.endswith("'"):
        input_str = input_str[1:-1]

    if input_str.startswith('"') and input_str.endswith('"'):
        input_str = input_str[1:-1]

    if input_str.startswith("[[") and input_str.endswith("]]"):
        input_str = input_str[1:-1]

    return input_str


def extract_llm_answer(input_str, key="Final Answer:"):
    # 1. 移除代码块标记
    input_str = input_str.replace("```json\n", "").replace("```", "").strip()

    if key in input_str:
        # 2. 分割字符串并获取关键字后的部分
        input_str = input_str.split(key)[-1].strip()

    # 替换 JSON 键的单引号为双引号
    input_str = re.sub(r"(?<=\{)\s*'(\w+)'\s*:", r'"\1":', input_str)
    input_str = re.sub(r",\s*'(\w+)'\s*:", r', "\1":', input_str)

    # 替换第一个值中的首尾单引号为双引号
    input_str = re.sub(r'"\s*:\s*\'(.*?)\'\s*(,|\})', r'": "\1"\2', input_str)

    # 替换 "answer" 字段中的所有单引号为双引号
    if '"answer":' in input_str:
        answer_match = re.search(r'("answer"\s*:\s*{.*?})', input_str, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1)
            # 将内容中的单引号替换为双引号
            answer_content = answer_content.replace("'", '"')
            input_str = input_str.replace(answer_match.group(1), answer_content)

    # 3. 使用正则表达式提取符合 {"thought": "", "answer": {}} 格式的 JSON 字符串
    json_pattern = r'\{\s*"thought"\s*:\s*"[^"]*",\s*"answer"\s*:\s*(\{.*?\}|\[.*?\]|".*?"|\d+)\s*\}'
    match = re.search(json_pattern, input_str, re.DOTALL)

    if match:
        json_str = match.group(0)
    else:
        # 如果未匹配到预期的 JSON 格式，使用整个 input_str
        json_str = input_str

    result = {}
    try:
        result = json.loads(json_str)
    except json.JSONDecodeError:
        logging.warning(f"无法将以下内容解析为 JSON: {json_str}")
        return {}

    # 4. 获取 'answer' 部分并返回整个字典
    answer = result.get('answer', {})

    return answer


def extract_ground_truth(test_case):
    expected_dataset_info = test_case.get('expected_datasets', {"None": []})
    return expected_dataset_info


if __name__ == "__main__":
    input_str = """
    ```
    Question: Please segment the left adrenal gland in CT images.
    Thought: I need to find CT datasets with annotations for the left adrenal gland.
    Action: Check datasets with CT modality for left adrenal gland labels.
    Action Input: Dataset information provided.
    Observation: Found datasets with left adrenal gland labels in CT.
    Thought: I now know the final answer.
    Final Answer: { "thought": "The left adrenal gland can be segmented using CT datasets. I found suitable annotations.", "dataset": "AMOS22_Task2, FLARE23, TotalSegmentator_v2", "labels": "[12], [8], [9]" }
    ```
    """
    res = extract_llm_answer(input_str)
    print(res)
