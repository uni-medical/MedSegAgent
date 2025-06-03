import json
import os.path as osp
from glob import glob

from autogen import config_list_from_json, UserProxyAgent, ConversableAgent


# Function to load JSON dataset information from a directory
def load_dataset_info(json_dir,
                      keys_to_keep=["name", "description", "input_modalities", "labels"],
                      file_list=None):
    """
    Complete list of keys: ["name", "description", "input_modalities", "labels", "models"]
    """
    json_info_list = []
    # Append .json extension to the filenames in the list
    if file_list is not None:
        file_list = {name + ".json" for name in file_list}

    for json_path in glob(osp.join(json_dir, "*.json")):
        file_name = osp.basename(json_path)

        # Check if the file is in the specified file list
        if file_list is None or file_name in file_list:
            with open(json_path, encoding='utf-8') as f:
                json_content = json.load(f)
                # Keep only specified keys
                filtered_content = {key: json_content.get(key) for key in keys_to_keep}
                json_info_list.append(json.dumps(filtered_content))

    return "\n".join(json_info_list)


# Function to extract JSON data from LLM output
def extract_json_from_output(llm_output, expected_type='list'):
    import re
    if expected_type == 'list':
        pattern = r"Final Answer:\s*(\[[\s\S]*?\])"
    elif expected_type == 'dict':
        pattern = r"Final Answer:\s*(\{[\s\S]*?\})"
    else:
        raise ValueError("expected_type must be 'list' or 'dict'")

    match = re.search(pattern, llm_output)
    if match:
        json_str = match.group(1)
        # Remove comments from JSON string
        json_str = re.sub(r'//.*', '', json_str)  # Remove single-line comments
        json_str = re.sub(r'/\*[\s\S]*?\*/', '', json_str)  # Remove multi-line comments
        try:
            json_data = json.loads(json_str)
            return json_data
        except json.JSONDecodeError as e:
            print("JSON decoding error:", e)
            print("Attempting to fix JSON string...")
            json_str_fixed = json_str.replace("'", '"')
            try:
                json_data = json.loads(json_str_fixed)
                return json_data
            except json.JSONDecodeError as e:
                print("Failed to fix JSON string.")
                return None
    else:
        print("No 'Final Answer' found in LLM output.")
        return None
    return [] if expected_type == 'list' else {}


PROJECT_DIR = osp.dirname(osp.dirname(__file__))
DATASET_DIR = osp.join(PROJECT_DIR, "dataset")

SYSTEM_PROMPT_QUERY_PARSER = """
# Role:
You are an AI assistant specialized in parsing queries related to medical image segmentation. Your task is to split the query into its imaging modality and segmentation target with as much specificity as possible.

## Skills:
1. Query Parsing:
- Identify and extract the imaging modality, including specific subtypes (e.g., CT contrast, CT non-contrast, MR ADC, MR FLAIR, etc.) from the query.
- If specific subtypes are not mentioned, extract only the main modality (e.g., CT, MR).
- Identify and extract the segmentation target (e.g., brain, liver, tumor, etc.).

2. Thought Process:
Use this format for reasoning:
Question: the input question
Thought: analyze the input query to extract "modality" and "target".
Action: return a dictionary with the parsed components.
Observation: the parsed components.

Final Answer: a dictionary in the format:
{"modality": "modality_name", "target": "segmentation_target"}
"""

FORMATTED_OUTPUT_DATASET_LIST = '["AbdomenAtlasMini", "AMOS22_Task2", "FLARE23"]'
FORMATTED_OUTPUT_DATASET_VALUE_DICT = '{"AbdomenAtlasMini": [1], "AMOS22_Task2": [2, 3], "FLARE23": [15]}'

SYSTEM_PROMPT_MODALITY_FILTER = f"""
# Role:
You are an AI assistant specializing in medical image segmentation datasets. Your task is to filter relevant datasets based on the specified imaging modalities.

## Skills:
1. Dataset Selection by Modality:
- Match datasets using synonyms, abbreviations, and partial matches.
- Handle composite modalities like "MR or CT", treating them as both "MR" and "CT".
- Include variants such as "head CT", "non-contrast CT", or "contrast CT" under "CT".
- Perform case-insensitive comparisons.

2. Thought Process:
Use this format for reasoning:
Modality Filter: the input filter condition specifying imaging modalities.
Thought: analyze the input modalities and match them with the datasets' modalities.
Action: list datasets that match the specified modalities.
Observation: datasets that match the filter.

Final Answer: a list of matching datasets in valid JSON format without using code blocks.

## Response Format:
- Return the answer as a JSON list of dataset names, e.g., ["Dataset1", "Dataset2"]
- Answer after **Final Answer**

## Edge Cases:
- If no relevant datasets are found, return an empty list: []

Here is the dataset information (only name and input modalities are provided):
"""

SYSTEM_PROMPT_RELEVANCE_FILTER = f"""
# Role:
You are an AI assistant specializing in medical image segmentation datasets. Your task is to filter datasets based on their labels for relevance to the specified segmentation target.

## Skills:
1. Dataset Relevance Filtering:
- Analyze the labels of the datasets.
- Identify datasets where a single label or a **combination of labels** semantically match the segmentation target, regardless of word order.
- Consider combinations where the label names contain all necessary components to form the target, even if the order differs.
- **Do not** include datasets where labels only partially match or where combinations do not fully represent the exact target.

2. Thought Process:
You MUST use this format to think which datasets and their labels should be chosen:
```
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take
Action Input: the input to the action
Observation: the result of the action
... (this process can repeat multiple times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question in JSON format (Looking for examples below)
```

Final Answer: a list of relevant datasets in valid JSON format without using code blocks.

## Response Format:
- Return the answer as a JSON list:
{FORMATTED_OUTPUT_DATASET_LIST}

## Edge Cases:
- If no relevant datasets within provided datasets are found, return an empty list: []

Here is the dataset information:
"""

SYSTEM_PROMPT_LABEL_SELECTION = f"""
# Role:
You are an AI expert in medical image segmentation. Your task is to match datasets and their relevant labels based on the specified segmentation target.

## Skills:
1. Dataset & Label Selection
- Choose datasets and corresponding labels based on the input question.
- Use the provided dataset information to ensure compatibility.
- Validate the selected labels can cover the need to segment the requested targets.

2. Thought Process:
You MUST use this format to think which datasets and their labels should be chosen:
```
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take
Action Input: the input to the action
Observation: the result of the action
... (this process can repeat multiple times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question in JSON format (Looking for examples below)
```

## Response Format:
- Return the answer as a dictionary:
{FORMATTED_OUTPUT_DATASET_VALUE_DICT}

## Edge Cases:
- If no relevant datasets or labels are found, return an empty dictionary.

Here is the dataset information:
"""

AUTOGEN_CONFIG_PATH = osp.join(osp.dirname(osp.dirname(__file__)), "OAI_CONFIG_LIST")


def chat_with_llm_c2f1(query, model, autogen_config_path):
    """
    query: A natural language string specifying the desired segmentation task.
    model: Name of the LLM model to use.
    autogen_config_path: Path to the configuration file for the autogen agent.
    """
    autogen_config = config_list_from_json(env_or_file=autogen_config_path,
                                           filter_dict={"model": [model]})[0]
    user_proxy = UserProxyAgent("user_proxy", code_execution_config=False)

    # Step 1: Query Parsing Agent
    query_parser_agent = ConversableAgent(
        name="query_parser_agent",
        system_message=SYSTEM_PROMPT_QUERY_PARSER,
        llm_config=autogen_config,
        human_input_mode="NEVER",
    )

    max_attempts = 3
    attempt = 0
    while attempt < max_attempts:
        query_parsing_result = user_proxy.initiate_chat(query_parser_agent,
                                                        message=query,
                                                        max_turns=1)

        # 从查询解析器的输出中提取 JSON
        parsed_query = extract_json_from_output(query_parsing_result.summary, expected_type='dict')

        if parsed_query is not None:
            # 成功解析，退出循环
            break

        attempt += 1

    if not parsed_query:
        print("Failed to parse query. Exiting.")
        return {}

    modality = parsed_query.get("modality")
    target = parsed_query.get("target") or parsed_query.get("targets")
    target = str(target) if target is not None and not isinstance(target, str) else target
    if not modality or not target:
        print("Parsed query is incomplete. Exiting.")
        return {}

    # Step 2: Modality Filtering Agent

    # Get all dataset names as a set
    all_dataset_names = set(
        [osp.splitext(osp.basename(path))[0] for path in glob(osp.join(DATASET_DIR, "*.json"))])

    # Initialize selected and unselected datasets
    modality_filtered_datasets = set()
    unselected_datasets = all_dataset_names.copy()

    max_iterations = 3
    iteration = 0

    modality_messages = [
            f"I want to segment that a image of the modality {modality} image", f"The image target modality is {modality}",
            f"Please segment the {modality}  modality image"
        ]

    while iteration < max_iterations and unselected_datasets:
        iteration += 1
        print(f"Modality filtering iteration {iteration}")

        # Load modality info for unselected datasets
        unselected_dataset_info_modality = load_dataset_info(
            DATASET_DIR, keys_to_keep=["name", "input_modalities"], file_list=unselected_datasets)

        # Create a new system prompt with updated dataset info
        current_system_prompt = SYSTEM_PROMPT_MODALITY_FILTER + unselected_dataset_info_modality

        # Create a new agent with the updated system prompt
        modality_filter_agent = ConversableAgent(
            name="modality_filter_agent",
            system_message=current_system_prompt,
            llm_config=autogen_config,
            human_input_mode="NEVER",
        )

        max_attempts = 3
        
        attempt = 0
        while attempt < max_attempts:
            modality_message = modality_messages[attempt+iteration-1] if attempt+iteration-1 < len(
                modality_messages) else modality_messages[-1]
            # Use the agent to filter datasets based on modality
            modality_filter_result = user_proxy.initiate_chat(modality_filter_agent,
                                                              message=modality_message,
                                                              max_turns=1)
            # Extract the newly selected datasets
            newly_selected_datasets_raw = extract_json_from_output(modality_filter_result.summary,
                                                                   expected_type='list')
            if newly_selected_datasets_raw is not None:
                break
            attempt += 1

        if newly_selected_datasets_raw is None:
            continue

        # Process the list to extract dataset names
        newly_selected_datasets = set()
        for item in newly_selected_datasets_raw:
            if isinstance(item, dict):
                if 'name' in item:
                    newly_selected_datasets.add(item['name'])
                else:
                    print("Warning: Dictionary item without 'name' key:", item)
            elif isinstance(item, str):
                newly_selected_datasets.add(item)
            else:
                print("Warning: Unexpected item type:", type(item), item)

        # If no new datasets are selected, exit the loop
        if not newly_selected_datasets:
            break

        # Update selected and unselected datasets
        modality_filtered_datasets.update(newly_selected_datasets)
        unselected_datasets -= newly_selected_datasets

    # 循环结束后，打印所有已选择的数据集
    print("\nFinal selected datasets after modality filtering:")
    print(sorted(modality_filtered_datasets))
    print()

    # Check if any datasets have been selected
    if not modality_filtered_datasets:
        print("No datasets match the modality filter after multiple iterations. Exiting.")
        return {}

    # Convert the set to a list for subsequent steps
    modality_filtered_datasets = list(modality_filtered_datasets)

    # Step 3: Relevance Filtering Agent

    # Initialize selected and unselected datasets for relevance filtering
    relevance_filtered_datasets = set()
    unselected_datasets = set(modality_filtered_datasets)

    max_iterations = 3
    iteration = 0
    target_messages = [
            f"I want to segment the {target} in the image", f"The segmentation target is {target}", f"Please segment {target}",
            f"What is the anatomical region of the {target} in the body, and can you provide datasets related to that region?"
        ]
    while iteration < max_iterations and unselected_datasets:

        target_message = target_messages[iteration] if iteration < len(
                target_messages) else target_messages[-1]

        iteration += 1
        print(f"Relevance filtering iteration {iteration}")

        # Load dataset info for unselected datasets
        unselected_dataset_info = load_dataset_info(DATASET_DIR,
                                                    keys_to_keep=["name", "labels"],
                                                    file_list=unselected_datasets)

        # Create a new system prompt with updated dataset info
        current_system_prompt = SYSTEM_PROMPT_RELEVANCE_FILTER + unselected_dataset_info

        # Create a new agent with the updated system prompt
        relevance_filter_agent = ConversableAgent(
            name="relevance_filter_agent",
            system_message=current_system_prompt,
            llm_config=autogen_config,
            human_input_mode="NEVER",
        )

        max_attempts = 3
        attempt = 0
        while attempt < max_attempts:
            target_message = target_messages[(iteration + attempt) % len(target_messages)]
            # Use the agent to filter datasets based on relevance
            relevance_filter_result = user_proxy.initiate_chat(relevance_filter_agent,
                                                               message=target_message,
                                                               max_turns=1)

            # Extract the newly selected datasets
            newly_selected_datasets_raw = extract_json_from_output(relevance_filter_result.summary,
                                                                   expected_type='list')

            if newly_selected_datasets_raw is not None:
                break
            attempt += 1

        # Process the list to extract dataset names
        newly_selected_datasets = set()
        for item in newly_selected_datasets_raw:
            if isinstance(item, dict):
                if 'name' in item:
                    newly_selected_datasets.add(item['name'])
                else:
                    print("Warning: Dictionary item without 'name' key:", item)
            elif isinstance(item, str):
                newly_selected_datasets.add(item)
            else:
                print("Warning: Unexpected item type:", type(item), item)

        # If no new datasets are selected, exit the loop
        if not newly_selected_datasets and iteration > 1:
            break

        # Update selected and unselected datasets
        relevance_filtered_datasets.update(newly_selected_datasets)
        unselected_datasets -= newly_selected_datasets

    # Check if any datasets have been selected
    if not relevance_filtered_datasets:
        print("No datasets match the relevance filter after multiple iterations. Exiting.")
        return {}

    # 循环结束后，打印所有已选择的数据集
    print("\nFinal selected datasets after target filtering:")
    print(sorted(relevance_filtered_datasets))
    print()

    # Convert the set to a list for subsequent steps
    relevance_filtered_datasets = list(relevance_filtered_datasets)

    # Step 4: Label Selection Agent

    # 修改标签选择部分的逻辑，模仿相关性过滤的迭代过程
    label_selected_labels = {}
    unselected_datasets = set(relevance_filtered_datasets)
    max_iterations = 2  # 一轮选择 + 一轮检查
    iteration = 0

    # 定义标签选择的消息
    label_selection_messages = [
        f"I want to segment the {target} in the {modality} image",
        f"The segmentation is {target} in {modality}",
        f"Please segment the {target} in the {modality} images"
    ]

    while iteration < max_iterations and unselected_datasets:
        iteration += 1
        print(f"Label selection iteration {iteration}")

        # 加载当前未选数据集的详细信息
        current_dataset_info = load_dataset_info(DATASET_DIR,
                                                 keys_to_keep=["name", "input_modalities", "labels"],
                                                 file_list=unselected_datasets)

        # 创建新的系统提示，包含当前数据集信息
        current_system_prompt = SYSTEM_PROMPT_LABEL_SELECTION + current_dataset_info

        # 创建新的标签选择代理
        label_selection_agent = ConversableAgent(
            name="label_selection_agent",
            system_message=current_system_prompt,
            llm_config=autogen_config,
            human_input_mode="NEVER",
        )

        # 发送消息并尝试获取标签选择结果
        attempt_inner = 0
        while attempt_inner < max_attempts:
            # 确保索引不会越界
            label_message_index = attempt_inner + iteration - 1 if (attempt_inner + iteration - 1) < len(label_selection_messages) else -1
            label_message = label_selection_messages[label_message_index] if label_message_index >= 0 else label_selection_messages[-1]
            label_selection_result = user_proxy.initiate_chat(label_selection_agent,
                                                              message=label_message,
                                                              max_turns=1)

            # 从标签选择代理的输出中提取 JSON
            dataset_label_dict = extract_json_from_output(label_selection_result.summary,
                                                          expected_type='dict')

            if dataset_label_dict is not None:
                break
            attempt_inner += 1

        if dataset_label_dict is None or not all(
            isinstance(labels, list) 
            and all(isinstance(x, (int, float)) for x in labels)
            for labels in dataset_label_dict.values()
        ):
            continue

        # 合并已选择的标签
        for dataset, labels in dataset_label_dict.items():
            if dataset in label_selected_labels:
                # 合并标签列表，避免重复
                existing_labels = set(label_selected_labels[dataset])
                new_labels = set(labels)
                combined_labels = existing_labels.union(new_labels)
                label_selected_labels[dataset] = list(combined_labels)
            else:
                label_selected_labels[dataset] = labels

        # 移除已选择的标签的数据集
        unselected_datasets -= set(dataset_label_dict.keys())

    return label_selected_labels


def main():
    result = chat_with_llm_c2f("I want to segment the liver vessel in this CT image", "claude-3-5-haiku-20241022", AUTOGEN_CONFIG_PATH)
    print(result)


if __name__ == "__main__":
    main()