import json
import os.path as osp
import logging
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
            logging.error("JSON decoding error:", e)
            logging.info("Attempting to fix JSON string...")
            json_str_fixed = json_str.replace("'", '"')
            try:
                json_data = json.loads(json_str_fixed)
                return json_data
            except json.JSONDecodeError as e:
                logging.error("Failed to fix JSON string.")
                return None
    else:
        logging.error("No 'Final Answer' found in LLM output.")
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

## Constraints:
1. Response Format
- MUST use the following JSON format for the Final Answer:
Final Answer: {"modality": "modality_name", "target": "segmentation_target"}
Here are some examples:
{"modality": "CT", "target": "liver"}
{"modality": "MR", "target": "spleen"}
{"modality": "MR ADC", "target": "brain tumor"}
"""

FORMATTED_OUTPUT_DATASET_LIST = '["AbdomenAtlasMini", "AMOS22_Task2", "FLARE23"]'
FORMATTED_OUTPUT_DATASET_VALUE_DICT = '{"AbdomenAtlasMini": [1], "AMOS22_Task2": [2, 3], "FLARE23": [15]}'

SYSTEM_PROMPT_MODALITY_FILTER = f"""
# Role:
You are an AI assistant specializing in medical image segmentation datasets. Your task is to filter relevant datasets based on the specified imaging modalities.

## Skills:
1. Dataset Selection by Modality:
- Accurately identify datasets that match the specified imaging modalities.
- Handle composite modalities like "MR or CT", treating them as both "MR" and "CT".
- Include variants such as "head CT", "non-contrast CT", or "contrast CT" under "CT".

## Constraints:
1. Response Format
- MUST use the following JSON format for the Final Answer:
Final Answer: ["Dataset1", "Dataset2"]


## Edge Cases:
- If no datasets match the specified imaging modalities are found, return an empty list: []

Here is the dataset information (only name and input modalities are provided):
"""

SYSTEM_PROMPT_RELEVANCE_FILTER = f"""
# Role:
You are an AI assistant specializing in the task of medical image segmentation datasets. Your task is to filter datasets based on their labels for relevance to the specified segmentation target.

## Skills:
1. Dataset Relevance Filtering:
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
Final Answer: ["Dataset1", "Dataset2"]
```

## Constraints:
1. Response Format
- MUST use the following JSON format for the Final Answer:
Final Answer: ["Dataset1", "Dataset2"]


## Edge Cases:
- If no relevant datasets are found, return an empty list: []

Here is the dataset information (only name and labels are provided):
"""

SYSTEM_PROMPT_LABEL_SELECTION = """
# Role:
You are an AI expert in the task of medical image segmentation. Your task is to match datasets and their relevant labels based on the specified segmentation target.

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
Final Answer: {"Dataset1": [1, 2], "Dataset2": [3]}
```

## Constraints:
1. Response Format
- MUST use the following JSON format for the Final Answer:
Final Answer: {"Dataset1": [1, 2], "Dataset2": [3]}

## Edge Cases:
- If no relevant datasets or labels are found, return an empty dictionary: {}

Here is the dataset information:
"""

AUTOGEN_CONFIG_PATH = osp.join(osp.dirname(osp.dirname(__file__)), "OAI_CONFIG_LIST")


def chat_with_llm_c2f2(query, model, autogen_config_path):
    """
    query: A natural language string specifying the desired segmentation task.
    model: Name of the LLM model to use.
    autogen_config_path: Path to the configuration file for the autogen agent.
    """
    autogen_config = config_list_from_json(env_or_file=autogen_config_path,
                                           filter_dict={"model": [model]})[0]
    user_proxy = UserProxyAgent("user_proxy", code_execution_config=False)

    logging.info("步骤 1: 查询解析代理")
    # Step 1: Query Parsing Agent
    query_parser_agent = ConversableAgent(
        name="query_parser_agent",
        system_message=SYSTEM_PROMPT_QUERY_PARSER,
        llm_config=autogen_config,
        human_input_mode="NEVER",
    )

    # 定义不同的查询前缀
    query_prefixes = [
        "My query is: ",
        "I have a question: ",
        "Can you help with: "
    ]
    max_attempts = 3
    attempt = 0
    parsed_query = None  # 初始化 parsed_query

    while attempt < max_attempts:
        prefix = query_prefixes[attempt % len(query_prefixes)]
        message = prefix + query
        query_parsing_result = user_proxy.initiate_chat(
            query_parser_agent,
            message=message,
            max_turns=1
        )

        # 从查询解析器的输出中提取 JSON
        temp_parsed_query = extract_json_from_output(query_parsing_result.summary, expected_type='dict')

        if temp_parsed_query:
            modality = temp_parsed_query.get("modality")
            target = temp_parsed_query.get("target") or temp_parsed_query.get("targets")
            if modality and target:
                parsed_query = temp_parsed_query
                logging.info(f"Query parsed successfully: {parsed_query}")
                break
            else:
                logging.warning(f"Attempt {attempt + 1}: Parsed query missing 'modality' or 'target'.")
        else:
            logging.warning(f"Attempt {attempt + 1} failed to parse query.")

        attempt += 1

    if not parsed_query:
        logging.error("Failed to parse query. Exiting.")
        return {}

    # 确保 target 是字符串
    modality = parsed_query.get("modality")
    target = parsed_query.get("target") or parsed_query.get("targets")
    target = str(target) if target is not None and not isinstance(target, str) else target
    if not modality or not target:
        plogging.error("Parsed query is incomplete. Exiting.")
        return {}

    logging.info("步骤 2: 对数据集针对模态进行过滤")
    # Step 2: Modality Filtering Agent

    # 获取所有数据集名称作为集合
    all_dataset_names = set(
        [osp.splitext(osp.basename(path))[0] for path in glob(osp.join(DATASET_DIR, "*.json"))]
    )
    # 初始化已选择和未选择的数据集
    modality_filtered_datasets = set()
    unselected_datasets = all_dataset_names.copy()

    max_iterations = 3
    iteration = 0

    # 定义不同的模态过滤消息
    modality_messages = [
        f"I would like to segment an image with the modality {modality}.",
        f"The target modality of the image is {modality}.",
        f"Please segment the image with modality {modality}."
    ]

    while iteration < max_iterations and unselected_datasets:
        
        logging.info(f"Modality filtering iteration {iteration+1}")

        # 加载未选择数据集的模态信息
        unselected_dataset_info_modality = load_dataset_info(
            DATASET_DIR, keys_to_keep=["name", "input_modalities"], file_list=unselected_datasets
        )

        # 创建包含更新后的数据集信息的系统提示
        current_system_prompt = SYSTEM_PROMPT_MODALITY_FILTER + unselected_dataset_info_modality

        # 创建新的模态过滤代理
        modality_filter_agent = ConversableAgent(
            name="modality_filter_agent",
            system_message=current_system_prompt,
            llm_config=autogen_config,
            human_input_mode="NEVER",
        )

        max_attempts = 3
        attempt = 0
        newly_selected_datasets_raw = None
        while attempt < max_attempts:
            modality_message = modality_messages[(iteration+attempt) % len(modality_messages)]
            modality_filter_result = user_proxy.initiate_chat(modality_filter_agent,
                                                              message=modality_message,
                                                              max_turns=1)
            newly_selected_datasets_raw = extract_json_from_output(modality_filter_result.summary,
                                                                   expected_type='list')
            if isinstance(newly_selected_datasets_raw, list) and all(isinstance(item, str) for item in newly_selected_datasets_raw):
                break

            attempt += 1
            
        if not (isinstance(newly_selected_datasets_raw, list) and all(isinstance(item, str) for item in newly_selected_datasets_raw)):
            return {}

        newly_selected_datasets = set(newly_selected_datasets_raw)
        
        if not newly_selected_datasets:
            break
        
        modality_filtered_datasets.update(newly_selected_datasets)
        unselected_datasets -= newly_selected_datasets

        iteration += 1

    # 检查是否有任何数据集被选中
    if not modality_filtered_datasets:
        logging.error("No datasets match the modality filter after multiple iterations. Exiting.")
        return {}

    # 循环结束后，打印所有已选择的数据集
    logging.info("Final selected datasets after modality filtering:")
    logging.info(sorted(modality_filtered_datasets))

    logging.info("步骤 3: 对数据集针对类别进行过滤")
    # Step 3: Relevance Filtering Agent

    # 初始化已选择和未选择的数据集
    relevance_filtered_datasets = set()
    unselected_datasets = set(modality_filtered_datasets)

    max_iterations = 2
    iteration = 0

    # 定义不同的相关性过滤消息
    target_messages = [
        f"I'm interested in segmenting the {target}.",
        f"The segmentation target is {target}.",
        f"I want to segment the {target} in the image."
    ]
    while iteration < max_iterations and unselected_datasets:

        logging.info(f"Relevance filtering iteration {iteration+1}")

        # 加载未选择数据集的相关信息
        unselected_dataset_info = load_dataset_info(
            DATASET_DIR, 
            keys_to_keep=["name", "labels"], 
            file_list=unselected_datasets
        )

        # 创建包含更新后的数据集信息的系统提示
        current_system_prompt = SYSTEM_PROMPT_RELEVANCE_FILTER + unselected_dataset_info

        relevance_filter_agent = ConversableAgent(
            name="relevance_filter_agent",
            system_message=current_system_prompt,
            llm_config=autogen_config,
            human_input_mode="NEVER",
        )

        max_attempts = 3
        attempt = 0
        newly_selected_datasets_raw = None

        while attempt < max_attempts:
            target_message = target_messages[(iteration + attempt) % len(target_messages)]
            relevance_filter_result = user_proxy.initiate_chat(
                relevance_filter_agent,
                message=target_message,
                max_turns=1
            )

            newly_selected_datasets_raw = extract_json_from_output(
                relevance_filter_result.summary,
                expected_type='list'
            )

            if isinstance(newly_selected_datasets_raw, list) and all(isinstance(item, str) for item in newly_selected_datasets_raw):
                break

            attempt += 1

        if not (isinstance(newly_selected_datasets_raw, list) and all(isinstance(item, str) for item in newly_selected_datasets_raw)):
            return {}

        newly_selected_datasets = set(newly_selected_datasets_raw)

        if not newly_selected_datasets and iteration > 0:
            break

        # 更新已选择和未选择的数据集
        relevance_filtered_datasets.update(newly_selected_datasets)
        unselected_datasets -= newly_selected_datasets

        iteration += 1

    # 检查是否有任何数据集被选中
    if not relevance_filtered_datasets:
        logging.error("No datasets match the relevance filter after multiple iterations. Exiting.")
        return {}

    # 循环结束后，打印所有已选择的数据集
    logging.info("Final selected datasets after relevance filtering:")
    logging.info(sorted(relevance_filtered_datasets))

    logging.info("步骤 4: 选择数据集和类别")
    # Step 4: Label Selection Agent

    # 修改标签选择部分的逻辑，模仿相关性过滤的迭代过程
    label_selected_labels = {}
    unselected_datasets = relevance_filtered_datasets
    max_iterations = 1  # 一轮选择 + 一轮检查
    iteration = 0

    # 定义标签选择的消息
    label_selection_messages = [
        f"I want to segment the {target} in the {modality} image",
        f"The segmentation is {target} in {modality}",
        f"Please segment the {target} in the {modality} images"
    ]

    while iteration < max_iterations and unselected_datasets:
        
        logging.info(f"Label selection iteration {iteration+1}")

        # 加载当前未选数据集的详细信息
        current_dataset_info = load_dataset_info(
            DATASET_DIR,
            keys_to_keep=["name", "input_modalities", "labels"],
            file_list=unselected_datasets
        )

        # 创建新的系统提示，包含当前数据集信息
        current_system_prompt = SYSTEM_PROMPT_LABEL_SELECTION + current_dataset_info

        # 创建新的标签选择代理
        label_selection_agent = ConversableAgent(
            name="label_selection_agent",
            system_message=current_system_prompt,
            llm_config=autogen_config,
            human_input_mode="NEVER",
        )

        max_attempts = 3
        attempt = 0
        dataset_label_dict = None  

        while attempt < max_attempts:
            
            label_selection_message = label_selection_messages[(iteration+attempt)%len(label_selection_messages)]

            label_selection_result = user_proxy.initiate_chat(
                label_selection_agent,
                message=label_selection_message,
                max_turns=1
            )

            # 从标签选择代理的输出中提取 JSON
            dataset_label_dict = extract_json_from_output(
                label_selection_result.summary,
                expected_type='dict'
            )

            # 判断是否成功提取，并且是包含整数或浮点数列表的字典
            if isinstance(dataset_label_dict, dict) and dataset_label_dict and all(
                isinstance(dataset, str) and isinstance(labels, list) and all(isinstance(x, (int, float)) for x in labels)
                for dataset, labels in dataset_label_dict.items()
            ):
                break
            attempt += 1

        if dataset_label_dict is None or not all(
            isinstance(labels, list) 
            and all(isinstance(x, (int, float)) for x in labels)
            for labels in dataset_label_dict.values()
        ):
            dataset_label_dict = {}

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

        iteration += 1
    
    logging.info("最终选择的标签:")
    logging.info(label_selected_labels)

    return label_selected_labels


def main():
    result = chat_with_llm_c2f2("I want to segment the liver vessel in this CT image", "claude-3-5-haiku-20241022", AUTOGEN_CONFIG_PATH)
    print(result)


if __name__ == "__main__":
    main()