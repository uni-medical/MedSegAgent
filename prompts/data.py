import json
import os.path as osp
from glob import glob
from collections import defaultdict


def remove_useless_keys(json_content):
    json_content.pop("models", None)
    json_content["labels"].pop("background", None)
    return json_content


def get_info_from_json_dir(json_dir, json_filter=lambda x: x):
    json_path_list = glob(osp.join(json_dir, "*.json"))

    json_info_list = [
        json.dumps(json_filter(json.load(open(json_path, encoding='utf-8'))))
        for json_path in json_path_list
    ]
    json_info = "\n".join(json_info_list)
    return json_info


def get_info_from_json_path_list(json_path_list, json_filter=remove_useless_keys):
    json_info_list = [
        json.dumps(json_filter(json.load(open(json_path, encoding='utf-8'))))
        for json_path in json_path_list
    ]
    json_info = "\n".join(json_info_list)
    return json_info


PROJECT_DIR = osp.dirname(osp.dirname(__file__))
DATASET_DIR = osp.join(PROJECT_DIR, "dataset")
DATASET_JSON_INFO = get_info_from_json_dir(DATASET_DIR, json_filter=remove_useless_keys)
MODEL_JSON_INFO = get_info_from_json_dir(osp.join(PROJECT_DIR, "dataset", "model"))

DATASET_NAME_TO_PATH = dict()
DATASET_NAME_TO_LABEL_INFO = dict()
DATASET_NAME_TO_FILTERED_INFO = dict()
for json_path in glob(osp.join(DATASET_DIR, "*.json")):
    dataset_info = json.load(open(json_path, encoding='utf-8'))
    dataset_name = dataset_info['name']
    DATASET_NAME_TO_PATH[dataset_name] = json_path
    # dataset_label_to_name = dict()
    # for label_name, label in dataset_info["labels"].items():
    #     dataset_label_to_name[str(label)] = label_name
    # DATASET_NAME_TO_INFO[dataset_name] = dataset_label_to_name
    DATASET_NAME_TO_LABEL_INFO[dataset_name] = dataset_info["labels"]
    DATASET_NAME_TO_FILTERED_INFO[dataset_name] = remove_useless_keys(dataset_info)

LABEL_NAME_TO_DATASET = defaultdict(list)
ALL_LABEL_LIST = set()
ALL_DATASET_LIST = set()
for dataset, info in DATASET_NAME_TO_FILTERED_INFO.items():
    ALL_DATASET_LIST.add(dataset)
    label_list = list(info["labels"].keys())
    for label in label_list:
        LABEL_NAME_TO_DATASET[label].append(dataset)
        ALL_LABEL_LIST.add(label)
ALL_LABEL_LIST = sorted(list(ALL_LABEL_LIST))
ALL_DATASET_LIST = sorted(list(ALL_DATASET_LIST))


def retrieve_dataset_label_info(dataset_name, label_list):
    result = ""
    dataset_info = DATASET_NAME_TO_LABEL_INFO.get(dataset_name, dict())

    for label in label_list:
        label_found = False
        for label_name, label_values in dataset_info.items():
            if (isinstance(label_values, int) and label == label_values):
                result += f"{label}({label_name}),"
                label_found = True
                break
            if (isinstance(label_values, list) and label_list == label_values):
                result += f"{label}({label_name}),"
                label_found = True
                break
        if not label_found:
            result += f"{label}(Not Found),"

    if result == "":
        return "No Label Found "

    return result[:-1]  # Remove the trailing comma


def retrieve_dataset_from_label_name(target_label, mode="label", top_k=5, backend="openai"):
    from utils.embed_utils import calculate_similarities
    retrieved_datasets = set()
    if mode == "label":
        top_k_labels = calculate_similarities(ALL_LABEL_LIST, target_label,
                                              backend=backend)[:top_k]
        for label, _ in top_k_labels:
            for dataset in LABEL_NAME_TO_DATASET[label]:
                retrieved_datasets.add(dataset)
    elif mode == "dataset" and backend == "openai":
        top_k_datasets = calculate_similarities(ALL_DATASET_LIST, target_label,
                                                backend=backend)[:top_k]
        for dataset, _ in top_k_datasets:
            retrieved_datasets.add(dataset)
    else:
        raise NotImplementedError()

    if not retrieved_datasets:
        return "No Dataset Found"

    results = "\n".join(
        [str(DATASET_NAME_TO_FILTERED_INFO[dataset]) for dataset in retrieved_datasets])
    return results


if __name__ == "__main__":
    # print(retrieve_dataset_label_info("KiTS23", [2, 3]))
    # print(retrieve_dataset_label_info("AMOS22_Task2", [1]))
    # print(retrieve_dataset_label_info("FLARE23", [8]))
    # print(retrieve_dataset_label_info("TotalSegmentator_v2", [9]))
    # print(retrieve_dataset_from_label_name("brain", mode="label", backend="openai", top_k=1000))
    print(retrieve_dataset_from_label_name("brain", mode="label", backend="spacy", top_k=1000))
    # print(retrieve_dataset_from_label_name("brain", mode="dataset", backend="openai", top_k=1000))
    # print(retrieve_dataset_from_label_name("brain", top_k=1000))
