import spacy
import numpy as np
from openai import OpenAI
from autogen import config_list_from_json
import os.path as osp
import os
import json

nlp = spacy.load("en_core_web_sm")
PROJECT_DIR = osp.dirname(osp.dirname(__file__))
AUTOGEN_CONFIG_PATH = osp.join(PROJECT_DIR, "OAI_CONFIG_LIST")
openai_embed_store_file = osp.join(PROJECT_DIR, "ckpt", "openai_embed_store.json")
OPENAI_EMBED_STORE = dict()
if (osp.exists(openai_embed_store_file)):
    OPENAI_EMBED_STORE = json.load(open(openai_embed_store_file, "r", encoding='utf-8'))

config = config_list_from_json(env_or_file=AUTOGEN_CONFIG_PATH, filter_dict={"model":
                                                                             ["gpt-4"]})[0]
client = OpenAI(api_key=config['api_key'], base_url=config['base_url'])


def get_text_embedding(text, model="text-embedding-ada-002"):
    if (text in OPENAI_EMBED_STORE):
        return OPENAI_EMBED_STORE[text]
    print(f"recompute since no cache found for: {text}")
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=text, model=model)
    embed = response.data[0].embedding
    return np.array(embed)


def calculate_similarities(label_list, target_label, backend="openai"):
    """
    计算目标标签与标签列表中每个标签的语义相似度，并返回相似度列表。

    :param label_list: 包含标签的字符串列表
    :param target_label: 目标标签字符串
    :return: 包含 (标签, 相似度) 元组的列表
    """
    if backend == "spacy":
        embedding_func = lambda x: nlp(x).vector
    elif backend == "openai":
        embedding_func = get_text_embedding
    else:
        raise NotImplementedError(f"{backend} is not supported for embeddon")

    similarities = []

    target_vector = embedding_func(target_label)
    for label in label_list:
        label_vector = embedding_func(label)

        similarity = np.dot(target_vector, label_vector) / (np.linalg.norm(target_vector) *
                                                            np.linalg.norm(label_vector))

        similarities.append((label, similarity))

    return sorted(similarities, key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    categories = [
        "kidney", "kidney_right", "kidney_left", "right kidney", "liver", "heart", "lung"
    ]
    target = "left kidney"

    for backend in ["spacy", "openai"]:
        print("backend:", backend)
        similarities = calculate_similarities(categories, target, backend=backend)
        for category, similarity in similarities:
            print(f"Category: {category}, Similarity: {similarity}")
        print("----------------------------------------------------------")
