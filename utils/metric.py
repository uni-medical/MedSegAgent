import numpy as np

# expected data format:
# {"dataset1": [1,2,3], "dataset2": [1]}
''' How many percentage of datasets are found'''
def dataset_hit_rate(gt_answer, llm_answer):
    try:
        hit_count = 0
        total_datasets = len(set(gt_answer.keys()) | set(llm_answer.keys()))

        for dataset, gt_labels in gt_answer.items():
            if dataset in llm_answer:
                val = llm_answer[dataset]
                # 若存在子列表，将其一层展平
                if isinstance(val, list) and any(isinstance(x, list) for x in val):
                    val = [elem for sub in val for elem in (sub if isinstance(sub, list) else [sub])]

                # 进行集合比较
                if set(gt_labels) == set(val):
                    hit_count += 1

        return hit_count / total_datasets
    except Exception as e:
        print(f"Error occurred: {e}")
        return 0


''' If any dataset is found successfully, return 1, else 0'''


def any_dataset_found_rate(gt_answer, llm_answer):
    hr = dataset_hit_rate(gt_answer, llm_answer)
    return 1 if (hr > 0) else 0


''' If all datasets are found, return 1, else 0'''


def all_dataset_found_rate(gt_answer, llm_answer):
    hr = dataset_hit_rate(gt_answer, llm_answer)
    return 1 if (hr == 1) else 0


if __name__ == "__main__":
    gt = {
        'TotalSegmentator_v2': [2, 3],
        'FLARE23': [2, 13],
        'AbdomenAtlasMini': [3, 4],
        'AMOS22_Task2': [2, 3],
        'KiTS23': [1],
        'CT-ORG': [4]
    }
    pred = {
        'AbdomenAtlasMini': [3, 4],
        'CT-ORG': [4],
        'KiTS23': [1, 2, 3],
        'LiTS': [1, 2],
        'TotalSegmentator_v2': [2, 3]
    }

    gt = {'SegRap2023 Task1': [40, 41]}
    pred = {'SegRap2023_Task1': [40, 41]}
    res = dataset_hit_rate(gt, pred)
    print(res)
