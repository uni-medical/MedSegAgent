from prompts.basic_v2 import FORMATTED_OUTPUT
from prompts.data import DATASET_NAME_TO_PATH, get_info_from_json_path_list

DEFAULT_EXPERT_PROFILE = "You are an AI expert on medical image segmentation, specializing in addressing clinical tasks efficiently."


def get_selected_dataset_info(dataset_name_list):
    json_path_list = [DATASET_NAME_TO_PATH.get(dn, None) for dn in dataset_name_list]
    if (None in json_path_list):
        raise ValueError(f"non-exist dataset name found: {dataset_name_list}")
    return get_info_from_json_path_list(json_path_list)


def generate_expert_sys_prompt_dict(expert_profile, dataset_name_list):
    selected_dataset_info = get_selected_dataset_info(dataset_name_list)
    ai_expert_sys_prompt = f"""
    # Role:
    {expert_profile}.

    ## Skills:
    1. Thought Process Optimization
    - Quickly analyze the specific clinical need and the segmentation targets.
    - Choose all suitable datasets and corresponding labels.

    2. Dataset & Label Selection
    - Use the provided dataset information to ensure compatibility.
    - Validate the selected labels can cover the need to segment the requested targets.

    ## Constraints:
    1. Thought Process Constraints
    - Thoughts must be concise and short (within 3 sentences).
    - All suitable datasets and corresponding labels should be selected.

    2. Response Format
    - Use the following JSON format:
    ```
    {FORMATTED_OUTPUT}
    ```

    3. Handling Missing Modalities
    - If fail to find any needed modalities, reply in JSON with: 'Missing modalities.' ('MR' is equal to 'MRI')
    - If no suitable dataset is available, reply in JSON with: 'Dataset: None\nLabels: None'

    Here's the dataset information:
    {selected_dataset_info}

    Let’s Begin!
    """
    return {"Expert": ai_expert_sys_prompt}


def generate_doctor_expert_sys_prompt_dict(expert_profile, dataset_name_list):
    selected_dataset_info = get_selected_dataset_info(dataset_name_list)
    doctor_sys_prompt = f""""
    You are a doctor who request a AI expert to select the correct tools (datasets and labels) to the segment requested targets on the medical image with specific modality. 
    After the expert's response, you will verify if the labels of the selected datasets meets your request, and give your feedback.

    The dataset informations are here: 
    {selected_dataset_info}. 

    Your request:
    """
    ai_expert_sys_prompt = generate_expert_sys_prompt_dict(expert_profile, dataset_name_list)
    ai_expert_sys_prompt = ai_expert_sys_prompt["Expert"].replace(
        "Let’s Begin!",
        "When you are given the feedback, you will adjust your response and return it with the correct format. Remember, ALL suitable datasets and corresponding labels should be selected. \n Let’s Begin!"
    )
    return {
        "Doctor": doctor_sys_prompt,
        "Expert": ai_expert_sys_prompt,
    }


def verify_dataset_coverage(profile_to_datasets_mapping):
    dataset_list = []
    for assigned_datasets in profile_to_datasets_mapping.values():
        dataset_list.extend(list(assigned_datasets))
    unique_values = set(dataset_list)
    target_set = set(DATASET_NAME_TO_PATH.keys())

    if unique_values == target_set:
        print("The sets are the same.")
    else:
        missing_items = target_set - unique_values
        extra_items = unique_values - target_set

        if missing_items:
            print(f"Missing items: {missing_items}")
        if extra_items:
            print(f"Extra items: {extra_items}")

    for dataset_name in unique_values:
        dataset_count = dataset_list.count(dataset_name)
        if (dataset_count > 1):
            print(f"The dataset '{dataset_name}' is included {dataset_count} times.")


DEPARTMENT_EXPERTS_INFO_OVERLAPPED = {
    "An Expert in Abdominal Medical Image Segmentation in Gastroenterology Department.":
    ["AMOS22_Task2", "FLARE23", "AbdomenAtlasMini"],
    "An Expert in Kidney and Urological Medical Image Segmentation in Nephrology/Urology Department.":
    ["KiTS23", "TotalSegmentator_v2", "CT-ORG"],
    "An Expert in Thoracic Medical Image Segmentation in Pulmonology Department.":
    ["ATM22", "Parse22", "TotalSegmentator_v2"],
    "An Expert in Head and Neck Medical Image Segmentation in Otolaryngology Department.":
    ["HECKTOR2022", "SegRap2023_Task1", "SegRap2023_Task2"],
    "An Expert in Brain and Neurological Medical Image Segmentation in Neurology Department.":
    ["BraTS21", "ISLES22", "TotalSegmentator_MRI"],
    "An Expert in Cardiac Medical Image Segmentation in Cardiology Department.":
    ["MM-WHS", "MMs", "TotalSegmentator_v2", "ImageCAS"],
    "An Expert in Hepatic and Oncological Medical Image Segmentation in Hepatology/Oncology Department.":
    ["LiTS", "AutoPET", "FLARE23", "Adrenal-ACC-Ki67-Seg"],
    "An Expert in Stroke Lesion Segmentation in Emergency/Radiology Department.":
    ["ISLES22", "ISLES22_ATLAS", "Instance22"],
}

DEPARTMENT_EXPERTS_INFO = {
    "An Expert in Full-Body CT Medical Image Segmentation in Radiology Department.":
    ["CT-ORG", "TotalSegmentator_v2"],
    "An Expert in Full-Body MRI Medical Image Segmentation in Radiology Department.":
    ["TotalSegmentator_MRI"],
    "An Expert in Abdominal Medical Image Segmentation in Gastroenterology Department.":
    ["AMOS22_Task2", "FLARE23", "AbdomenAtlasMini", "KiTS23"],
    "An Expert in Thoracic Medical Image Segmentation in Pulmonology Department.":
    ["ATM22", "Parse22"],
    "An Expert in Head and Neck Medical Image Segmentation in Otolaryngology Department.":
    ["HECKTOR2022", "SegRap2023_Task1", "SegRap2023_Task2"],
    "An Expert in Brain and Neurological Medical Image Segmentation in Neurology Department.":
    ["BraTS21", "ISLES22", "ISLES22_ATLAS", "Instance22"],
    "An Expert in Cardiac Medical Image Segmentation in Cardiology Department.":
    ["MM-WHS", "MMs", "ImageCAS"],
    "An Expert in Oncological Medical Image Segmentation in Oncology Department.":
    ["LiTS", "AutoPET", "Adrenal-ACC-Ki67-Seg"],
}

if __name__ == "__main__":
    print(
        generate_expert_sys_prompt_dict(
            "An Expert in Abdominal Medical Image Segmentation in Gastroenterology Department.",
            ["AMOS22_Task2"]))

    verify_dataset_coverage(DEPARTMENT_EXPERTS_INFO)
