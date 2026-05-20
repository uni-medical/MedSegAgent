# MedSegAgent: A Universal and Scalable Multi-Agent System for Instructive Medical Image Segmentation

This work has been accepted by the *IEEE Journal of Biomedical and Health Informatics (JBHI)*. Paper: [IEEE Xplore](https://ieeexplore.ieee.org/document/11455620) | [PDF](MedSegAgent_JBHI_2026.pdf).

<p align="center">
  <img src="assets/medsegagent-framework.png" alt="MedSegAgent framework" width="100%">
</p>

<p align="center">
  <em>Overview of the MedSegAgent framework: natural language query parsing, coarse-to-fine dataset matching, and final segmentation with result integration.</em>
</p>

MedSegAgent is a multi-agent system for instructive medical image segmentation. Instead of training one universal segmentation model, it orchestrates specialized dataset-specific models through natural language understanding, coarse-to-fine dataset matching, and execution-time result integration.

## Key Features:
* **Universal & Scalable:** Tackle diverse medical image segmentation tasks using natural language instructions.
* **Precise Automation:** Automatically selects the most suitable segmentation models.
* **Enhanced Robustness:** Improves reliability through multi-model integration and ensemble capabilities.

## Overview
MedSegAgent parses a free-form segmentation request, filters candidate datasets from modality to anatomy to label, and then runs the matched segmentation models. The current repository organizes each integrated dataset with a standardized JSON metadata entry in [`dataset/`](dataset), making it straightforward to extend the model library with new tasks.

In the paper setting, MedSegAgent integrates 23 datasets and supports 343 segmentation targets across CT, MRI, PET/CT, and ultrasound-related scenarios.

## Skill
This repository also includes a reusable skill at [`skills/medsegagent-nnunet-runner/`](skills/medsegagent-nnunet-runner) for inspecting `nnUNet_results`, selecting a deployed task, and running or preparing nnUNet inference.

The skill is intended for environments that already have nnUNet v2 available. To run inference, the target environment must provide a configured `nnUNet_results` path and usable nnUNet CLI or Python API support.

## Supported Datasets
The repository currently includes metadata for the following datasets and targets. The summary below is adapted from the dataset table in the paper. The links point to official dataset access pages rather than direct-download mirrors hosted by this repository; some datasets still require registration, challenge participation, or a signed data-use agreement before download.

| Dataset | Modalities | Body Region | Representative Targets | Data access |
| --- | --- | --- | --- | --- |
| TotalSegmentator v2 | CT | Whole-body | 117 structures including organs, vessels, bones, and brain | [Zenodo](https://doi.org/10.5281/zenodo.6802613) |
| TotalSegmentator MRI | MRI | Whole-body | 56 structures including organs, vessels, spine, muscles, and brain | [Zenodo](https://doi.org/10.5281/zenodo.11367004) |
| CT-ORG | CT | Whole-body | liver, bladder, lungs, kidneys, bone, brain | [TCIA](https://www.cancerimagingarchive.net/collection/ct-org/) |
| AutoPET | PET/CT | Whole-body | whole-body tumor sites | [TCIA FDG-PET-CT-Lesions](https://www.cancerimagingarchive.net/collection/fdg-pet-ct-lesions/) |
| SegRap2023 Task1 | CT | Head and neck | 45 OAR structures | [Grand Challenge](https://segrap2023.grand-challenge.org/dataset/) |
| BraTS21 | MRI | Head and neck | whole tumor, tumor core, enhancing tumor | [CBICA BraTS 2021](https://www.med.upenn.edu/cbica/brats2021/) |
| ISLES22 | MRI | Head and neck | stroke lesion | [Zenodo](https://doi.org/10.5281/zenodo.7153326) |
| ISLES22 ATLAS | MRI | Head and neck | stroke lesion | [Grand Challenge](https://atlas.grand-challenge.org/) |
| Instance22 | CT | Head and neck | intracranial hemorrhage | [Grand Challenge](https://instance.grand-challenge.org/) |
| HECKTOR2022 | PET/CT | Head and neck | GTVp, GTVnd | [Grand Challenge](https://hecktor.grand-challenge.org/Data/) |
| SegRap2023 Task2 | CT | Head and neck | GTVp, GTVnd | [Grand Challenge](https://segrap2023.grand-challenge.org/dataset/) |
| MM-WHS | MRI, CT | Heart | cardiac chambers, myocardium, great vessels | [Challenge site](https://zmiclab.github.io/zxh/0/mmwhs/) |
| ACDC | MRI | Heart | left ventricle, right ventricle, myocardium | [Challenge site](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html) |
| ImageCAS | CT | Heart | coronary artery | [Kaggle](https://www.kaggle.com/datasets/xiaoweixumedicalai/imagecas) |
| Parse22 | CT | Thorax | pulmonary artery | [Grand Challenge](https://parse2022.grand-challenge.org/Dataset/) |
| ATM22 | CT | Thorax | pulmonary airway | [Grand Challenge](https://atm22.grand-challenge.org/) |
| AbdomenAtlasMini | CT | Abdomen | kidneys, liver, pancreas, spleen, stomach, vessels | [Hugging Face](https://huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas1.0Mini) |
| AMOS22 Task2 | MRI, CT | Abdomen | 15 abdominal and pelvic structures | [Zenodo](https://doi.org/10.5281/zenodo.7155725) |
| FLARE22 | CT | Abdomen | 13 abdominal organs | [Grand Challenge](https://flare22.grand-challenge.org/) |
| WORD | CT | Abdomen | abdominal organs, bowel, bladder, femurs | [GitHub](https://github.com/HiLab-git/WORD) |
| KiTS23 | CT | Abdomen | kidneys, renal tumors, renal cysts | [Challenge site](https://kits-challenge.org/kits23/) |
| LiTS | CT | Abdomen | liver, liver tumor | [Kaggle](https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation) |
| Adrenal-ACC-Ki67-Seg | CT | Abdomen | adrenocortical carcinoma | [TCIA](https://www.cancerimagingarchive.net/collection/adrenal-acc-ki67-seg/) |

## Quick Start:
### Setup your environment
```
conda create -n medsegagent python=3.12
conda activate medsegagent
pip install uv
uv pip install -r requirements.txt
```
Then set your LLM service API keys like [OAI_CONFIG_LIST.example](OAI_CONFIG_LIST.example), the config file should be named `OAI_CONFIG_LIST`.
```
[
    {
        "model": "Qwen/Qwen2.5-32B-Instruct",
        "api_key": "<Your API KEY>",
        "base_url": "https://api.siliconflow.cn/v1",
        "tags": ["silicon"]
    },
    {
        "model": "gpt-4o-2024-08-06",
        "api_key": "<Your API KEY>",
        "base_url": "<your BASE URL>",
        "tags": ["openai"]
    }
]
```
### Test script
Start from eval_example.sh to try our Coarse-to-Fine setting of seg model selection.
```
python ./evaluate.py \
    --test_file_path "model_selection_test_case.jsonl" \
    --test_pattern "C2F"  \
    --model "gpt-4o-2024-08-06"    \
    --log_to_file
```

## Acknowledgments
This project builds on the open contributions of the medical image segmentation community. We gratefully acknowledge the creators and maintainers of the public datasets integrated in MedSegAgent, whose annotations, benchmarks, and challenge platforms make this system possible.

We also acknowledge [`nnU-Net`](https://github.com/MIC-DKFZ/nnUNet), which provides the strong self-configuring segmentation framework used for the dataset-specific models in our study.

## TODO
- Upload the trained segmentation models.

## Citation
If you use MedSegAgent in your research, please cite:

```bibtex
@ARTICLE{11455620,
  author={Huang, Ziyan and Wang, Haoyu and Ye, Jin and Ji, Yuanfeng and Hu, Xiaowei and Liu, Lihao and Yang, Zhikai and Li, Wei and Hu, Ming and Su, Yanzhou and Li, Tianbin and Gu, Yun and Zhang, Shaoting and Qiao, Yu and Gu, Lixu and He, Junjun},
  journal={IEEE Journal of Biomedical and Health Informatics},
  title={MedSegAgent: A Universal and Scalable Multi-Agent System for Instructive Medical Image Segmentation},
  year={2026},
  volume={},
  number={},
  pages={1-12},
  keywords={Image segmentation;Medical diagnostic imaging;Filtering;Solid modeling;Natural languages;Computed tomography;Computational modeling;Accuracy;Multi-agent systems;Liver;Universal Medical Image Segmentation;Multi-Agent System;Natural Language Instruction},
  doi={10.1109/JBHI.2026.3677444}
}
```
