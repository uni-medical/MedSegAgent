# 数据集 JSON 模板说明

在添加一个新的数据集时，需要按照以下 JSON 模板进行准备。请确保每个字段都正确填写，并遵循模板中的格式。

```json
{
    "name": "dataset name",
    "description": "",
    "input_modalities": {
        "0": "modality1",
        "1": "modality2"
    },
    "labels": {
        "background": 0,
        "class1": 1,
        "class2": 2
    },
    "numTraining": 100,
    "file_ending": ".nii.gz",
    "models": {
        "nnV1": [
            "nnUNet",
            "STU-Net-S",
            "STU-Net-B",
            "STU-Net-L"
        ],
        "nnV2": [
            "nnUNet",
            "STU-Net-S",
            "STU-Net-B",
            "STU-Net-L"
        ],
        "monai": [
            "SwinUNETR",
            "UNETR"
        ]
    }
}
```
## 字段说明

- **name**: 数据集的名称。
  - 类型: 字符串
  - 示例: `"dataset name"`

- **description**: 数据集的描述。
  - 类型: 字符串
  - 示例: `""`（可为空）

- **input_modalities**: 输入模态的列表，每个模态对应一个键值对。
  - 类型: 对象
  - 示例:
    ```json
    {
        "0": "modality1",
        "1": "modality2"
    }
    ```

- **labels**: 标签列表，每个标签对应一个键值对，键为标签名称，值为标签的数字表示。
  - 类型: 对象
  - 示例:
    ```json
    {
        "background": 0,
        "class1": 1,
        "class2": 2
    }
    ```

- **numTraining**: 训练样本的数量。
  - 类型: 整数
  - 示例: `100`

- **file_ending**: 数据文件的后缀名。
  - 类型: 字符串
  - 示例: `".nii.gz"`

- **models**: 模型列表，每个模型框架包含多个模型名称。
  - 类型: 对象
  - 示例:
    ```json
    {
        "nnV1": [
            "nnUNet",
            "STU-Net-S",
            "STU-Net-B",
            "STU-Net-L"
        ],
        "nnV2": [
            "nnUNet",
            "STU-Net-S",
            "STU-Net-B",
            "STU-Net-L"
        ],
        "monai": [
            "SwinUNETR",
            "UNETR"
        ]
    }
    ```
