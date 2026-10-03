## Dataset

SVTRv2 的实验数据集分为 **English Dataset** 和 **Chinese Dataset (BCTR)** 两部分，分别用于英文和中文场景下的模型训练与评估。

### Dataset Location：

```bash
/data/huawei/wzc/Dataset/
```

```text
Dataset
|
├── 中文数据集
|—— training/
│   └── BCTR/
│       ├── scene_train/
│       ├── document_train/
│       └── web_train/
|
|
│—— validation/
|   └── BCTR/
│       ├── scene_val/
│       ├── document_val/
│       └── web_val/
|
│—— test/
│   └── BCTR/
│       ├── scene_test/
│       ├── document_test/
│       └── web_test/
│
|
|——英文数据集
├── training
|   └──Union14M-L-LMDB-Filtered/
│        └── filter_train_medium/
|
│       
│
└── evaluation_En/
    ├── CUTE80/
    ├── IC13/
    ├── IC15/
    ├── IIIT5K/
    ├── SVT/
    └── SVTP/
```

