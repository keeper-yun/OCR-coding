# OCR (Optical Character Recognition)

## Introduction

This project primarily focuses on the **SVTRv2** model to explore OCR capabilities for smart glasses, covering datasets, model configurations, model training, and evaluation.

<br>

## 1. Dataset

The SVTRv2 experiments use two main groups of datasets: **English Dataset** and **Chinese Dataset (BCTR)**, which are used for model training and evaluation in English and Chinese text recognition scenarios, respectively.

### English Dataset

* **Training Set:** `filter_train_medium` from the **Union14M-L-LMDB-Filtered** dataset.

  | Dataset               |      Easy |  Medium |    Hard |    Norm | Challenging |     Total |
  | --------------------- | --------: | ------: | ------: | ------: | ----------: | --------: |
  | **Union14M-L**        | 2,076,161 | 145,525 | 308,025 | 218,154 |     482,877 | 3,230,742 |
  | **Union14M-L-Filter** | 2,073,822 | 144,677 | 306,771 | 217,070 |     481,803 | 3,224,143 |

* **Test Set:** The `evaluation` dataset provided in the paper, containing six classic English scene text recognition benchmarks:

  | Dataset    | Description                                      |
  | ---------- | ------------------------------------------------ |
  | **IIIT5K** | IIIT 5K-Words scene text recognition dataset     |
  | **SVT**    | Street View Text scene text recognition dataset  |
  | **IC13**   | ICDAR 2013 scene text recognition dataset        |
  | **IC15**   | ICDAR 2015 scene text recognition dataset        |
  | **SVTP**   | SVT Perspective scene text recognition dataset   |
  | **CUTE80** | CUTE80 irregular/curved text recognition dataset |

### Chinese Dataset — BCTR

The Chinese text recognition dataset is **BCTR (Benchmark for Chinese Text Recognition)**, downloaded from the `_chinese_lmdb` dataset provided by OpenOCR.

The BCTR dataset is divided into three text scenarios: **Scene**, **Document**, and **Web**, with separate training, validation, and test sets.

* **Training Set:** `scene_train`, `document_train`, and `web_train`

* **Validation Set:** `scene_val`, `document_val`, and `web_val`

* **Test Set:** `scene_test`, `document_test`, and `web_test`

### Dataset Summary

| Type        | Dataset                  | Training                                     | Validation                                        | Test                                              |
| ----------- | ------------------------ | -------------------------------------------- | ------------------------------------------------- | ------------------------------------------------- |
| **English** | Union14M-L-LMDB-Filtered | `filter_train_medium`                        | `IIIT5K`, `SVT`, `IC13`, `IC15`, `SVTP`, `CUTE80` | `IIIT5K`, `SVT`, `IC13`, `IC15`, `SVTP`, `CUTE80` |
| **Chinese** | BCTR                     | `scene_train`, `document_train`, `web_train` | `scene_val`, `document_val`, `web_val`            | `scene_test`, `document_test`, `web_test`         |

### Dataset Location

```bash
/data/huawei/wzc/Dataset/
```

<br>

## 2. Model

### 2.1 SVTRv2 — English

* **RCTC:** Uses `svtrv2_rctc_try.yml` for **Stage 1 training**.
* **SMTR + GTC + RCTC:** Uses `svtrv2_smtr_gtc_rctc_try.yml` for **Stage 2 training**, with `svtrv2_smtr_gtc_rctc_infer_try.yml` used for evaluation.
* **CTC:** Not used in the current experiments.

> **Note:** Configuration files with the `try` suffix were modified according to the experimental requirements of this project. The remaining configuration files are the original OpenOCR configurations.

### 2.2 SVTRv2 — Chinese

* `svtrv2_ch.yml`
* `svtrv2_smtr_gtc_rctc_ch.yml`

### Configuration Location

```bash
/data/huawei/wzc/OpenOCR/configs/rec/svtrv2/*.yml
```

Example:

```bash
/data/huawei/wzc/OpenOCR/configs/rec/svtrv2/svtrv2_ch_try.yml
```

<br>

## 3. Preliminary Results

Preliminary training and evaluation experiments have been completed on both the Chinese and English datasets using SVTRv2.

**Stage 1 training results:**

<img width="500" height="354" alt="svtrv2_train_medium_epoch20_stage1" src="https://github.com/user-attachments/assets/8ac049fa-23ee-4280-92cb-f562340012c3" />

<br>

**Stage 2 training results:**

<img width="500" height="354" alt="svtrv2_train_medium_epoch20_stage2" src="https://github.com/user-attachments/assets/753250f1-bbc3-485d-a68f-cad130df1727" />

**Evaluation results:**

| Language            | Dataset     |   Accuracy | Norm Edit Distance | Samples |       FPS |
| ------------------- | ----------- | ---------: | -----------------: | ------: | --------: |
| **Chinese**         | BCTR        | **78.87%** |         **88.11%** | 127,704 |    212.96 |
| **English**         | CUTE80      | **89.58%** |         **96.82%** |     288 |  1,010.92 |
| **English**         | IC13        | **94.87%** |         **98.70%** |     857 |  9,986.60 |
| **English**         | IC15        | **81.17%** |         **94.22%** |   1,811 | 19,238.95 |
| **English**         | IIIT5K      | **95.10%** |         **98.77%** |   3,000 | 20,381.08 |
| **English**         | SVT         | **90.26%** |         **97.49%** |     647 | 10,172.56 |
| **English**         | SVTP        | **81.09%** |         **93.58%** |     645 |  9,935.42 |
| **English Average** | **Average** | **88.68%** |         **96.60%** |   1,208 | 11,787.59 |

> **Note:** The Chinese results are preliminary evaluation results on BCTR. The English results are obtained from the individual datasets and average results in `evaluation_En`.

<br>

## 4. Quick Start

### Environment

* PyTorch version >= 1.13.0
* Python version >= 3.7

```bash
git clone -b develop https://github.com/Topdu/OpenOCR.git
cd OpenOCR

# Ubuntu 20.04, CUDA 11.8
conda create -n openocr python==3.8
conda activate openocr

conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Installation

```bash
cd OpenOCR
python build_package.py
pip install ./build/dist/openocr_python-*.whl
```

### Text Detection + Recognition (OCR)

#### Command Line Usage

```bash
# Basic usage
openocr --task ocr --input_path path/to/img

# With visualization
openocr --task ocr --input_path path/to/img --is_vis

# Process a directory with custom output
openocr --task ocr --input_path ./images --output_path ./results --is_vis

# Use server mode (higher accuracy)
pip install torch torchvision
openocr --task ocr --input_path path/to/img --mode server --backend torch
```

#### Python Usage

```python
# SVTRv2 end-to-end OCR
from openocr import OpenOCR
import os
import json

ocr = OpenOCR(task='ocr', mode='mobile', backend='onnx')

# Input image
image_path = r'D:\...\sample.jpg'

# Output directory
save_dir = r'D:\...\output'

# OCR
results, time_dicts = ocr(
    image_path=image_path,
    save_dir=save_dir,
    is_visualize=True,
    rec_batch_num=200,
)

# Specify output file name
txt_path = os.path.join(save_dir, 'Network_sample.txt')

# Extract only transcription results
with open(txt_path, 'w', encoding='utf-8') as f:
    for result in results:
        image_name, json_data = result.split('\t', 1)
        ocr_data = json.loads(json_data)

        for item in ocr_data:
            text = item["transcription"]
            f.write(text + '\n')

print('Successfully saved!')
```
<br>

## 5. Location of Output Files and Weights

The model weights and files generated during the training process are stored in:

```bash
/data/huawei/wzc/OpenOCR/output/rec/training/
```

For example, the following directory indicates:

* **Dataset:** English `u14m_filter_medium`
* **Training Stage:** Stage 1
* **Training Run:** 5th training run
* **Training Epochs:** 20 epochs

```bash
/data/huawei/wzc/OpenOCR/output/rec/training/u14m_filter/svtrv2_rctc_try/stage1/v5.0_epoch20/
```
<br>

## Repo Layout

```text
Dataset/
    Readme.md
    数据集位置.md

Evaluate/
    tools/
        eval_rec.py                       # Evaluate validation set
        eval_rec_all_ch.py                # Evaluate test set
    evaluation_basic_En_ch1.py            # Basic English and Chinese evaluation (Stage 1)
    evaluation_second_short_En1.py        # English evaluation after Stage 2 training
    svtrv2_ch_try.yml                     # Chinese evaluation configuration
    svtrv2_smtr_gtc_rctc_infer_try.yml    # English Stage 2 evaluation configuration

Model/
    # Model configuration files
    svtrv2_ch_try.yml                     # Chinese baseline training
    svtrv2_rctc_try.yml                   # English Stage 1 training
    svtrv2_smtr_gtc_rctc_try.yml          # English Stage 2 training

Training/
    tools/
        train_rec.py                      # Training entry point
    train_svtrv2_En.py                    # English Stage 1 training
    train_svtrv2_En2.py                   # English Stage 2 training
    train_svtrv2_ch.py                    # Chinese baseline training
```
<br>

## Reference

[1] Y. Du, Z. Chen, H. Xie, C. Jia and Y.-G. Jiang, "SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition," *2025 IEEE/CVF International Conference on Computer Vision (ICCV)*, Honolulu, HI, USA, 2025, pp. 20147–20156, doi: 10.1109/ICCV51701.2025.01874.<br>

[2] Du, Y., "UniRec-0.1B: Unified Text and Formula Recognition with 0.1B Parameters," *arXiv e-prints*, Art. no. arXiv:2512.21095, 2025. doi: 10.48550/arXiv.2512.21095.<br>

[3] TAO W, HE S, LU K, et al. "Value-Driven Mixed-Precision Quantization for Patch-Based Inference on Microcontrollers," *arXiv*, 2024. https://arxiv.org/abs/2401.13714. DOI: 10.48550/ARXIV.2401.13714.<br>

[4] Yansong Sun, Jialuo He, Dirk Kutscher, and Huangxun Chen. "AdaptQNet: Optimizing Quantized DNN on Microcontrollers via Adaptive Heterogeneous Processing Unit Utilization." In *Proceedings of the 31st Annual International Conference on Mobile Computing and Networking (ACM MobiCom '25)*. Association for Computing Machinery, New York, NY, USA, 2025.
