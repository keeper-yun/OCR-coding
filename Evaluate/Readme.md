## Evaluate

### 中文评估

1. **执行命令**

   ```bash
   验证validation命令：
   python -u tools/eval_rec.py -c CONFIG_PATH -o Global.checkpoints=checkpoint_path
   
   ex:python -u tools/eval_rec.py -c configs/rec/svtrv2/svtrv2_ch_try.yml -o Global.checkpoints=/data/huawei/wzc/OpenOCR/output/rec/training/BCTR/svtrv2_ch_try/v1.0_epoch100/best.pth
   ```
   ```bash
   测试test命令：（暂时失败，缺少参数）
   python -u tools/eval_rec_all_ch.py -c CONFIG_PATH -o Global.checkpoints=checkpoint_path

   ex:python -u tools/eval_rec_all_ch.py -c configs/rec/svtrv2/svtrv2_ch_try.yml -o Global.checkpoints=/data/huawei/wzc/OpenOCR/output/rec/training/BCTR/svtrv2_ch_try/v1.0_epoch100/best.pth
   ```

   或

   ```bash
   python evaluation_basic_En_ch1.py   (默认是验证，可调整为测试)
   ```

2. **config_path 和 Checkpoint**

   ```python
   config_path = " configs/rec/svtrv2/svtrv2_ch_try.yml"
   
   checkpoint_path = (
       "/data/huawei/wzc/OpenOCR/"
       "output/rec/training/BCTR/"
       "svtrv2_ch_try/v1.0_epoch100/"
       "best.pth"
   )
   ```

   或使用原文提供的预训练模型：

   ```python
   # checkpoint_path = "/data/huawei/wzc/OpenOCR/openocr_svtrv2_ch.pth"
   ```

   > `openocr_svtrv2_ch.pth` 是原文提供的预训练模型。

<br>
<br>

### 英文评估

#### 验证validation

1. **执行命令**
   ```bash
   验证validation命令：
   python -u tools/eval_rec.py -c CONFIG_PATH -o Global.checkpoints=checkpoint_path
   
   ex:python -u tools/eval_rec.py -c configs/rec/svtrv2/svtrv2_smtr_gtc_rctc_try.yml -o Global.checkpoints=/data/huawei/wzc/OpenOCR/output/rec/training/BCTR/svtrv2_ch_try/v1.0_epoch100/best.pth
   ```
   <br>
   
#### 测试test

1. **短文本评估**

   Common、Union14M-Benchmark、OST：（未实现）

   ```bash
   python tools/eval_rec_all_en.py --c configs/rec/svtrv2/svtrv2_smtr_gtc_rctc_infer_try.yml
   ```
    或

   ```bash
   python evaluation_second_short_En1.py
   ```
   <br>
2. **长文本评估**

   LTB：（未实现）

   ```bash
   python tools/eval_rec_all_long.py --c configs/rec/svtrv2/svtrv2_smtr_gtc_rctc_infer_try.yml --o Eval.loader.max_ratio=20
   ```
   






