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

1. **单次评估**
   ```bash
   验证validation命令：
   python -u tools/eval_rec.py -c CONFIG_PATH -o Global.checkpoints=checkpoint_path
   
   ex:python -u tools/eval_rec.py -c configs/rec/svtrv2/svtrv2_smtr_gtc_rctc_try.yml -o Global.checkpoints=/data/huawei/wzc/OpenOCR/output/rec/training/BCTR/svtrv2_ch_try/v1.0_epoch100/best.pth
   ```
   
   或

   ```bash
   python evaluation_basic_En_ch1.py   (默认是验证，可调整为测试)
   ```
   <br>

2. **综合评估——短文本（Common, Union14M-Benchmark, OST）**
   ```bash
   验证validation命令：
   python tools/eval_rec_all_en.py --c CONFIG_PATH
   
   ex:python tools/eval_rec_all_en.py --c configs/rec/svtrv2/svtrv2_smtr_gtc_rctc_try.yml
   ```
   <br>

3. **综合评估——长文本（LTB）**
   <br><br>
   *没有长文本，执行不了*
   ```bash
   验证validation命令：
   python tools/eval_rec_all_long.py --c CONFIG_PATH
   
   ex:python tools/eval_rec_all_long.py --c configs/rec/svtrv2/svtrv2_smtr_gtc_rctc_try.yml
   ```
   <br>
   
#### 测试test(未实现)

1. **短文本评估**

   Common、Union14M-Benchmark、OST：

   ```bash
   python tools/eval_rec_all_en.py --c configs/rec/svtrv2/svtrv2_smtr_gtc_rctc_infer_try.yml
   ```

   <br>
2. **长文本评估**

   LTB：

   ```bash
   python tools/eval_rec_all_long.py --c configs/rec/svtrv2/svtrv2_smtr_gtc_rctc_infer_try.yml --o Eval.loader.max_ratio=20
   ```
   






