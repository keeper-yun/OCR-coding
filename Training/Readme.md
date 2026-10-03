## Training

### 中文训练

1. **找到配置文件**

   `/data/huawei/wzc/OpenOCR/configs/rec/svtrv2/svtrv2_ch_try.yml`

2. **配置参数**

   `output_dir`、`pretrained_model`、`Train:data_dir`、`Eval:data_dir`

   > `svtrv2_ch_try.yml` 已配置

3. **运行指令**

   ```bash
   python tools/train_rec.py -c configs/rec/svtrv2/svtrv2_ch_try.yml
   ```

   或

   ```bash
   python train_svtrv2_ch.py
   ```

4. **训练结果**

   `查看svtrv2_ch_try.yml中的output_dir`：

   `/data/huawei/wzc/OpenOCR/output/rec/training/BCTR/...`

<br>
<br>

### 英文训练

#### 阶段 1

1. **找到配置文件**

   `/data/huawei/wzc/OpenOCR/configs/rec/svtrv2/svtrv2_rctc_try.yml`

2. **配置参数**

   `output_dir`、`pretrained_model`、`Train:data_dir`、`Eval:data_dir`

   > `svtrv2_rctc_try.yml` 已配置

3. **运行指令**

   ```bash
   python tools/train_rec.py -c configs/rec/svtrv2/svtrv2_rctc_try.yml
   ```

   或

   ```bash
   CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 tools/train_rec.py --c configs/rec/svtrv2/svtrv2_rctc_try.yml
   ```

   或

   ```bash
   python train_svtrv2_En.py
   ```

4. **训练结果**

   `查看svtrv2_rctc_try.yml中的output_dir,以output_dir中内容为主`：

   `ex:/data/huawei/wzc/OpenOCR/output/rec/training/u14m_filter/svtrv2_rctc_try/stage1/v5.0_epoch20/`

   * `train_curve.png`：训练 Acc、训练评估 Acc 和训练 Loss
   
   * `best.pth`：阶段 2 的预训练权重

<br>
<br>

#### 阶段 2

1. **找到配置文件**

   `/data/huawei/wzc/OpenOCR/configs/rec/svtrv2/svtrv2_smtr_gtc_rctc_try.yml`

2. **配置参数**

   `output_dir`、`pretrained_model`、`Train:data_dir`、`Eval:data_dir`

   > `svtrv2_smtr_gtc_rctc_try.yml` 已配置

3. **运行指令**

   ```bash
   CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port=23332 --nproc_per_node=1 tools/train_rec.py --c configs/rec/svtrv2/svtrv2_smtr_gtc_rctc_try.yml --o Global.pretrained_model=/data/huawei/wzc/OpenOCR/output/rec/training/u14m_filter/svtrv2_rctc_try/stage1/v5.0_epoch20/best.pth
   ```

   或

   ```bash
   python train_svtrv2_En2.py
   ```

4. **训练结果**

   `查看svtrv2_smtr_gtc_rctc_try.yml中的output_dir`：

   `/data/huawei/wzc/OpenOCR/output/rec/training/u14m_filter/svtrv2_rctc_try/stage2/v5.0_epoch20/`

   * `train_curve.png`：第二阶段训练 Acc、训练评估 Acc 和训练 Loss

5. **注意**

   ```bash
   python plt_train_result.py
      ```
   
   会根据train.log中记录信息来生成训练过程acc图表
   
   `位置：/data/huawei/wzc/OpenOCR/plt_train_result.py`
<br>
<br>

