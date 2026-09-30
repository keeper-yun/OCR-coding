# Reference SVTRv2 ONNX CPU run

This script runs a supplied **OpenOCR SVTRv2 mobile ONNX** model on generated text images, records per-call time and decoded output, and writes asset hashes. Supply the ONNX file outside this repository.

```bash
python -m pip install numpy onnxruntime opencv-python-headless pillow
python tools/reference_svtrv2_onnx_cpu.py \
  --model /external/path/openocr_rec_model.onnx \
  --out /external/path/ocr_cpu_run
```

This repository has several SVTRv2 configurations, but no corresponding team checkpoint was supplied for this run. The public mobile ONNX model used by the local Windows workload is a separate reference. These synthetic samples check the inference path and timing only; the result is neither a model accuracy measurement nor an i.MX93 M33/BCU baseline.
