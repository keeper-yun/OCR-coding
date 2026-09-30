"""Host CPU smoke baseline for a supplied OpenOCR SVTRv2 ONNX model.

The model file is supplied explicitly; this does not export this repository's
checkpoint or measure i.MX93 M33/NPU execution.
"""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont


DEFAULT_DICT = Path(__file__).resolve().parent / "utils/ppocr_keys_v1.txt"


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dict", type=Path, default=DEFAULT_DICT)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.model), options, providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 1:
        raise ValueError("this reference runner expects one OCR image input")
    input_name = session.get_inputs()[0].name
    characters = ["blank"] + args.dict.read_text(encoding="utf-8").splitlines() + [" "]
    rows = []
    for index in range(args.count):
        label = ("HELLO 123", "EDGE AI", "KWS OCR TTS", "IMX93")[index % 4]
        image = Image.new("RGB", (320, 48), "white")
        ImageDraw.Draw(image).text((6, 8), label, fill="black", font=ImageFont.load_default())
        # Matches OpenOCR RecDynamicResize for the public mobile reference model.
        resized = cv2.resize(np.asarray(image), (320, 48)).astype("float32")
        sample = (resized.transpose(2, 0, 1)[None] / 255.0 - 0.5) / 0.5
        start = time.perf_counter_ns()
        prediction = session.run(None, {input_name: sample})[0]
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        if prediction.ndim != 3 or not np.isfinite(prediction).all():
            raise RuntimeError("invalid OCR output")
        ids = prediction.argmax(axis=2)[0]
        decoded, previous = [], -1
        for value in ids:
            current = int(value)
            if current != 0 and current != previous:
                decoded.append(characters[current] if current < len(characters) else "?")
            previous = current
        rows.append({"iteration": index, "service_ms": round(elapsed_ms, 4),
                     "rendered_input": label, "decoded": "".join(decoded)})
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "calls.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {"execution": "host CPU ONNX Runtime", "model_sha256": digest(args.model),
                "dictionary_sha256": digest(args.dict), "image_shape": [1, 3, 48, 320],
                "input": "rendered synthetic text; recognition quality is not assessed",
                "scope": "public OpenOCR mobile reference; not a team checkpoint or M33 measurement"}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"saved {len(rows)} CPU calls to {args.out}")


if __name__ == "__main__":
    main()
