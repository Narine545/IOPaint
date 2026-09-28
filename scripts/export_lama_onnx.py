"""Export the LaMa erase model to ONNX and (optionally) benchmark it.

Why: on a CPU-only machine (Ryzen 5700G, integrated Vega) the PyTorch
TorchScript build of LaMa is the slowest part of the pipeline. ONNX Runtime
usually gives 1.3-2x on the same CPU, and it also unlocks DirectML, which can
use the integrated GPU.

Usage (Windows, inside the IOPaint environment):

    python scripts/export_lama_onnx.py --benchmark

The script downloads the usual LaMa weights (the same file IOPaint already
uses), traces them at a fixed resolution and writes:

    <model-dir>/erase_models/lama.onnx

Send back the printed report (the last block) so the numbers can be used to
tune the defaults.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from iopaint.const import DEFAULT_MODEL_DIR
from iopaint.download import cli_download_model  # noqa: F401  (keeps CLI parity)
from iopaint.runtime import setup_model_dir


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(DEFAULT_MODEL_DIR),
        help="Where models are stored (same value you pass to iopaint start)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .onnx path (default: <model-dir>/erase_models/lama.onnx)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Resolution used for tracing; LaMa is fully convolutional so the "
        "exported graph keeps dynamic height/width",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Compare PyTorch vs ONNX Runtime on this machine",
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--bench-size",
        type=int,
        default=1024,
        help="Resolution used for the benchmark (a realistic photo crop)",
    )
    return parser.parse_args()


class LamaWrapper(torch.nn.Module):
    """image/mask in [0,1] float, NCHW -> inpainted image in [0,1]."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, mask):
        masked = image * (1 - mask)
        return self.model(masked, mask)


def load_lama(model_dir: Path):
    from iopaint.model.lama import LaMa

    model = LaMa(device=torch.device("cpu"))
    # LaMa keeps the traced module in .model
    return model.model


def export(args) -> Path:
    model_dir = setup_model_dir(args.model_dir)
    output = args.output or (Path(model_dir) / "erase_models" / "lama.onnx")
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Loading LaMa (this downloads ~200MB on the first run)...")
    net = load_lama(Path(model_dir))
    net.eval()

    wrapper = LamaWrapper(net).eval()
    image = torch.rand(1, 3, args.size, args.size)
    mask = (torch.rand(1, 1, args.size, args.size) > 0.9).float()

    print(f"[2/3] Exporting to {output} (opset {args.opset})...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (image, mask),
            str(output),
            input_names=["image", "mask"],
            output_names=["output"],
            opset_version=args.opset,
            dynamic_axes={
                "image": {0: "batch", 2: "height", 3: "width"},
                "mask": {0: "batch", 2: "height", 3: "width"},
                "output": {0: "batch", 2: "height", 3: "width"},
            },
        )
    size_mb = output.stat().st_size / 1024 / 1024
    print(f"[3/3] Done: {output} ({size_mb:.1f} MB)")
    return output


def benchmark(args, onnx_path: Path):
    import onnxruntime as ort

    n = args.bench_size
    image = np.random.rand(1, 3, n, n).astype(np.float32)
    mask = (np.random.rand(1, 1, n, n) > 0.9).astype(np.float32)

    print(f"\nBenchmark at {n}x{n}, {args.runs} runs")

    net = LamaWrapper(load_lama(Path(args.model_dir))).eval()
    torch_image = torch.from_numpy(image)
    torch_mask = torch.from_numpy(mask)
    with torch.no_grad():
        net(torch_image, torch_mask)  # warmup
        start = time.time()
        for _ in range(args.runs):
            net(torch_image, torch_mask)
        torch_ms = (time.time() - start) / args.runs * 1000
    print(f"  PyTorch CPU      : {torch_ms:8.1f} ms")

    results = {"pytorch-cpu": torch_ms}
    for provider in ("CPUExecutionProvider", "DmlExecutionProvider"):
        if provider not in ort.get_available_providers():
            print(f"  {provider:<17}: not available")
            continue
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(str(onnx_path), opts, providers=[provider])
        feed = {"image": image, "mask": mask}
        sess.run(None, feed)  # warmup
        start = time.time()
        for _ in range(args.runs):
            sess.run(None, feed)
        ms = (time.time() - start) / args.runs * 1000
        results[provider] = ms
        speedup = torch_ms / ms if ms else 0
        print(f"  {provider:<17}: {ms:8.1f} ms  ({speedup:.2f}x vs PyTorch)")

    print("\n----- copy everything below this line -----")
    import platform

    print(f"machine   : {platform.processor() or platform.machine()}")
    print(f"python    : {sys.version.split()[0]}")
    print(f"torch     : {torch.__version__}")
    print(f"onnxruntime: {ort.__version__} {ort.get_available_providers()}")
    print(f"size      : {n}x{n}, runs: {args.runs}")
    for key, value in results.items():
        print(f"{key}: {value:.1f} ms")
    print("----- copy everything above this line -----")


def main():
    args = parse_args()
    onnx_path = export(args)
    if args.benchmark:
        benchmark(args, onnx_path)


if __name__ == "__main__":
    main()
