# cracked-inference

A from-scratch toolkit for squeezing modern large language models down to roughly **1.58 bits per weight** and running them on a single consumer GPU. The project pairs a small set of hand-tuned CUDA kernels with a streaming quantizer and a drop-in inference runtime, so an 8B-parameter model that normally needs ~16 GB of VRAM can sit comfortably inside the 12 GB budget of an RTX 3080 Ti while still producing coherent text.

The repository is intentionally lean: a few CUDA source files, a few Python scripts, and no heavyweight framework on top. Everything is meant to be readable end-to-end.

---

## Why this exists

Most quantization stacks ship as opaque binaries or as part of a much larger framework. This project takes the opposite stance and exposes every step of the pipeline so the trade-offs are visible:

- **Hadamard rotation** to crush activation and weight outliers before quantization, following the QuaRot / SpinQuant recipe.
- **Group-wise ternary weights** in the spirit of BitNet b1.58, where every weight is collapsed to one of three values: `-1`, `0`, or `+1`, scaled by a per-group magnitude.
- **A 1-bit residual correction term** ("QJL" sign bits with a per-group mean-absolute-error scale) that recovers most of the accuracy lost to ternarization without meaningfully growing the on-disk size.
- **Custom Ampere kernels** for the Fast Walsh–Hadamard Transform and for packing/unpacking the bit-level weights, written specifically for compute capability 8.6 (RTX 30-series consumer cards).

The combined format is referred to throughout the code as **TurboQuant**, and the resulting model directories are called **vacuum-sealed** checkpoints.

---

## What you get

The repository contains four things you can actually run:

1. A **CUDA extension** (`cracked_quant`) that exposes a fused FWHT, plus pack and unpack routines for ternary weights and 1-bit residuals.
2. A **streaming quantizer** that walks a Hugging Face causal-LM checkpoint one linear layer at a time, rotates and quantizes it on the GPU, and writes a packed `safetensors` file per layer to disk.
3. An **inference engine** that loads the original model architecture from Hugging Face, swaps every linear layer for a `TurboLinear` module backed by the packed tensors, and runs token-by-token generation with sampling, repetition penalty, and an n-gram loop guard.
4. A **benchmark and debug harness** for sanity-checking kernel correctness and measuring throughput against `torch.matmul`.

---

## Hardware and software targets

This project is currently tuned for a very specific environment:

- **GPU:** NVIDIA Ampere consumer cards (RTX 3080, 3080 Ti, 3090, A40-class). The kernels are compiled with `sm_86` flags and assume 12–24 GB of VRAM.
- **OS:** Ubuntu 22.04 / 24.04, including WSL2 on Windows. A bootstrap script is provided for fresh WSL or Docker containers.
- **Toolchain:** A recent CUDA toolkit (12.x or 13.x), a C++17-capable compiler, Python 3.10+, and PyTorch built against the same CUDA major version.

The CUDA kernels will not magically run on Hopper, Ada, or Turing without retuning. If you want to target a different architecture, change the `gencode` flag in `setup.py` and the build helpers and rebuild from source.

---

## How the pipeline works, in plain English

Imagine a single weight matrix inside a transformer. It contains millions of floating-point numbers, almost all of them tiny, but a handful are huge "outlier" values that dominate the average magnitude. Naively rounding each weight to the nearest of `-1`, `0`, or `+1` would crush almost everything to zero because those few outliers drag the scale upward.

The pipeline avoids that collapse in three stages:

1. **Smoothing.** A block-diagonal Hadamard transform is applied to each row of the weight matrix. Hadamard transforms are orthonormal, so this is a lossless reshaping, but it spreads the energy of every outlier across many coordinates. The result is a much more Gaussian-looking distribution that ternary quantization can handle gracefully. Crucially, the same transform can be applied to activations at inference time, so the rotation cancels out mathematically and costs nothing in accuracy.

2. **Ternarization with residuals.** The rotated matrix is split into small groups along its input dimension. For each group, the average absolute value is recorded as a scale, every weight is rounded to `{-1, 0, +1}`, and the leftover error is summarized by a single sign bit per coordinate plus one extra magnitude scale per group. That sign-and-magnitude pair is the "1-bit QJL residual" referenced in the code.

3. **Packing.** The ternary codes are packed four-to-a-byte and the residual signs are packed eight-to-a-byte, so a layer that started as 16-bit floats ends up taking roughly 2 bits per weight on disk and in VRAM.

When inference runs, the `TurboLinear` module reverses these steps just-in-time inside a CUDA kernel: it unpacks the ternary and residual bits, applies the per-group scales, undoes the Hadamard rotation, and feeds the reconstructed weight matrix into a standard matmul against the activations. Optionally the unpacked weight can be cached in VRAM after the first forward pass to trade memory for speed.

---

## Layout of the repository

A short tour of the files most likely to matter:

- `fused_fwht.cu` — fused Fast Walsh-Hadamard Transform kernel for FP16 tensors, last-dim sizes up to 1024.
- `ternary_packing.cu` — pack and unpack kernels for ternary weights and 1-bit QJL residual signs.
- `cracked_quant.cpp` — pybind11 bindings that glue the two CUDA files into a single Python module.
- `setup.py` — ahead-of-time build script for the extension. The same source files can also be JIT-compiled via `torch.utils.cpp_extension.load`, which is what most of the Python entry points do.
- `quantize_real_llama.py` — the streaming quantizer for real Hugging Face checkpoints. Writes one `safetensors` file per linear layer plus a `report.json` with per-type SNR and compression statistics.
- `quantize_model.py` — a single-layer synthetic demo that walks through every stage of the pipeline with verbose logging. Useful for understanding what the streaming quantizer is doing internally.
- `inference.py` — the runtime. Loads a vacuum-sealed directory, replaces matching linear layers with `TurboLinear`, and exposes a small interactive REPL with token-per-second reporting.
- `debug_layers.py` — numerical integrity harness that compares a single original linear layer against its quantized counterpart on identical inputs.
- `benchmark.py` — speed comparison between the fused FWHT kernel and a plain `torch.matmul` against a Hadamard matrix.
- `bootstrap_wsl_docker.sh` — opinionated setup script that installs system packages, creates a virtual environment, and pulls the right CUDA-flavored PyTorch wheel.

---

## Getting started

A typical first run on a clean machine looks like this:

1. Run the bootstrap script (or replicate it manually): install build tools, create a virtual environment in `.venv`, and install PyTorch and the project requirements.
2. Set a Hugging Face token if you plan to quantize a gated model such as Llama 3, then launch the streaming quantizer pointed at the model of your choice. The output goes into a `vacuum_sealed_*` directory next to the source files.
3. Once quantization finishes, point the inference script at the same directory and start chatting. The runtime will rebuild the CUDA extension on first use, load the original tokenizer and architecture from Hugging Face, and substitute the packed layers automatically.

There is no separate config file. Group size, Hadamard block size, the option to keep the first and last few transformer blocks in full precision, and similar knobs are all command-line arguments on the quantizer and inference scripts.

---

## What is *not* covered

This project is deliberately scoped down. It does not include:

- A training or fine-tuning loop. Quantization is post-training only.
- Support for non-CUDA backends. CPU and ROCm are out of scope.
- Pre-built wheels. The CUDA extension is always built locally, either ahead of time via `setup.py` or just-in-time the first time a script imports it.
- Distribution-ready model artifacts. The vacuum-sealed directories produced by the quantizer are intentionally excluded from version control because they are large and easy to regenerate.

---

## Status

The project is research-grade and evolves quickly. Expect the kernel flags, the on-disk packed format, and the script interfaces to change between revisions. If you depend on a specific layout, pin the commit you are using and regenerate vacuum-sealed checkpoints whenever you upgrade.

Contributions, especially around portability to other Ampere and Ada GPUs, additional residual schemes, and faster fused matmul-with-unpack kernels, are welcome.
