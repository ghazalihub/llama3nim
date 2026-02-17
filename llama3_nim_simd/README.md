# Llama3.nim (SIMD Optimized)

A high-performance port of `Llama3.java` to Nim.

## Features
- **AVX2 & FMA Accelerated**: Hand-tuned SIMD kernels for Matrix-Vector multiplication and RoPE.
- **Multi-threaded**: Uses the `malebolgia` library for efficient task-based parallelism.
- **Quantization Support**: Supports GGUF Q4_0 and Q8_0 formats.
- **BPE Tokenizer**: Fully compatible Llama 3 tokenizer with regex support.
- **Memory Efficient**: Uses `mmap` for fast model loading and low memory overhead.

## Requirements
- Nim 2.2.0 or newer.
- CPU with AVX2, FMA, and F16C support.
- `nimsimd` and `malebolgia` Nim packages.

## Compilation
To compile with maximum performance:
```bash
nim c -d:danger --threads:on --passC:"-mavx2 -mfma -mf16c" llama3.nim
```

## Usage
```bash
./llama3 --model path/to/llama3-q4_0.gguf --prompt "Hello, how are you?"
```
Check `--help` for more options.
