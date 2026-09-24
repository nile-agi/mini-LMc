
# mini-LMc

A from-scratch LLM inference engine, in C — cross-platform (Linux, macOS, Windows).

## Requirements
- CMake >= 3.16
- A C11 compiler (gcc, clang, or MSVC)

## Build
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build
    ctest --test-dir build --output-on-failure

## Download a test model
    mkdir -p models
    curl -L -o models/model.gguf "https://huggingface.co/<repo>/resolve/main/<file>.gguf"

Any GGUF v3 file works — verified against real Qwen2.5 (291 tensors, mixed Q4_K/Q5_0/Q6_K/Q8_0/F32).

## Usage
    ./build/minilmc show models/model.gguf [max_tensors]
    ./build/minilmc info models/model.gguf <tensor_name>

## Status
Phase 1 (GGUF reader) complete and verified: byte-exact parsing, tensor-range
validation, safe tensor-data accessor. 

Dequantization kernels (Phase 2) next.