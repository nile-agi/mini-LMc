
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
    ./build/minilmc forward models/model.gguf <token_id> [token_id ...]
    ./build/minilmc generate models/model.gguf "Your prompt" [max_tokens] [temperature] [top_k]
    ./build/minilmc chat models/model.gguf [max_tokens] [temperature] [top_k]

`generate` defaults to 128 tokens, greedy decoding (`temperature` 0), and
top-k 40 when temperature is positive. Set top-k to 0 to sample from the full
vocabulary. Generation stops at the model's EOS token or its context limit.
Quote prompts containing spaces. The command accepts prompt text as supplied;
for instruct/chat models, include the model's expected chat template in the
prompt.

`chat` starts an interactive terminal conversation with a Qwen2 chat model.
Enter a message at each `You>` prompt; type `/exit` to leave. It keeps prior
turns in the model context until the context limit is reached. The chat command
uses the Qwen message markers and a default helpful-assistant system prompt.

## Status
GGUF tensor lookup, alignment validation, and
scalar F32/F16/BF16/Q8_0 dequantization with fused GEMV.

Qwen2 forward passes with embedding,
RMSNorm, half-split RoPE, GQA attention, KV caching, SwiGLU, Q/K/V biases, and
logits. Block 3 adds GGUF byte-level BPE tokenization, UTF-8 token decoding,
greedy and temperature/top-k sampling, and the `generate` command. The
tokenizer follows the Qwen2/GPT-2 pre-tokenization pattern and reads the
vocabulary and merge ranks from GGUF metadata.
