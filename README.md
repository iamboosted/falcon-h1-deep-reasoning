[README.md](https://github.com/user-attachments/files/28279016/README.md)
---
license: apache-2.0
base_model: tiiuae/Falcon-H1-1.5B-Deep-Instruct
tags:
  - falcon
  - falcon-h1
  - mamba
  - mamba2
  - hybrid
  - qlora
  - lora
  - math
  - reasoning
  - fine-tune
model_type: falcon_h1
pipeline_tag: text-generation
datasets:
  - meta-math/MetaMathQA
---

# Falcon-H1-1.5B-Deep-Reasoning

**QLoRA math reasoning adapter for the deepest, narrowest Mamba-2 hybrid model.**

This is a LoRA adapter trained on [tiiuae/Falcon-H1-1.5B-Deep-Instruct](https://huggingface.co/tiiuae/Falcon-H1-1.5B-Deep-Instruct) using [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) to improve math and reasoning capabilities. The base model has 66 layers at only 1.5B parameters — the deepest, narrowest model in the Falcon-H1 family — and already performs on par with many 7-10B models.

## Results

20-question math/reasoning benchmark, greedy decoding:

| Model | Score | Change |
|---|---|---|
| Falcon-H1-1.5B-Deep-Instruct (base) | 10/20 (50%) | — |
| **+ Reasoning adapter** | **13/20 (65%)** | **+30% relative** |

### What Improved

The adapter's primary effect was eliminating the base model's degenerate repetition loops. The base instruct model frequently fell into patterns like `"Three friends, Andy, Beth..."` or `"Three\nThree\nThree..."` instead of solving problems. The reasoning adapter replaced every one of these failures with actual step-by-step math chains.

**8 questions gained:** fuel calculation, geometric sequences, prime counting, syllogistic logic, integer sums, hexagon diagonals, GCD, and algebra — all previously repetition-loop failures.

**5 questions lost:** 2 are parser artifacts (model outputs correct answer with formatting the extractor misreads), 3 are genuine regressions on arithmetic.

### Reasoning Quality

The fine-tuned model shows clear reasoning patterns:
- Uses `<|begin_of_thought|>` tags for structured reasoning
- Applies formulas explicitly: `n(n+1)/2`, difference of squares `(a+b)(a-b)`
- Shows work step by step before arriving at answers

## Training Details

- **Method:** QLoRA (4-bit NF4, double quantization)
- **Dataset:** MetaMathQA, 2000 examples, 1 epoch
- **LoRA rank:** 16, alpha 32
- **Target modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `in_proj` (attention + Mamba)
- **Sequence length:** 512
- **Batch size:** 2 (effective 4 with grad accumulation)
- **Learning rate:** 2e-4 with warmup
- **Training time:** ~70 minutes on RTX 3060 12GB
- **Final loss:** ~0.28, token accuracy ~91%

### Note on Mamba LoRA Targets

PEFT explicitly blocks `out_proj` and `conv1d` as LoRA targets for Mamba-based models. This aligns with TII's documentation that `out_proj` weights are used directly within the Mamba kernel and should not be modified. The adapter targets `in_proj` for the Mamba layers instead.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=["mamba.out_proj"],
)

model = AutoModelForCausalLM.from_pretrained(
    "tiiuae/Falcon-H1-1.5B-Deep-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained("tiiuae/Falcon-H1-1.5B-Deep-Instruct")

model = PeftModel.from_pretrained(model, "iAmBoosted/falcon-h1-1.5b-deep-reasoning")

prompt = "Solve step by step: If 4 workers can build a wall in 6 days, how many days would it take 3 workers?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Hardware

- **Training:** RTX 3060 12GB, QLoRA 4-bit, ~70 minutes
- **VRAM usage:** ~4GB during training
- **Requires:** `mamba-ssm` CUDA kernels for reasonable training speed (naive path is ~10x slower)
- **Container:** `nvcr.io/nvidia/pytorch:24.12-py3`

## Part of a Series

This is one of several experiments exploring non-transformer architectures:
- [Falcon-H1 SLERP Merge](https://huggingface.co/iAmBoosted/falcon-h1-7b-instruct-x-h1r-slerp) — First SLERP merge of Mamba-2 hybrids
- [Zamba2 SLERP Merge](https://github.com/iamboosted/Zamba2-SLERP-Merge) — Weight-sharing breaks standard merge tooling

## License

Apache 2.0 (inherited from base model). Training data (MetaMathQA) is MIT licensed.
