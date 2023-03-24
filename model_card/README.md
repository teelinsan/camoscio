---
license: openrail
language:
- it
---

# Camoscio: An Italian instruction-tuned LLaMA

## Usage

Check the Github repo with code: https://github.com/teelinsan/camoscio

```python
from peft import PeftModel
from transformers import LLaMATokenizer, LLaMAForCausalLM, GenerationConfig

tokenizer = LLaMATokenizer.from_pretrained("decapoda-research/llama-7b-hf")
model = LLaMAForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf",
    load_in_8bit=True,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, "teelinsan/camoscio-7b-llama")
```

Generation Example: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/teelinsan/camoscio/blob/master/notebooks/camoscio-lora.ipynb) 