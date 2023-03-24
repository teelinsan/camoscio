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
model = PeftModel.from_pretrained(model, "teelinsan/camoscio-7b-lora")
```


## Data

We translated the [alpaca_data.json](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json) to portuguese using ChatGPT. Even if this translation was not the best, the tradeoff between costs and results were. We paid around US$ 8.00 to translate the full dataset to portuguese.
If you want to know more about how the dataset was built go to: [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca).