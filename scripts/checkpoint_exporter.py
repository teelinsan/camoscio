# Imported from the orginal Alpaca LoRA repo

import torch
import fire
from peft import PeftModel
from transformers import LlamaForCausalLM



def merge_checkpoints(base_model="decapoda-research/llama-7b-hf", lora_model="teelinsan/camoscio-7b-llama", save_path="./camoscio_merged_ckpt"):
    """
    Utility function to merge a base LLaMA model with a LoRA weights.
    """

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
    )

    model = PeftModel.from_pretrained(
        model,
        lora_model,
        device_map={"": "cpu"},
        torch_dtype=torch.float16,
    )

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(save_path)


if __name__ == "__main__":
    fire.Fire(merge_checkpoints)