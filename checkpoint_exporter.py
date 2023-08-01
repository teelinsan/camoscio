# Imported from the orginal Alpaca LoRA repo

import torch
import fire
from peft import PeftModel
from transformers import LlamaForCausalLM



def merge_checkpoints(base_model="decapoda-research/llama-7b-hf", lora_model="teelinsan/camoscio-7b-llama", save_path="./camoscio_merged_ckpt"):


    base_model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
    )

    first_weight = base_model.model.layers[0].self_attn.q_proj.weight
    first_weight_old = first_weight.clone()

    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_model,
        device_map={"": "cpu"},
        torch_dtype=torch.float16,
    )

    lora_weight = lora_model.base_model.model.model.layers[0].self_attn.q_proj.weight

    assert torch.allclose(first_weight_old, first_weight)

    # merge weights
    for layer in lora_model.base_model.model.model.layers:
        layer.self_attn.q_proj.merge_weights = True
        layer.self_attn.v_proj.merge_weights = True

    lora_model.train(False)

    # did we do anything?
    assert not torch.allclose(first_weight_old, first_weight)

    lora_model_sd = lora_model.state_dict()
    deloreanized_sd = {
        k.replace("base_model.model.", ""): v
        for k, v in lora_model_sd.items()
        if "lora" not in k
    }

    LlamaForCausalLM.save_pretrained(
        base_model, save_path, state_dict=deloreanized_sd, max_shard_size="400MB"
    )


if __name__ == "__main__":
    fire.Fire(merge_checkpoints)