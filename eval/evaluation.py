## Credits to gsarti, this is a modified version of his code from the original repo of it5 https://github.com/gsarti/it5/

import os

MODEL_NAME = "linhvu/decapoda-research-llama-7b-hf"
ADAPTER_NAME = "teelinsan/camoscio-7b-llama"
BATCH_SIZE = 1
OUT_DIR = "./results"

command1 = f"""python3 inference.py \\
    --model_name_or_path {MODEL_NAME} \\
    --peft_model {ADAPTER_NAME} \\
    --batch_size {BATCH_SIZE} \\
    --output_dir {OUT_DIR} \\
    --dataset_name it5/datasets \\
"""


settings = {
    "fst_i2f": {
        "source_len": 128,
        "target_len": 128,
        "config": "fst",
        "suffix": "informal-to-formal",
        "source_column": "informal",
        "target_column": "formal",
        "splits": ["test_0"],
        "prompt": "Dato il seguente testo scritto in modo informale, riscrivilo in modo formale."
    },
    "fst_f2i": {
        "source_len": 128,
        "target_len": 128,
        "config": "fst",
        "suffix": "formal-to-informal",
        "source_column": "formal",
        "target_column": "informal",
        "splits": ["test_0", "test_1", "test_2", "test_3"],
        "prompt": "Dato il seguente testo scritto in modo formale, riscrivilo in modo informale."
    },
    "ns": {
        "source_len": 512,
        "target_len": 128,
        "suffix": "news-summarization",
        "source_column": "source",
        "target_column": "target",
        "splits": ["test_fanpage", "test_ilpost"],
        "prompt": "Dopo aver letto il testo qui sotto, riassumilo adeguatamente.",
    },
    "qa": {
        "source_len": 512,
        "target_len": 64,
        "suffix": "question-answering",
        "source_column": "source",
        "target_column": "target",
        "prompt": "Dopo aver letto il paragrafo qui sotto, rispondi correttamente alla successiva domanda.",
    }
}

def main():
    with open("eval_new.sh", "w") as f:
        for task, value in settings.items():
            source_len = value["source_len"]
            target_len = value["target_len"]
            config = value.get("config", task)
            suffix = value.get("suffix", task)
            source_column = value["source_column"]
            target_column = value["target_column"]
            prompt = value["prompt"]
            temperature = value.get("temperature", 0.0)
            for split in value.get("splits", ["test"]):
                print(f"Evaluating dataset {config} on split {split}...")
                command2 = f"""    --dataset_config {config} \\
                --dataset_split {split} \\
                --source_column={source_column} \\
                --target_column={target_column} \\
                --max_source_length {source_len} \\
                --max_target_length {target_len} \\
                --prompt "{prompt}" \\
                --prefix_config_name {suffix} \\
                """
                command = command1 + command2
                f.write(command + "\n")

if __name__ == '__main__':
    main()