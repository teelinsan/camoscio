## Credits to gsarti, this is a modified version of his code from the original repo of it5 https://github.com/gsarti/it5/

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

from transformers import (
    HfArgumentParser,
)

from src.utils import generate_prompt, split_token

from transformers.utils import check_min_version
from transformers.utils.versions import require_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.15.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    from_flax: bool = field(
        default=False,
        metadata={
            "help": "If true, the model will be loaded from a saved Flax checkpoint."
        },
    )
    peft_model: str = field(
        default="teelinsan/camoscio-7b-llama",
        metadata={
            "help": "PEFT checkpoint to load"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    prefix_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the prefix (via the datasets library)."}
    )
    dataset_split: Optional[str] = field(
        default="test", metadata={"help": "The split of the dataset to use (via the datasets library)."}
    )
    source_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    target_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    num_beams: Optional[int] = field(
        default=4,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )

    temperature: Optional[float] = field(
        default=0.2,
        metadata={
            "help": "Temperature to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )

    top_p: Optional[float] = field(
        default=0.75,
        metadata={
            "help": "Top-P to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )

    top_k: Optional[int] = field(
        default=40,
        metadata={
            "help": "Top-k to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )

    no_repeat_ngram_size: Optional[int] = field(
        default=0,
        metadata={
            "help": "No repeat ngram size to use for evaluation. This argument will be passed to ``model.generate``, "
        },
    )

    batch_size: Optional[int] = field(
        default=8,
        metadata={"help": "Batch size used for inference."},
    )
    output_dir: Optional[str] = field(
        default=".",
        metadata={"help": "Output dir."},
    )
    prompt: str = field(
        default=None,
        metadata={"help": "This is the instruction to be used for the task. E.g., Traduci da Italiano ad Inglese."},
    )


name_mapping = {
    "fst": ("formal", "informal"),
    "hg": ("text", "target"),
    "ns": ("source", "target"),
    "qa": ("source", "target"),
    "qg": ("text", "target"),
    "st_g2r": ("full_text", "headline"),
    "st_r2g": ("full_text", "headline"),
    "wits": ("source", "summary"),
}



def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    model_shortname = model_args.model_name_or_path if "/" not in model_args.model_name_or_path else \
    model_args.model_name_or_path.split("/")[-1]

    print(f"Loading model {model_args.model_name_or_path} and tokenizer from {model_args.tokenizer_name}...")
    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_flax=model_args.from_flax,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token,
        load_in_8bit=True,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(model, model_args.peft_model)

    tokenizer = LlamaTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token,
    )
    tokenizer.pad_token = 0
    #tokenizer.padding_side = "left"


    model.resize_token_embeddings(len(tokenizer))
    print(f"Loading dataset {data_args.dataset_name} with config {data_args.dataset_config_name}")
    dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        use_auth_token=model_args.use_auth_token
    )
    column_names = dataset[data_args.dataset_split].column_names

    # Get the column names for input/target.
    dataset_columns = name_mapping.get(data_args.dataset_config_name, None)
    if data_args.source_column is None:
        source_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        source_column = data_args.source_column
        if source_column not in column_names:
            raise ValueError(
                f"--source_column' value '{data_args.source_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.target_column is None:
        target_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        target_column = data_args.target_column
        if target_column not in column_names:
            raise ValueError(
                f"--target_column' value '{data_args.target_column}' needs to be one of: {', '.join(column_names)}"
            )

    prompt_len = len(tokenizer.tokenize(generate_prompt(data_args.prompt))) + 1
    tot_len = prompt_len + data_args.max_source_length
    def preprocess_function(examples):
        # remove pairs where at least one record is None
        inputs, targets = [], []
        for i in range(len(examples[source_column])):
            if examples[source_column][i] is not None and examples[target_column][i] is not None:
                inputs.append(examples[source_column][i])
                targets.append(examples[target_column][i])
        inputs = [generate_prompt(data_args.prompt, input) for input in inputs]
        #model_inputs = tokenizer(inputs, max_length=tot_len, padding="max_length", truncation=True)
        model_inputs = tokenizer(inputs, max_length=tot_len, truncation=True)
        return model_inputs

    predict_dataset = dataset[data_args.dataset_split].map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on prediction dataset",
    )
    print(f"Example: {predict_dataset[0]}")
    predict_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    dataloader = torch.utils.data.DataLoader(predict_dataset, batch_size=data_args.batch_size)
    gen_kwargs = {
        #"max_length": tot_len + data_args.max_target_length,
        "max_new_tokens": data_args.max_target_length,
        "num_beams": data_args.num_beams,
        "temperature": data_args.temperature,
        "top_p": data_args.top_p,
        "top_k": data_args.top_k,
        "no_repeat_ngram_size": data_args.no_repeat_ngram_size,
    }
    generation_config = GenerationConfig(**gen_kwargs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    print(f"Inferencing...")
    predictions = []
    fname = f"{model_shortname}_{data_args.prefix_config_name}_{data_args.dataset_split}.txt"
    out_path = os.path.join(data_args.output_dir, fname)

    for i, batch in enumerate(tqdm(dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}

        out = model.generate(**batch, generation_config=generation_config)
        outputs = tokenizer.batch_decode(out.to("cpu"), skip_special_tokens=True)
        f_outputs = ["" if len(o.split(split_token)) < 2 else o.split(split_token)[1].strip() for o in outputs]
        outputs = f_outputs
        if i == 0:
            print(outputs[:2])
        with open(out_path, 'a') as f:
            oo = outputs[0].replace("\n", " ")
            f.write(f"{oo}\n")
        #predictions.extend(outputs)
    assert len(predictions) == len(predict_dataset)


if __name__ == "__main__":
    main()