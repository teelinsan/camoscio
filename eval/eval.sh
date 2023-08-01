#! /bin/bash

python3 inference.py \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --peft_model teelinsan/camoscio-7b-llama \
    --batch_size 1 \
    --output_dir ./results \
    --dataset_name it5/datasets \
    --dataset_config fst \
                --dataset_split test_0 \
                --source_column=informal \
                --target_column=formal \
                --max_source_length 128 \
                --max_target_length 128 \
                --prompt "Dato il seguente testo scritto in modo informale, riscrivilo in modo formale."
                
python3 inference.py \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --peft_model teelinsan/camoscio-7b-llama \
    --batch_size 1 \
    --output_dir ./results \
    --dataset_name it5/datasets \
    --dataset_config fst \
                --dataset_split test_0 \
                --source_column=formal \
                --target_column=informal \
                --max_source_length 128 \
                --max_target_length 128 \
                --prompt "Dato il seguente testo scritto in modo formale, riscrivilo in modo informale."
                
python3 inference.py \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --peft_model teelinsan/camoscio-7b-llama \
    --batch_size 1 \
    --output_dir ./results \
    --dataset_name it5/datasets \
    --dataset_config fst \
                --dataset_split test_1 \
                --source_column=formal \
                --target_column=informal \
                --max_source_length 128 \
                --max_target_length 128 \
                --prompt "Dato il seguente testo scritto in modo formale, riscrivilo in modo informale."
                
python3 inference.py \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --peft_model teelinsan/camoscio-7b-llama \
    --batch_size 1 \
    --output_dir ./results \
    --dataset_name it5/datasets \
    --dataset_config fst \
                --dataset_split test_2 \
                --source_column=formal \
                --target_column=informal \
                --max_source_length 128 \
                --max_target_length 128 \
                --prompt "Dato il seguente testo scritto in modo formale, riscrivilo in modo informale."
                
python3 inference.py \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --peft_model teelinsan/camoscio-7b-llama \
    --batch_size 1 \
    --output_dir ./results \
    --dataset_name it5/datasets \
    --dataset_config fst \
                --dataset_split test_3 \
                --source_column=formal \
                --target_column=informal \
                --max_source_length 128 \
                --max_target_length 128 \
                --prompt "Dato il seguente testo scritto in modo formale, riscrivilo in modo informale."
                
                
python3 inference.py \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --peft_model teelinsan/camoscio-7b-llama \
    --batch_size 1 \
    --output_dir ./results \
    --dataset_name it5/datasets \
    --dataset_config ns \
                --dataset_split test_fanpage \
                --source_column=source \
                --target_column=target \
                --max_source_length 512 \
                --max_target_length 128 \
                --prompt "Dopo aver letto il testo qui sotto, riassumilo adeguatamente."
                
python3 inference.py \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --peft_model teelinsan/camoscio-7b-llama \
    --batch_size 1 \
    --output_dir ./results \
    --dataset_name it5/datasets \
    --dataset_config ns \
                --dataset_split test_ilpost \
                --source_column=source \
                --target_column=target \
                --max_source_length 512 \
                --max_target_length 128 \
                --prompt "Dopo aver letto il testo qui sotto, riassumilo adeguatamente."
                
python3 inference.py \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --peft_model teelinsan/camoscio-7b-llama \
    --batch_size 1 \
    --output_dir ./results \
    --dataset_name it5/datasets \
    --dataset_config qa \
                --dataset_split test \
                --source_column=source \
                --target_column=target \
                --max_source_length 512 \
                --max_target_length 64 \
                --prompt "Dopo aver letto il paragrafo qui sotto, rispondi correttamente alla successiva domanda."

                
