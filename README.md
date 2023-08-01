# 🇮🇹🦙🤏  Camoscio: An Italian instruction-tuned LLaMA

This repository contains code for Camoscio, an Italian instruction-tuned LLaMA based on [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) and trained with [low-rank adaptation (LoRA)](https://arxiv.org/pdf/2106.09685.pdf). The companion paper of this repo is available [at the following url](https://arxiv.org/abs/2307.16456).

Part of the code in the repo is based on [Alpaca-LoRA](https://github.com/tloen/alpaca-lora) and [cabrita](https://github.com/22-hours/cabrita).
Following these previous approaches, we translated the Stanford Alpaca instruction-tuning dataset into Italian using the ChatGPT API.
We provide the translated dataset (`camoscio_data.json` - Now also on [Hugging Face's datasets](https://huggingface.co/datasets/teelinsan/camoscio)), the model (available on the [Hugging Face's hub](https://huggingface.co/teelinsan/camoscio-7b-llama)) and the code to reproduce the results.

The model provide zero-shot performance comparable with existing trained models for Italian (see [our paper for the evaluation](https://arxiv.org/abs/2307.16456)) and can run [on a Raspberry Pi](https://twitter.com/miolini/status/1634982361757790209) (for research).

To finetune the model on the Italian dataset we adapted the scripts from [cabrita](https://github.com/22-hours/cabrita) and run the training on a single 3090 for 1 day (see details below).

Please note that it is highly possible that the model output contains biased, conspiracist, offensive, or otherwise inappropriate and potentially harmful content.
The model is intended for research purposes only and should be used with caution at your own risk. Production usage is not allowed.



- [1] LLaMA: Open and Efficient Foundation Language Models. Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample. https://arxiv.org/abs/2302.13971v1

- [2] Self-Instruct: Aligning Language Model with Self Generated Instructions. Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi. https://arxiv.org/abs/2212.10560

## 🖥️ Demo - Talk with Camoscio [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/teelinsan/camoscio/blob/master/notebooks/camoscio-gradio.ipynb)

Clic the button "Open in Colab" above to open the notebook in Google Colab and try out the demo in gradio!

[![gradio.png](assets%2Fgradio.png)](https://colab.research.google.com/github/teelinsan/camoscio/blob/master/notebooks/camoscio-gradio.ipynb)

## 📚 How to use
We provide an example notebook on how to load and use the model [here](notebooks/camoscio-lora.ipynb).

```python
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
model = LlamaForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf",
    load_in_8bit=True,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, "teelinsan/camoscio-7b-llama")
```

## 🏋️ Reproduce the training 
### Setup

1. Install dependencies

```
pip install -r requirements.txt
```

2. If bitsandbytes doesn't work, [install it from source.](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md) Windows users can follow [these instructions](https://github.com/tloen/alpaca-lora/issues/17).

### Translate the Dataset to Italian (`translate_data.py`)
Download the dataset from [here](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json) and put it in the `data` folder. Then run:

```
python script/translate_data.py
```

### Train the model (`train.py`)
Just run the command (change hyperparameters as needed):

```
python scripts/train.py
```


### Checkpoint export

You can use the script `checkpoint_exporter.py` in this repo to merge the original weights of LLaMA with the Camoscio LoRA weights (LLaMA checkpoint + Camoscio LoRA checkpoints = Camoscio).
The model obtained with this procedure is the final Camoscio model that is equivalent to a LLaMA model tailored for Italian (same architecture and number of parameters) and can be used for finetuning on your downstream task.

To use the script just run the command `python scripts/checkpoint_exporter.py`

Checkout also the script `export_hf_checkpoint.py` from the original [Alpaca-LoRa repo](https://github.com/tloen/alpaca-lora/blob/main/export_hf_checkpoint.py) to export the checkpoint to the HuggingFace format or the script [export_state_dict_checkpoint.py](https://github.com/tloen/alpaca-lora/blob/main/export_state_dict_checkpoint.py) to export the checkpoint to the PyTorch format.

If you are looking for fast and efficient inference, take a look at projects like [llama.cpp](https://github.com/ggerganov/llama.cpp) or [alpaca.cpp](https://github.com/antimatter15/alpaca.cpp).


## 🔧 Finetune Camoscio on your own task in Italian

To finetune Camoscio on your own task in Italian you can modify the `train.py` script according to your needs or use the `finetune.py` script in the [Alpaca-LoRA](https://github.com/tloen/alpaca-lora) repo. We propose here the steps to follow for this second approach.

1. Prepare your dataset. Your dataset should be a json in the format `instruction, input, output`. See `data/camoscio_data.json` for an example.
2. Merge the checkpoints. Follow the instruction in the section of this readme and run `python scripts/checkpoint_exporter.py`.
3. Clone the [Alpaca-LoRA](https://github.com/tloen/alpaca-lora) repo and copy in the `templates` folder the template of Camoscio (`templates/camoscio.json`)
4. Run the `finetune.py` script in the [Alpaca-LoRA](https://github.com/tloen/alpaca-lora) repo.

```bash
python finetune.py \
    --base_model='{path_camoscio_merged_ckpt}' \
    --data_path='{your_dataset_path.json}'\
    --train_on_inputs=False \
    --num_epochs=10 \
    --cutoff_len=512 \
    --group_by_length \
    --output_dir='./outdir' \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r=16 \
    --micro_batch_size=8 \
    --prompt_template 'camoscio'
```


## ❓F.A.Q.

<details>
<summary> Why did you use ChatGPT for translation?</summary>
We tested several translation APIs including Google Translate, DeepL, and Azure Translator. Our preliminary tests found that these services also translated code snippets (e.g., for i in list -> per i in lista). 
We found that ChatGPT provided the best results. However, we are aware that the translation is not perfect and there are several artifacts in the dataset due to translation. There are also other <a href="https://github.com/gururise/AlpacaDataCleaned"> known issues</a> with the original Alpaca dataset.
</details>

<details>
<summary> Did you pay for the translation service? </summary>
Yes, we used ChatGPT which is a paid service.
</details>

<details>
<summary> Are you aware that some examples are not correct? </summary>
Yes, we put them specifically to show some limitations of the model (e.g., hallucinations, factual errors, etc...). The model is not perfect and it is just a first step in the direction of building an instruction-tuned model for Italian.
</details>

<details>
<summary> Is the model open? </summary>
The code in this repository and the dataset are open under the Apache 2.0 license. The weights finetuned for the Italian language (LoRA weights) are open under the RAIL license. However, the weights of the base model LLaMA are currently under the <a href="https://github.com/facebookresearch/llama/pull/184">"Non-commercial bespoke"</a> license which does not allow for usage in production.
</details>

For other questions, open an issue or contact me on [Twitter](https://twitter.com/teelinsan).

## 🔍 Evaluation

Results on the evaluation datasets are available [in the paper](https://arxiv.org/abs/2307.16456).

We provide all the results on the evaluation datasets (NewsSum-IT, SQuAD-IT, XFORMAL IT) in the folder `eval/results`.

These results are obtained by running the script `eval/eval.sh`. Final scores included in the paper are obtained via the notebook `compute_eval_scores.ipynb`

## 📝 Example outputs

**Instruction**: Riassumi la storia di Pinocchio.

**Alpaca-LoRA**: Once upon a time, there was a wooden puppet named Pinocchio who dreamed of becoming a real boy. He was created by a woodcarver named Geppetto, who loved him very much. Pinocchio was a mischievous puppet who often got into trouble, but he always tried to do the right thing in the end. One day, Pinocchio met a fairy who offered him the chance to become a real boy if he proved himself to be brave, truthful, and unselfish. Pinocchio accepted the fairy's offer and set off on a series of advent

**ChatGPT**: La storia di Pinocchio racconta di un burattino di legno, creato dal falegname Geppetto, che prende vita grazie alla fata turchina. Pinocchio ha un forte desiderio di diventare un bambino vero, ma cade in numerose avventure e inganni, spesso guidato dal suo istinto e dalla sua ingenuità. Nel corso della storia, Pinocchio viene ingannato da un gatto e una volpe, viene imprigionato e trasformato in un asino, ma alla fine dimostra il suo coraggio, la sua lealtà e la sua generosità salvando il padre e ottenendo la sua trasformazione in un bambino vero.

**Camoscio**: La storia di Pinocchio racconta di un bambino di nome Pinocchio che vive con i suoi genitori in un piccolo villaggio. Un giorno, Pinocchio incontra un burattino magico che gli dona la possibilità di diventare un vero bambino. Pinocchio inizia una serie di avventure mentre cerca di diventare un vero bambino. Durante il suo viaggio, Pinocchio fa amicizia con molti personaggi e impara molte lezioni di vita. Alla fine, Pinocchio raggiunge il suo obiettivo e diventa un vero bambino.

---

**Instruction**: Dimmi di più sul presidente della repubblica Italiana del 2019.

**Alpaca-LoRA**: The President of the Republic of Italy in 2019 is Sergio Mattarella. He was elected in 2015 and re-elected in 2018. He is the 12th President of the Republic of Italy.

**ChatGPT**: Il presidente della Repubblica Italiana del 2019 era Sergio Mattarella. Nato a Palermo nel 1941, Mattarella è stato eletto presidente nel 2015, succedendo a Giorgio Napolitano. Prima di diventare presidente, ha ricoperto importanti incarichi politici, tra cui quello di ministro della Difesa, della Pubblica Istruzione e della Giustizia. 
Durante il suo mandato, Mattarella è stato impegnato nella promozione dell'unità nazionale e nella difesa della Costituzione italiana. Ha anche sostenuto l'integrazione europea e la cooperazione internazionale. Nel 2018, ha svolto un ruolo importante nella formazione del governo italiano, che ha visto la nomina di un governo di coalizione tra il Movimento 5 Stelle e la Lega. 
Mattarella è stato molto attivo nella difesa dei diritti umani e della giustizia sociale, promuovendo iniziative per combattere la povertà e la disuguaglianza. 

**Camoscio**: Il presidente della Repubblica Italiana del 2019 è Sergio Mattarella, che è stato eletto il 31 gennaio 2015 e ha prestato giuramento il 31 gennaio 2015.

---

**Instruction**: Dimmi quattro parole che fanno rima con 'gatto'.

**Alpaca-LoRA**: Cat, hat, mat, rat.

**ChatGPT**: Piatto, matto, spratto, grattò.

**Camoscio**: 'Cattivo, cattivo, cattivo, cattivo.'


---

**Instruction**: Scrivi una breve biografia su Dante Alighieri

**Alpaca-LoRA**: Dante Alighieri (1265-1321) was an Italian poet, philosopher, and political figure. He is best known for his epic poem, The Divine Comedy, which is widely considered to be one of the greatest works of literature ever written. Dante was born in Florence, Italy, and spent much of his life in exile. He wrote The Divine Comedy while in exile, and it tells the story of his journey through Hell, Purgatory, and Heaven. Dante is considered to be one of the founders of the Italian language, and

**ChatGPT**: Dante Alighieri è stato uno dei più grandi poeti italiani del Medioevo. Nato a Firenze nel 1265, ha studiato nella prestigiosa Scuola di Santa Croce e ha mostrato fin da giovane una grande passione per la poesia. Ha scritto la sua prima opera, "Il Convivio", all'età di 30 anni, ma è soprattutto noto per la sua opera più famosa, "La Divina Commedia". Questo poema epico, scritto in terzine, descrive il viaggio immaginario di Dante attraverso l'Inferno, il Purgatorio e il Paradiso, e rappresenta una sintesi della cultura medievale, della filosofia e della teologia. Dante ha anche avuto una vita politica attiva, partecipando alla vita pubblica di Firenze e sostenendo la fazione guelfa. Tuttavia, dopo essere stato esiliato dalla città nel 1302, ha trascorso il resto della sua vita in esilio, scrivendo le sue opere più importanti e cercando di rientr

**Camoscio**: Dante Alighieri (1265-1321) è considerato uno dei più grandi poeti e scrittori italiani di tutti i tempi. Nato a Firenze, è conosciuto principalmente per il suo poema epico La Divina Commedia, che è considerato uno dei capolavori della letteratura occidentale.


---

**Instruction**: Dimmi tutte le regioni italiane in ordine alfabetico.

**Alpaca-LoRA**: Abruzzo, Basilicata, Calabria, Campania, Emilia-Romagna, Friuli-Venezia Giulia, Lazio, Liguria, Lombardia, Marche, Molise, Piemonte, Puglia, Sardegna, Sicilia, Toscana, Trentino-Alto Adige, Umbria, Veneto.

**ChatGPT**: Abruzzo, Basilicata, Calabria, Campania, Emilia-Romagna, Friuli-Venezia Giulia, Lazio, Liguria, Lombardia, Marche, Molise, Piemonte, Puglia, Sardegna, Sicilia, Toscana, Trentino-Alto Adige/Südtirol, Umbria, Valle d'Aosta/Vallée d'Aoste, Veneto.

**Camoscio**: Le regioni italiane in ordine alfabetico sono: Abruzzo, Basilicata, Calabria, Campania, Emilia-Romagna, Friuli-Venezia Giulia, Lazio, Liguria, Lombardia, Marche, Molise, Piemonte, Puglia, Sardegna, Sicilia, Toscana, Trentino-Alto Adige, Umbria, Valle d'Aosta, Veneto.

---

**Instruction**: Scrivi un programma che stampa i numeri da 1 a 100. Ma per i multipli di tre stampa 'Fizz' al posto del numero e per i multipli di cinque stampa 'Buzz'. Per i numeri che sono multipli sia di tre che di cinque stampa 'FizzBuzz'.

**Alpaca-LoRA**:

```python
for i in range(1, 101):
    if i % 3 == 0 and i % 5 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)
```

**ChatGPT**:

```python
for i in range(1, 101):
    if i % 3 == 0 and i % 5 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)
```

**Camoscio**:

```python
for i in range(1, 101):
    if i % 3 == 0 and i % 5 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)
```

## 🖊️ Citations

If you use camoscio or the camoscio dataset in your research, please cite our paper:

```bibtex
@misc{santilli2023camoscio,
      title={Camoscio: an Italian Instruction-tuned LLaMA}, 
      author={Andrea Santilli and Emanuele Rodolà},
      year={2023},
      eprint={2307.16456},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

You should also cite the original LLaMA paper [1], the Self-Instruct paper [2], the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) repo, and the [Alpaca-LoRA](https://github.com/tloen/alpaca-lora) repo.








