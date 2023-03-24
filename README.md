# 🇮🇹🦙🤏  Camoscio: An Italian instruction-tuned LLaMA

This repository contains code for reproducing the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) in Italian using [low-rank adaptation (LoRA)](https://arxiv.org/pdf/2106.09685.pdf).
The repo is a fork of [cabrita](https://github.com/22-hours/cabrita) and [Alpaca-LoRA](https://github.com/tloen/alpaca-lora).
Following previous approaches, we translated the Stanford Alpaca instruction-tuning dataset into Italian using the ChatGPT API.
We provide the translated dataset (`camoscio_data.json`), the model (available [here]() on the Hugging Face's hub) and the code to reproduce the results.
The model should provide output of similar quality to `text-davinci-003` (to be evaluated) that can run [on a Raspberry Pi](https://twitter.com/miolini/status/1634982361757790209) (for research).

To finetune the model on the Italian dataset we adapted the scripts from [cabrita](https://github.com/22-hours/cabrita) and run the training on a single 3090 for 1 day (see details below).

Please note that it is highly possible that the model output contains biased, conspiracist, offensive, or otherwise inappropriate and potentially harmful content.
The model is intended for research purposes only and should be used with caution at your own risk. Production usage is not allowed.


[1]: LLaMA: Open and Efficient Foundation Language Models. Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample. https://arxiv.org/abs/2302.13971v1

[2]: Self-Instruct: Aligning Language Model with Self Generated Instructions. Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi. https://arxiv.org/abs/2212.10560

## 🖥️ Demo - Talk with Camoscio [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/teelinsan/camoscio/blob/master/notebooks/camoscio-lora.ipynb) 

Clic the button "Open in Colab" above to open the notebook in Google Colab and try it out!

## 📚 How to use
We provide an example notebook on how to load and use the model [here](notebooks/camoscio-lora.ipynb).

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

## 🏋️ Reproduce the training 
### Setup

1. Install dependencies

```
pip install -r requirements.txt
```

2. If bitsandbytes doesn't work, [install it from source.](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md) Windows users can follow [these instructions](https://github.com/tloen/alpaca-lora/issues/17).

### Translate the Dataset to Italian (`translate_data.py`)
Download the dataset from [here]((https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)) and put it in the `data` folder. Then run:

```
python script/translate_data.py
```

### Train the model (`train.py`)
Just lunch the command (change hyperparameters as needed):

```
python notebooks/train.py
```


### Checkpoint export
You can use the script `export_hf_checkpoint.py` from the original [Alpaca-LoRa repo](https://github.com/tloen/alpaca-lora/blob/main/export_hf_checkpoint.py) to export the checkpoint to the HuggingFace format or use the script [export_state_dict_checkpoint.py](https://github.com/tloen/alpaca-lora/blob/main/export_state_dict_checkpoint.py) to export the checkpoint to the PyTorch format.

These files contain scripts that merge the LoRA weights back into the base model
for export to Hugging Face format and to PyTorch `state_dicts`.
They should help users
who want to run inference in projects like [llama.cpp](https://github.com/ggerganov/llama.cpp)
or [alpaca.cpp](https://github.com/antimatter15/alpaca.cpp).

## 🔍 Evaluation

Coming soon.

## 📝 Example outputs

**Instruction**: Riassumi la storia di Pinocchio.

**Alpaca-LoRA**: Once upon a time, there was a wooden puppet named Pinocchio who dreamed of becoming a real boy. He was created by a woodcarver named Geppetto, who loved him very much. Pinocchio was a mischievous puppet who often got into trouble, but he always tried to do the right thing in the end. One day, Pinocchio met a fairy who offered him the chance to become a real boy if he proved himself to be brave, truthful, and unselfish. Pinocchio accepted the fairy's offer and set off on a series of advent

**ChatGPT**: La storia di Pinocchio racconta di un burattino di legno, creato dal falegname Geppetto, che prende vita grazie alla fata turchina. Pinocchio ha un forte desiderio di diventare un bambino vero, ma cade in numerose avventure e inganni, spesso guidato dal suo istinto e dalla sua ingenuità. Nel corso della storia, Pinocchio viene ingannato da un gatto e una volpe, viene imprigionato e trasformato in un asino, ma alla fine dimostra il suo coraggio, la sua lealtà e la sua generosità salvando il padre e ottenendo la sua trasformazione in un bambino vero.

**Camoscio**: 

---

**Instruction**: Dimmi di più sul presidente della repubblica Italiana del 2019.

**Alpaca-LoRA**: The President of the Republic of Italy in 2019 is Sergio Mattarella. He was elected in 2015 and re-elected in 2018. He is the 12th President of the Republic of Italy.

**ChatGPT**: Il presidente della Repubblica Italiana del 2019 era Sergio Mattarella. Nato a Palermo nel 1941, Mattarella è stato eletto presidente nel 2015, succedendo a Giorgio Napolitano. Prima di diventare presidente, ha ricoperto importanti incarichi politici, tra cui quello di ministro della Difesa, della Pubblica Istruzione e della Giustizia. 
Durante il suo mandato, Mattarella è stato impegnato nella promozione dell'unità nazionale e nella difesa della Costituzione italiana. Ha anche sostenuto l'integrazione europea e la cooperazione internazionale. Nel 2018, ha svolto un ruolo importante nella formazione del governo italiano, che ha visto la nomina di un governo di coalizione tra il Movimento 5 Stelle e la Lega. 
Mattarella è stato molto attivo nella difesa dei diritti umani e della giustizia sociale, promuovendo iniziative per combattere la povertà e la disuguaglianza. 

**Camoscio**: 

---

**Instruction**: Dimmi quattro parole che fanno rima con 'gatto'.

**Alpaca-LoRA**: Cat, hat, mat, rat.

**ChatGPT**: Piatto, matto, spratto, grattò.

**Camoscio**: 


---

**Instruction**: Scrivi una breve biografia su Dante Alighieri

**Alpaca-LoRA**: Dante Alighieri (1265-1321) was an Italian poet, philosopher, and political figure. He is best known for his epic poem, The Divine Comedy, which is widely considered to be one of the greatest works of literature ever written. Dante was born in Florence, Italy, and spent much of his life in exile. He wrote The Divine Comedy while in exile, and it tells the story of his journey through Hell, Purgatory, and Heaven. Dante is considered to be one of the founders of the Italian language, and

**ChatGPT**: Dante Alighieri è stato uno dei più grandi poeti italiani del Medioevo. Nato a Firenze nel 1265, ha studiato nella prestigiosa Scuola di Santa Croce e ha mostrato fin da giovane una grande passione per la poesia. Ha scritto la sua prima opera, "Il Convivio", all'età di 30 anni, ma è soprattutto noto per la sua opera più famosa, "La Divina Commedia". Questo poema epico, scritto in terzine, descrive il viaggio immaginario di Dante attraverso l'Inferno, il Purgatorio e il Paradiso, e rappresenta una sintesi della cultura medievale, della filosofia e della teologia. Dante ha anche avuto una vita politica attiva, partecipando alla vita pubblica di Firenze e sostenendo la fazione guelfa. Tuttavia, dopo essere stato esiliato dalla città nel 1302, ha trascorso il resto della sua vita in esilio, scrivendo le sue opere più importanti e cercando di rientr

**Camoscio**: 


---

**Instruction**: Dimmi tutte le regioni italiane in ordine alfabetico.

**Alpaca-LoRA**: Abruzzo, Basilicata, Calabria, Campania, Emilia-Romagna, Friuli-Venezia Giulia, Lazio, Liguria, Lombardia, Marche, Molise, Piemonte, Puglia, Sardegna, Sicilia, Toscana, Trentino-Alto Adige, Umbria, Veneto.

**ChatGPT**: Abruzzo, Basilicata, Calabria, Campania, Emilia-Romagna, Friuli-Venezia Giulia, Lazio, Liguria, Lombardia, Marche, Molise, Piemonte, Puglia, Sardegna, Sicilia, Toscana, Trentino-Alto Adige/Südtirol, Umbria, Valle d'Aosta/Vallée d'Aoste, Veneto.

**Camoscio**: Abruzzo, Basilicata, Calabria, Campania, Emilia-Romagna, Friuli-Venezia Giulia, Lazio, Liguria, Lombardia, Marche, Molise, Piemonte, Puglia, Sardegna, Sicilia, Toscana, Trentino-Alto Adige, Umbria, Veneto.

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

---







