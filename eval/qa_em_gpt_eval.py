import openai
import json
from tqdm import tqdm
from datasets import load_dataset
import os
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

# read eviroment variable
openai.api_key = os.environ.get('OPENAI_API_KEY')

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def openai_call(context, question, correct_answer, answers, index):
    prompt = f"Given the context below and the corresponding question, please indicate for each answer provided (A) by a bunch of models whether it is correct (1) or not (0). Use a dict format in the response e.g., model: 1.\n"
    prompt += f"Context: {context}\n"
    prompt += f"Question: {question}\n"
    prompt += f"Correct gold answer: {correct_answer}\n\n"
    for model, answer in answers.items():
        prompt += f"A {model}: {answer[index]}"
    #print(prompt)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        max_tokens=1024,
        temperature=0,
        )
    return response.choices[0]["message"]["content"].strip()

squad_it = load_dataset("squad_it")



#TODO insert in the folder results_qa a file for each model with the results (one response per line, see camoscio-7b-question-answering_test.txt for an example)
files = os.listdir("results_qa")
res_dict = {}
for file in files:
    with open(f"results_qa/{file}") as f:
        lines = f.readlines()
        res_dict[file.split("question-answering")[0][:-1]] = lines



for index, elem in tqdm(enumerate(squad_it['test']), total=len(squad_it['test'])):
    response = openai_call(elem['context'], elem['question'], elem['answers']['text'][0], res_dict, index)
    with open('openai_results.json', 'a') as f:
        try:
            resp_dict = json.loads(response)
            resp_dict['index'] = index
            json.dump(resp_dict, f)
            f.write('\n')
        except:
            print(response)