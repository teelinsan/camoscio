import openai
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


# read eviroment variable
openai.api_key = os.environ.get('OPENAI_API_KEY')

# read all the json files in the folder_path and merge them into one json file
def merge_json_files(folder_path):
    """
    Merge all the json files in the folder_path into one json file
    """
    json_files = [pos_json for pos_json in os.listdir(folder_path) if pos_json.endswith('.json')]
    json_data = []
    for file in json_files:
        with open(os.path.join(folder_path, file)) as f:
            json_data.extend(json.load(f))
    return json_data

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def translate_text(value):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Translate the following text to Italian: '{value}'"},
            ],
        max_tokens=1024,
        temperature=0,
        )
    return response.choices[0]["message"]["content"].strip()

def translate_item(item):
    translated_item = {}
    for key, value in item.items():
        if value:
            translated_value = translate_text(value)
            translated_item[key] = translated_value
        else:
            translated_item[key] = ''
    return translated_item


if __name__ == '__main__':


    DATASET_TO_TRANSLATE = '../data/alpaca_data.json'
    OUTPUT_FILE = '../data/camoscio.json'
    OUTPUT_FOLDER = '../data/translated_data'

    MAX_PARALLEL_REQUESTS = 50  # Maximum number of parallel requests to make to the API

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # Assuming the input JSON is in a file named 'input.json'
    with open(DATASET_TO_TRANSLATE, 'r') as f:
        data = json.load(f)

    CHUNK_SIZE = 1000
    start = 0
    end = len(data)
    # Translate the data in chunks of 1000 items
    for i in range(start, end, CHUNK_SIZE):
        start = i
        end = i + CHUNK_SIZE

        translated_data = []
        data_new = data[start:end]

        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_REQUESTS) as executor:
            futures = {executor.submit(translate_item, item): item for item in data_new}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Translating"):
                translated_data.append(future.result())


        # Save the translated data to a new JSON file named 'translated_data.json'
        with open(f'{OUTPUT_FOLDER}/from_{start}_to_{end}.json', 'w') as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=4)

        print(f"Translation complete. The translated data is saved in 'translated_data_from_{start}_to_{end}.json'")


    print("Merging chunks now...")


    merged_data = merge_json_files(OUTPUT_FOLDER)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)

    print("Merging complete. The merged data is saved in 'camoscio_data.json'")
