import time
import json
import os
import random
import re
import string
from functools import partial
from multiprocessing import Pool
import requests

import numpy as np
import tqdm
import utils
from utils import save_json_data,read_json_file,time_format
import fire
import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt = '%m/%d/%Y %H:%M:%S',level = logging.INFO)
logger = logging.getLogger(__name__)
def encode_prompt(prompt_instructions):
    """Encode multiple prompt instructions into a single string."""
    prompt = open("./code-prompt.txt").read() + "\n"

    for idx, task_dict in enumerate(prompt_instructions):
        try:
            (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        except TypeError:
            print(task_dict)
            print("")
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<noinput>" if input.lower() == "" else input
        prompt += f"###\n"
        prompt += f"{idx + 1}. Instruction: {instruction}\n"
        prompt += f"{idx + 1}. Input:\n{input}\n"
        prompt += f"{idx + 1}. Output:\n{output}\n"
    prompt += f"###\n"
    prompt += f"{idx + 2}. Instruction:"
    return prompt


def post_process_gpt3_response(num_prompt_instructions, response):
    if ('choices' not in response.keys()) or (response['choices'] is None) or (response['choices'][0] is None) :
        return []
    raw_instructions = f"{num_prompt_instructions+1}. Instruction:" + response['choices'][0]["text"]
    raw_instructions = re.split("###", raw_instructions)
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        if idx == len(raw_instructions) - 1 and response['choices'][0]["finish_reason"] == "length":
            continue
        idx += num_prompt_instructions + 1
        splitted_data = re.split(f"{idx}\.\s+(Instruction|Input|Output):", inst)
        if len(splitted_data) != 7:
            continue
        else:
            inst = splitted_data[2].strip()
            input = splitted_data[4].strip()
            input = "" if input.lower() == "<noinput>" else input
            output = splitted_data[6].strip()
        # filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue
       
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        if not inst[0].isascii():
            continue
        instructions.append({"instruction": inst, "input": input, "output": output})
    return instructions


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def generate_instruction_following_data(
    output_dir="./output/",
    seed_tasks_path="./seed_tasks.jsonl",
    num_instructions_to_generate=202,
    model_name="text-davinci-003",
    num_prompt_instructions=3,
    request_batch_size=1,
    temperature=1.0,
    top_p=1.0,
    num_cpus=16,
):
    logger.info("output_dir: %s"% output_dir)
    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
    seed_instruction_data = [
        {"instruction": t["instruction"], "input": t["instances"][0]["input"], "output": t["instances"][0]["output"]}
        for t in seed_tasks
    ]
    print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")

    # load the 200 seed task instructions
    
    if os.path.exists("se_seed_tasks.json"):
        # instruction_data = [json.loads(l) for l in open("se_seed_tasks.json", "r")]
        instruction_data = utils.jload("se_seed_tasks.json")
        seed_instruction_data += instruction_data
        print(f"Loaded {len(instruction_data)} se-related instructions")
        


    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    machine_instruction_data = []
    if os.path.exists(os.path.join(output_dir, "regen.json")):
        machine_instruction_data = utils.jload(os.path.join(output_dir, "regen.json"))
        print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")

    # similarities = {}
    # scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    if machine_instruction_data:
        progress_bar.update(len(machine_instruction_data))

    # first we tokenize all the seed instructions and generated machine instructions
    all_instructions = [d["instruction"] for d in seed_instruction_data] + [
        d["instruction"] for d in machine_instruction_data
    ]
    # all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions] #
    all_request_start = time.perf_counter() 
    while len(machine_instruction_data) < num_instructions_to_generate:
        if len(machine_instruction_data) > 0 and len(machine_instruction_data) % 1000 == 0:
            time.sleep(600)


        request_idx += 1

        batch_inputs = []
        for _ in range(request_batch_size):
            prompt_instructions = random.sample(seed_instruction_data, num_prompt_instructions)
            if len (machine_instruction_data) > 1:
                prompt_machine_sample = random.sample(machine_instruction_data, 2)
                prompt_instructions = prompt_instructions + prompt_machine_sample 
            prompt = encode_prompt(prompt_instructions)
            batch_inputs.append(prompt)
        request_start = time.time()
        headers = {
                'content-type': 'application/json',
                'api-key': 'xxxxx'}
        url = "https://augloop-cs-test-scus-shared-open-ai-0.openai.azure.com/openai/deployments/text-davinci-003"
        results = []
        max_times=3
        for prompt in batch_inputs:
            data = {"prompt": prompt , "max_tokens": 3000}
            for i in range(max_times):
                # response= requests.post(url, headers = headers, data = data)
                try:
                    response = requests.post(url, headers = headers, json = data)
                except:
                    continue
                if response.status_code != 200:
                    print ("Warning: request fails for %d times" % (i, ))
                    continue
                break
            results.append(json.loads(str(response.content, encoding = "utf-8") )  )
        request_duration = time.time() - request_start

        process_start = time.time()
        instruction_data = []
        for result in results:
            new_instructions = post_process_gpt3_response(num_prompt_instructions, result)
            instruction_data += new_instructions

        total = len(instruction_data)
        
            
        if total > 0:
            machine_instruction_data +=instruction_data
            progress_bar.update(total)
            if len(machine_instruction_data) > 0 and len(machine_instruction_data) % 1000 == 0:
                utils.jdump(machine_instruction_data, os.path.join(output_dir, "tmp","regen.json"))
    utils.jdump(machine_instruction_data, os.path.join(output_dir, "regen.json"))
    time_cost = time_format(time.perf_counter() - all_request_start)
    save_json_data(output_dir, "time_cost.jsonl", [time_cost])
    logger.info(" time cost :" + time_cost)

def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
#     generate_instruction_following_data()