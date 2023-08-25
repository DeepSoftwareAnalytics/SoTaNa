import os
import argparse
import logging
import time
import json

import torch
from peft import PeftModel

from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig

from evaluation import truncate_response, evaluate_with_test_code, pass_at_K

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt = '%m/%d/%Y %H:%M:%S',level = logging.INFO)
logger = logging.getLogger(__name__)

def generate_prompt(prompt):
    # Generate the fucntion body according to the function signature and problem description. Do not repeat the given problem description in the response.
    return f"""\
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Continue to write a function in python.

### Input:
{prompt}

### Response:
"""


def read_json_file(filename):
    with open(filename, 'r') as fp:
        data = fp.readlines()
    if len(data) == 1:
        data = json.loads(data[0])
    else:
        data = [json.loads(line) for line in data]
    return data

def save_json_data(data_dir, filename, data):
    os.makedirs(data_dir, exist_ok=True)
    file_name = os.path.join(data_dir, filename)
    with open(file_name, 'w') as output:
        if type(data) == list:
            if type(data[0]) in [str, list,dict]:
                for item in data:
                    output.write(json.dumps(item))
                    output.write('\n')

            else:
                json.dump(data, output)
        elif type(data) == dict:
            json.dump(data, output)
        else:
            raise RuntimeError('Unsupported type: %s' % type(data))
    logger.info("saved dataset in " + file_name)


def time_format(time_cost):
    m, s = divmod(time_cost, 60)
    h, m = divmod(m, 60)
    # print("time_cost: %d" % (time_cost))
    return "%02d:%02d:%02d" % (h, m, s)

def evaluate(instruction, **kwargs):
    prompt = generate_prompt(instruction['prompt'])
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    generation_config = GenerationConfig(
        temperature=0.1,
        do_sample=True,
        top_p=0.95,
        top_k=40,
        **kwargs,
    )
    logger.info(f"{generation_config}")
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=512,
    )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return prompt, output

def evaluate_passk(predictions):
    not_correct_ending_total = 0
    with_additional_description = 0
    for prediction in predictions:
        truncated, not_correct_ending, addtional_desc = truncate_response(prediction['metadata']["entry_point"], prediction['response'])        
        prediction['trunc_response'] = truncated
        if not_correct_ending:
            not_correct_ending_total += 1
        if addtional_desc:
            with_additional_description += 1
    exec_result = evaluate_with_test_code(predictions, timeout=2)
    pass_at_K(logger, exec_result)
    # print(f'not correct ending: {not_correct_ending_total/len(predictions)}')
    # print(f'with additional description: {with_additional_description/len(predictions)}')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only_use_pre_train', action='store_true', help='only_use_pre_train', required=False)
    # parser.add_argument('--do_web_ui', action='store_true', help='using gradio', required=False)
    parser.add_argument("--input_dir", default="dataset/", type=str, required=False,
                        help="")
    parser.add_argument("--input_filename", default="human-eval-v2-20210705.jsonl", type=str, required=False,
                        help="")
    parser.add_argument("--output_dir", default="saved_models/pre-train", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--lora_model_path", default="/home/t-enshengshi/msracir001/t-enshengshi/LLM/alpaca-lora-plus/output/finetune/7B/20230317122659/lora-alpaca/", type=str, required=False,
                        help="The filename of lora model")
    parser.add_argument("--llama_model", default="decapoda-research/llama-7b-hf", type=str, required=False,
                        help="The filename of lora model")
    parser.add_argument("--data_num", default=10, type=int, required=False,
                        help="the number of data")
    args = parser.parse_args()
    return  args

if __name__ == "__main__":
    # testing code for readme
    args = parse_args()
    logger.info("Args is %s", args)
    logger.info("tokenizer")
    tokenizer = LlamaTokenizer.from_pretrained(args.llama_model)

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    logger.info("loading model")
    model = LlamaForCausalLM.from_pretrained(
        args.llama_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map={'': 0},
        # device_map="auto",
    )
    # model = AutoModelForCausalLM.from_pretrained(args.llama_model, low_cpu_mem_usage=True, load_in_8bit=True).cuda()

    if not args.only_use_pre_train:
        logger.info("loading lora from %s"%args.lora_model_path)
        # model = PeftModel.from_pretrained(
        #     model, "tloen/alpaca-lora-7b", torch_dtype=torch.float16,device_map="auto",
        # )
        model = PeftModel.from_pretrained(
            model, args.lora_model_path, torch_dtype=torch.float16, device_map={'': 0},
        )
        logger.info("loaded lora")
    model.eval()


    instructions =  read_json_file(os.path.join(args.input_dir, args.input_filename))[:args.data_num]

    results = []
    start = time.perf_counter()
    for i, instruction in enumerate(instructions):
        prompt, response = evaluate(instruction)
        sample = {
            "instruction":prompt,
            "response":response,
            "metadata": instruction
        }
        results.append(sample)
        logger.info(f'[{i+1}/{len(instructions)}] completed')
        logger.info("Response: %s "%response)
        logger.info(80*"*")

        if i % 10 == 0:
            print('evaluating....')
            evaluate_passk(results)
    
    output_dir = args.output_dir
    time_cost = time_format(time.perf_counter() - start)
    save_json_data(output_dir, "time_cost.jsonl", [time_cost])
    logger.info(" time cost :" + time_cost)
    print('evaluating....')
    evaluate_passk(results)

    # output_dir = sys.argv[1]
    filename= "results.jsonl"
    save_json_data(output_dir, filename, results)
