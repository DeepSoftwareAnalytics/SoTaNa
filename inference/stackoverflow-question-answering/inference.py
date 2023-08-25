import torch
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig

import os
import sys
import argparse
import logging
import gradio as gr
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt = '%m/%d/%Y %H:%M:%S',level = logging.INFO)
logger = logging.getLogger(__name__)

import time
import sys
sys.path.append("../")
sys.path.append("../utils")
sys.path.append("../metric")
from evaluation import metetor_rouge_cider, Sentence_BLUE_SM2,bleus

from utils import save_json_data,read_json_file,time_format
def generate_prompt(data_point, input=None):
     return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Answer the question on stackoverflow.

### Input:
Question title: {data_point['question_title']}

Question body: {data_point['question_body'][:650]}

### Response:
"""

def evaluate(instruction, input=None, **kwargs):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        **kwargs,
    )
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=256,
    )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return prompt, output.split("### Response:")[1].strip().replace("</s>","")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only_use_pre_train', action='store_true', help='only_use_pre_train', required=False)
    # parser.add_argument('--do_web_ui', action='store_true', help='using gradio', required=False)
    parser.add_argument("--input_dir", default="dataset", type=str, required=False,
                        help="")
    parser.add_argument("--input_filename", default="dataset.jsonl", type=str, required=False,
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
        # device_map={'': 0},
        device_map="auto",
    )

    if not args.only_use_pre_train:
        logger.info("loading lora from %s"%args.lora_model_path)
        # model = PeftModel.from_pretrained(
        #     model, "tloen/alpaca-lora-7b", torch_dtype=torch.float16,device_map="auto",
        # )
        model = PeftModel.from_pretrained(
            model, args.lora_model_path, torch_dtype=torch.float16,device_map="auto",
        )
        logger.info("loaded lora")
    model.eval()


    instructions =  read_json_file(os.path.join(args.input_dir, args.input_filename))[:args.data_num]

    results = []
    refs=[]
    preds = []
    start = time.perf_counter()
    for i, instruction in enumerate(instructions) :
        prompt, response = evaluate(instruction)
        # response = response.split("\n")[0]
        if len(response) < 1:
            response = "null"
        sample = {"instruction":prompt,
        "response":response 
        }
        results.append(sample)
        if i < 3:
            logger.info("Instruction: %s"%prompt)
            logger.info("Response: %s "%response)
            logger.info(80*"*")
        refs.append(instruction[ 'answer_body'].lower().split())
        preds.append(response.lower().split())
        torch.cuda.empty_cache()
        # preds.append(response.lower().split())
    
    output_dir = args.output_dir
    time_cost = time_format(time.perf_counter() - start)
    save_json_data(output_dir, "time_cost.jsonl", [time_cost])
    logger.info(" time cost :" + time_cost)
   
    # output_dir = sys.argv[1]
    filename= "results.jsonl"
    save_json_data(output_dir, filename, results)

    with open(output_dir+"/test.output",'w') as f, open(output_dir+"/test.gold",'w') as f1:

        for pred, instruction in zip(preds,instructions):
            # if 'choices' in result.keys():
            fid = instruction['question_id']

            infer = " ".join(pred)
            gold = instruction[ 'answer_body']

            f.write(str(fid) +'\t'+infer+'\n')
            f1.write(str(fid) +'\t'+gold+'\n')   

    args.evalutate_with_metrics =True
    if args.evalutate_with_metrics:
        refs = [[t] for t in refs]
        preds = [t for t in preds]
        logger.info("A ref is: %s"% (" ".join(refs[0][0])) )
        logger.info("A preds is: %s"% (" ".join(preds[0])) )
        # exit(1)
        # scores = Sentence_BLUE_SM2(refs, preds)
        scores = bleus(refs, preds)
        # logger.info("BLEU: %.2f"%bleus_score)
        results = metetor_rouge_cider(refs, preds)
        scores = {** scores, **results}
        logger.info(scores)
        filename= "scores.jsonl"
        save_json_data(output_dir, filename, scores)
