from collections import defaultdict
from concurrent.futures import as_completed, ProcessPoolExecutor

import json
import logging
import argparse
import numpy as np
from typing import List, Union
import itertools

import tqdm
import ipdb
from execution import check_correctness


logging.basicConfig(
    format="SystemLog: [%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def pass_at_K(logger, results, k = [1, 10, 50, 100]):
    def _turn_list_into_dict(result_lines):
        result_dict = defaultdict(list)
        for line in result_lines:
            result_dict[line['metadata']['task_id']].append(line['passed'])
        logger.info(f'{len(result_dict)} tasks to be evaluated')
        return result_dict

    # Calculate pass@k.
    total, correct = [], []
    task_id_and_pass = dict()
    result_by_task = _turn_list_into_dict(results)
    for task_id in result_by_task.keys():
        passed = result_by_task[task_id]
        total.append(len(passed))
        correct.append(sum(passed))
        task_id_and_pass[task_id] = passed

    total = np.array(total)
    correct = np.array(correct)

    ks = k
    pass_at_k = {f"pass@{k}": round(_estimate_pass_at_k(total, correct, k).mean(), 6)
                 for k in ks if (total >= k).all()}
    logger.info(pass_at_k)

def _estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()
    data = [json.loads(line) for line in data]
    return data

def truncate_response_zero_shot(prompt, content):
    # locate prompt in the content and remove them
    prompt_lines = prompt.split('\n')
    content_lines = content.split('\n')
    # check if the last line of prompt is in the content
    try:
        # the second last is """
        assert prompt_lines[-2] == content_lines[len(prompt_lines)-2]
    except:
        ipdb.set_trace()
    content = '\n'.join(content_lines[len(prompt_lines)-1:])
    for identifier in ['\nclass', '\ndef', '\n#', '\nif', '\nprint']:
        if identifier in content:
            content = content.split(identifier)[0]
    return content

def truncate_response(entry_point, content):
    content = content.split("### Response")[1]
    flag_str = f"def {entry_point}("
    located_lines = []
    flag = False
    for line in content.split('\n'):
        if line.startswith(flag_str):
            flag = True
            continue
        if flag:
            located_lines.append(line)
    content = '\n'.join(located_lines)
    not_with_correct_ending = False
    for identifier in ['\nclass', '\ndef', '\n#', '\nif', '\nprint']:
        if identifier in content:
            not_with_correct_ending = True
            content = content.split(identifier)[0]
    content = content.split('</s>')[0]
    additional_desc = True if content.count('"""') > 1 else False
    return content, not_with_correct_ending, additional_desc


def evaluate_with_test_code(
    samples,
    timeout
):
    logger.info(f'Start evaluation with test code, timeout={timeout}')
    # Check the generated samples against test suites.
    with ProcessPoolExecutor(max_workers=10) as executor:

        futures = []
        existed_completion = defaultdict(set)
        results = defaultdict(defaultdict)

        print(len(samples))
        for sample in samples:
            task_id = sample['metadata']["task_id"]
            prompt = sample['metadata']['prompt']
            test = sample['metadata']['test']
            entry_point = sample['metadata']['entry_point']
            completion = sample["trunc_response"]
            if completion in existed_completion[task_id]:
                continue
            existed_completion[task_id].add(completion)
            args = (task_id, prompt, completion, test, entry_point, timeout)
            future = executor.submit(check_correctness, *args)
            futures.append(future)
        logger.info(f'{len(futures)} execution requests are submitted')
        
        for idx, future in tqdm.tqdm(enumerate(as_completed(futures)), total=len(futures)):
            # logger.info('[{}/{}] execution completed'.format(idx+1, len(futures)))
            result = future.result()
            results[result["task_id"]][result["completion"]] = result

    logger.info('execution finished! start parsing results')
    samples_with_result = []
    for sample in samples:
        task_id = sample['metadata']["task_id"]
        completion = sample["trunc_response"]
        result = results[task_id][completion]
        sample["result"] = result["result"]
        sample["passed"] = result["passed"]
        samples_with_result.append(sample)

    assert len(samples_with_result) == len(samples), "Some problems are not attempted."

    return samples_with_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=2, help="how many seconds to wait during execution for each test case")
    parser.add_argument("--preds_filename", type=str, default='results.jsonl', help="A list of extracted solution samples")
    args = parser.parse_args()

    predictions = load_jsonl(args.preds_filename)
    
    for prediction in predictions:
        truncated, not_correct_ending, additional_desc = truncate_response(prediction['metadata']["entry_point"], prediction['response'])        
        prediction['trunc_response'] = truncated
    exec_result = evaluate_with_test_code(predictions, timeout=args.timeout)
    pass_at_K(logger, exec_result)