import math
import os
import io
import sys
import time
import json



import tqdm

import copy
import json
import os 
import logging
logger = logging.getLogger(__name__)




def read_json_file(filename):
    with open(filename, 'r') as fp:
        data = fp.readlines()
    if len(data) == 1:
        data = json.loads(data[0])
    else:
#         print("here")
#         print(data)
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
    
def read_to_list(filename):
    f = open(filename, 'r',encoding="utf-8")
    res = {}
    for row in f:
        (rid, text) = row.split('\t')
        res[int(rid)] = text.strip()
    #         res.append(row.lower().split())
    return res

def time_format(time_cost):
    m, s = divmod(time_cost, 60)
    h, m = divmod(m, 60)
    # print("time_cost: %d" % (time_cost))
    return "%02d:%02d:%02d" % (h, m, s)
