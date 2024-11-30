
import json
import os

def read_jsonl(path):
    datas = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            datas.append(json.loads(line))
    return datas 

def save_jsonl(path, datas, mode='w'):
    dir = os.path.dirname(path)
    os.makedirs(dir, exist_ok=True)
    
    with open(path, mode=mode, encoding='utf-8') as f:
        f.write(''.join([json.dumps(data) for data in datas]))
    
