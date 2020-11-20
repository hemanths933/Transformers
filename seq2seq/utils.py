import json
import numpy as np
import time
from sklearn.metrics import f1_score, accuracy_score
from torchtext.data.metrics import bleu_score

def get_config():
    with open('config.json', 'r') as f:
        config = json.load(f)
    return config

def log_metrics(writer, epoch, predicted, labels, variable_name, config, loss=None, num_samples=None):
    
    '''calculate blue'''
    
    blue = bleu_score(predicted, labels)
    #print(f"blue score is {blue}")
    writer.add_scalar(f'Blue/{variable_name}', blue, epoch)

    '''
    log loss
    '''
    if loss != None:
        average_loss = loss/ num_samples
        print(f"{variable_name} loss is {average_loss}")
        writer.add_scalar(f'Loss/{variable_name}', average_loss, epoch)

        return blue, average_loss
    return blue, None

def get_timestamp():
    local_time = time.localtime()
    day = local_time.tm_mday
    month = local_time.tm_mon
    year = local_time.tm_year
    hour = local_time.tm_hour
    minutes = local_time.tm_min
    sec = local_time.tm_sec
    timestamp = str(day) + '_' + str(month) + '_' + str(year) + '_' + str(hour) + '_' + str(minutes) + '_' + str(sec)
    
    return timestamp
