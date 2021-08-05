# This file was modified as part of this research project.

import os
import json
import pandas as pd
from glob import glob

evaluation_base_path = 'C:/Users/Niall/College/Year_5/Dissertation/Results/disp_maps/'
path_extensions = [('AiF/evaluation/**/*.json', 'AiF'),
                   ('refocused_0.3/evaluation/**/*.json', 'refocused_0.3'),
                   ('refocused_0.7/evaluation/**/*.json', 'refocused_0.7')]

# The metric wanted. Possible values: badpix_0010, badpix_0030, badpix_0070, mse_100, q_25_100
metric = "badpix_0070"

for path, test in path_extensions:
    scores = {}
    json_files = glob(evaluation_base_path + path)
    for file in json_files:
        algorithm_name = file.split('\\')[-2]
        with open(file, 'r') as f:
            data = json.load(f)
            for scene in data:
                if algorithm_name in scores:
                    scores[algorithm_name][scene] = round(data[scene]['scores'][metric]['value'], 2)
                else:
                    scores[algorithm_name] = {}

    df = pd.DataFrame.from_dict(scores)
    df.to_csv('{}_{}_results.csv'.format(metric, test))
