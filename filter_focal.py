import pandas as pd
import glob
import os
import re

dir = '/scratch/zc1592/small_data/experiments/focal_v1'
eval_paths = glob.glob(os.path.join(dir, '**', 'losses.csv'), recursive = True)

results = []
for path in eval_paths:
    df = pd.read_csv(path)
    min_eval_loss = df['eval_loss'].min()
    min_index = df['eval_loss'].idxmin() 
    eval_dir = os.path.dirname(path)
    with open(os.path.join(eval_dir, 'eval_results.txt'), 'r') as file:
        content = file.read()
    pattern = r"average non-zero elemnts in eval ([\d.]+)"
    matches = re.findall(pattern, content)
    non_zero_elements = [float(match) for match in matches]
    name =  os.path.basename(eval_dir)

    results.append({
        'name': name,
        'min_eval_loss': min_eval_loss,
        'index': min_index,
        'avg_non_zero_elements': non_zero_elements[min_index]
    })
result_df = pd.DataFrame(results)
result_df = result_df.sort_values(by='index', ascending=False)
print(result_df.head())