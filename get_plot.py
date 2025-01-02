import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_dataframes(df_with_names, col, output_filename, best_n, starting_epoch):
    plt.figure(figsize=(14, 10))  

    for name, df,_ in df_with_names[:best_n]:
        if '128' in name:
            line_style = '-'
        elif '256' in name:
            line_style = '--'
        else:
            line_style = '-.'
      
        plt.plot(df[col][starting_epoch:], label=name, linestyle = line_style)  

    plt.title(f'{col} vs. Epoch')  
    plt.xlabel('Epoch') 
    plt.ylabel('value')  
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')  
    plt.tight_layout()
    plt.grid()  
    plt.savefig(output_filename)  
    plt.close()  

dir = '/scratch/zc1592/small_data/experiments/focal'
loss_path = glob.glob(os.path.join(dir, '**', 'losses.csv'))
loss_df = [pd.read_csv(i) for i in loss_path]
names = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
best_n = 10
eval_df_with_mins = [
    (name, df, df['eval_loss'].min()) for name, df in zip(names, loss_df)
]
eval_df_with_mins_sorted = sorted(eval_df_with_mins, key=lambda x: x[2])


for name, _, min_value in eval_df_with_mins_sorted[:best_n]:
    print(f"Name: {name}, Min Value: {min_value}")

plot_dataframes(eval_df_with_mins_sorted, 'eval_loss', os.path.join(dir, 'eval_loss.png'),best_n,1)