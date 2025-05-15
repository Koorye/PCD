import numpy as np
import os
import pandas as pd
import re


PCD_COLOR = 'blue!10'
ROUND = 2


def parse_value(value):
    # parse value from string to list/int/float/bool
    if isinstance(value, str) and ',' in value:
        return [parse_value(v) for v in value.split(',')]
    if re.match(r'^-?\d+$', value):
        return int(value)
    if re.match(r'^-?\d+\.\d+$', value):
        return float(value)
    if value == 'True' or value == 'False':
        return value == 'True'
    return value


_ROOTS = ['results/default']
# _ROOTS = ['results/motivation']
# _ROOTS = ['results/light_variant']
_TASKS = [
    'google_robot_close_drawer', 
    'google_robot_move_near', 
    'google_robot_open_drawer', 
    'google_robot_pick_coke_can', 
    'google_robot_place_apple_in_closed_top_drawer', 
    'widowx_carrot_on_plate', 
    'widowx_put_eggplant_in_basket',
    'widowx_spoon_on_towel',
    'widowx_stack_cube',
]
# _TASKS = [
#     'google_robot_pick_coke_can',
#     'google_robot_pick_coke_can_dark',
#     'google_robot_pick_coke_can_drawer_variant',
#     'google_robot_pick_coke_can_light_variant',
#     'google_robot_pick_coke_can_table_paper_variant',
#     'google_robot_pick_coke_can_table_stone_variant',
# ]
_MODES = ['baseline', 'contrast--by=box_tracking', 'contrast--by=point_tracking', 'contrast--by=grounded_sam_tracking']

def is_intersection(l1, l2):
    return bool(set(l1) & set(l2))

def mean_func(x, prefix=''):
    # if number, mean
    # if str, last
    if len(x) == 0:
        return None
    
    for each in x:
        if isinstance(each, str):
            if prefix != '':
                return prefix + '_' + 'average'
            return 'average'
    return x.mean()

def delta_func(x, prefix=''):
    if len(x) == 0:
        return None
    
    for each in x:
        if isinstance(each, str):
            if prefix != '':
                return prefix + '_' + 'delta'
            return 'delta'
    return x.max() - x.min()

def find_result_dirs(root):
    result_dirs = []
    for root, dirs, files in os.walk(root):
        if is_intersection(dirs, _TASKS):
            result_dirs.append(root)
    return result_dirs

def parse_result_dir(dir_):
    s = dir_.replace('\\', '/').split('/')[2:]
    model = s[1]
    mode = s[0]
    if mode == 'contrast':
        mode += '--'+ s[2]
    return {'model': model, 'mode': mode}

def remove_results(dir_):
    for filename in os.listdir(dir_):
        if filename.startswith('results') and filename.endswith('.csv'):
            os.remove(os.path.join(dir_, filename))

def find_log(dir_):
    for root, dirs, files in os.walk(dir_):
        for file in files:
            if file.endswith('.log') and 'success' in file:
                return os.path.join(root, file)
    return None

def read_log(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    
    # remove timestamps
    lines = [line.split(' - ')[1] for line in lines]
    
    # exclude lines before 'Results:'
    for i, line in enumerate(lines):
        if line.startswith('Results:'):
            lines = lines[i+1:]
        
    result = {}
    for line in lines:
        if 'success' in line:
            key, value = line.split(':')
            result[key.strip()] = parse_value(value.strip())
    return result

def get_task_type(task):
    if 'google_robot' in task:
        return 'google_robot'
    if 'widowx' in task:
        return 'widowx'
    return 'total'

def bold_max(df):
    # bold max value in each row
    df = df.copy()
    for idx, row in df.iterrows():
        max_value = row.max()
        for col in df.columns:
            if row[col] == max_value:
                df.at[idx, col] = '\\textbf{' + ('{:.' + str(ROUND) + 'f}').format(row[col]) + '}'
                
    return df

def colorize_columns(df, key, color='red'):
    # colorize columns
    df = df.copy()
    for col in df.columns:
        if key in ' '.join(col):
            values = []
            for v in df[col].values:
                if isinstance(v, str):
                    values.append('\\cellcolor{' + color + '}{' + v + '}')
                else:
                    values.append('\\cellcolor{' + color + '}{' + ('{:.' + str(ROUND) + 'f}').format(v) + '}')
            df[col] = values
    return df


dfs = []
for root in _ROOTS:
    result_dirs = find_result_dirs(root)
    for result_dir in result_dirs:
        print(result_dir)
        remove_results(result_dir)
        
        results = []
        for task in _TASKS:
            log_path = find_log(os.path.join(result_dir, task))
            if not log_path:
                continue
            result = read_log(log_path)
            result['task'] = task
            results.append(result)

        if len(results) == 0:
            continue

        df = pd.DataFrame(results)
        df = pd.concat([pd.DataFrame(df.apply(mean_func, axis=0)).T, df], ignore_index=True)

        if 'google_robot' in ' '.join(df['task'].values):
            # df.loc[len(df)] = df[df['task'].str.startswith('google_robot')].apply(mean_func, prefix='google_robot')
            df = pd.concat([pd.DataFrame(df[df['task'].str.startswith('google_robot')].apply(mean_func, axis=0, prefix='google_robot')).T, df], ignore_index=True)
        if 'widowx' in ' '.join(df['task'].values):
            # df.loc[len(df)] = df[df['task'].str.startswith('widowx')].apply(mean_func, prefix='widowx')
            df = pd.concat([pd.DataFrame(df[df['task'].str.startswith('widowx')].apply(mean_func, axis=0, prefix='widowx')).T, df], ignore_index=True)
        
        average_success = df[df['task'] == 'average'].head(1)['success'].values[0]
        success = round(average_success, 2)
        df.round(ROUND).to_csv(os.path.join(result_dir, f'results_success{success}.csv'), index=False)

        result = parse_result_dir(result_dir)
        if result['mode'] in _MODES:
            for k, v in result.items():
                df[k] = v
            dfs.append(df)

df = pd.concat(dfs)
df = df[['model', 'mode', 'task', 'success', 'final_success', 'first_success']]

df['mode_type'] = df['mode'].apply(lambda x: x.split('--')[-1] if '--' in x else ' ')
df['mode'] = df['mode'].apply(lambda x: x.split('--')[0] if '--' in x else x)

df['task_type'] = df['task'].apply(lambda x: get_task_type(x))
df['task'] = df['task'].apply(lambda x: x.replace('google_robot_', '').replace('widowx_', ''))


df = df.replace({'octo-base': 'Octo', 'open-pi-zero': 'Pi-0', 'openvla-7b': 'OpenVLA',
                 'baseline': 'Base', 'contrast': '+PCD', 
                 'by=point_tracking': 'Pt.', 'by=box_tracking': 'Box', 'by=grounded_sam_tracking': 'Seg.',
                 'google_robot': 'Google Robot', 'widowx': 'WidowX',
                 'close_drawer': 'Close Drawer', 'move_near': 'Move Near', 'open_drawer': 'Open Drawer', 'pick_coke_can': 'Pick Coke Can',
                 'place_apple_in_closed_top_drawer': 'Apple Drawer', 'put_eggplant_in_basket': 'Eggplant Basket', 'spoon_on_towel': 'Spoon Towel',
                 'stack_cube': 'Stack Cube', 'carrot_on_plate': 'Carrot Plate', 
                 'total': 'Total', 'average': 'Average'})

df = df.sort_values(['model', 'mode'])

tmp_df = df[df['task_type'] == 'Total']
for model in tmp_df['model'].unique():
    tmp_df2 = tmp_df[tmp_df['model'] == model]
    base_row = tmp_df2[tmp_df2['mode'] == 'Base']
    for idx, row in tmp_df[tmp_df['model'] == model].iterrows():
        if row['mode'] == '+PCD':
            # add new row average_delta
            df.loc[len(df)] = row
            df.at[len(df)-1, 'task'] = 'Average Delta'
            df.at[len(df)-1, 'task_type'] = 'Total'
            df.at[len(df)-1, 'success'] = row['success'] - base_row['success'].values[0]
            df.at[len(df)-1, 'final_success'] = row['final_success'] - base_row['final_success'].values[0]
            df.at[len(df)-1, 'first_success'] = row['first_success'] - base_row['first_success'].values[0]
            df.at[len(df)-1, 'mode'] = '+PCD'
            df.at[len(df)-1, 'mode_type'] = row['mode_type']
            df.at[len(df)-1, 'model'] = row['model']

df = df.pivot_table(values='success', index=['task_type', 'task'], columns=['model', 'mode', 'mode_type'], sort=False)
df = df.reindex(['Total', 'Google Robot', 'WidowX'], level=0)
df = df.transpose().reindex(['Base', '+PCD'], level=1).transpose()

df.to_csv('results.csv')

# bold max values for each model
df = df.round(ROUND)
for model in df.columns.levels[0]:
    df.loc[:, (model, slice(None), slice(None))] = bold_max(df.loc[:, (model, slice(None), slice(None))])
df = df.astype(str)

df.iloc[1] = df.iloc[1].apply(lambda x: '{\\footnotesize\\textcolor{red}{+' + x + '}}' if x != 'nan' else ' ')

# colorize if model type is not baseline
df = colorize_columns(df, '+PCD', color=PCD_COLOR)

result = df.to_latex(float_format='%.2f')
result = result.replace(' model ', ' ').replace(' mode ', ' ').replace(' mode_type ', ' ') \
               .replace('task_type & task &  &  &  &  &  &  &  &  &  &  &  &  \\\\\n', '') \
               .replace('llllllllllllll', 'll|cccc|cccc|cccc') \
               .replace('\multicolumn{4}{r}{Octo}', '\\multicolumn{4}{c|}{Octo}') \
               .replace('\multicolumn{4}{r}{OpenVLA}', '\\multicolumn{4}{c|}{OpenVLA}') \
               .replace('\multicolumn{4}{r}{Pi-0}', '\\multicolumn{4}{c}{Pi-0}') \
               .replace('\multicolumn{3}{r}{+PCD} \\', '\\multicolumn{3}{c}{+PCD} \\\\') \
               .replace('\multicolumn{3}{r}{+PCD}', '\\multicolumn{3}{c|}{+PCD}') \
               .replace('\multirow[t]', '\\multirow[c]') \
               .replace('Google Robot', '\makecell{Google\\\\Robot}').replace('WidowX', 'WidX.') \
               .replace('+PCD', '\\cellcolor{'+PCD_COLOR+'}{+PCD}').replace('Pt.', '\\cellcolor{'+PCD_COLOR+'}{Pt.}') \
               .replace('Box', '\\cellcolor{'+PCD_COLOR+'}{Box}').replace('Seg.', '\\cellcolor{'+PCD_COLOR+'}{Seg.}') \
               .replace('\\cline{1-14}\n\\bottomrule', '\\bottomrule') \
               .replace('Base', '\\multirow{2}{*}{Base}') \
               .replace(' Average Delta ', ' ') \
               .replace('\multirow[c]{2}{*}{Total} & Average', '\\multirow[c]{2}{*}{Total} & \\multirow[c]{2}{*}{Average}') \
               .replace('Average', '\\textbf{Average}') \
               .replace('\cline{1-14}', '\\midrule') \
               .replace(' & & \multirow{2}{*}{Base}', '\multicolumn{2}{c|}{Task} & \multirow{2}{*}{Base}')

# write to file
with open('results.tex', 'w') as f:
    f.write('\\documentclass{article}\n')
    f.write('\\usepackage{booktabs}\n')
    f.write('\\usepackage{multirow}\n')
    f.write('\\usepackage{graphicx}\n')
    f.write('\\usepackage{caption}\n')
    f.write('\\usepackage{xcolor}\n')
    f.write('\\usepackage{colortbl}\n')
    f.write('\\usepackage{makecell}\n')
    f.write('\\begin{document}\n')
    f.write('\\begin{table}[h!]\n')
    f.write('\\tabcolsep=2pt\n')
    f.write(result)
    f.write('\\caption{Success Rate}\n')
    f.write('\\label{tab:success_rate}\n')
    f.write('\\end{table}\n')
    f.write('\\end{document}\n')
print('Results saved to results.tex')
