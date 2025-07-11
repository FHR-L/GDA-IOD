import re
import pandas as pd
import matplotlib.pyplot as plt

import re


def extract_lines_from_log(log_file_path):
    # 正则表达式匹配所需的行
    pattern = r'\| (\w+\s*\w*)\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|'

    # 存储提取的结果
    category_result = {}

    # 打开并读取文件
    with open(log_file_path, 'r') as file:
        for line in file:
            # 查找匹配的行
            match = re.search(pattern, line)
            if match:
                # 如果找到匹配项，则将结果添加到列表中
                category = str(match.group(1).strip())
                if category == 'category':
                    continue
                mAP = float(match.group(2).strip())
                mAP_50 = float(match.group(3).strip())
                mAP_75 = float(match.group(4).strip())
                mAP_s = float(match.group(5).strip())
                mAP_m = float(match.group(6).strip())
                mAP_l = float(match.group(7).strip())

                # 将结果添加到列表中
                print(mAP)
                if category not in category_result.keys():
                    category_result[category] = []
                category_result[category].append(
                    {
                        'mAP': mAP,
                        'mAP_50': mAP_50,
                        'mAP_75': mAP_75,
                        'mAP_s': mAP_s,
                        'mAP_m': mAP_m,
                        'mAP_l': mAP_l
                    }
                )

    return category_result

def parse_log(log_file):
    pattern = r'Epoch\(val\) \[(\d+)\]\[\d+/172\] .* Evaluating bbox.*\n(.*?)\n', re.DOTALL
    results = []
    with open(log_file, 'r') as file:
        content = file.read()
        for match in re.finditer(pattern, content):
            epoch = int(match.group(1))
            table_str = match.group(2)
            lines = [line.split('|') for line in table_str.split('\n')]
            headers = [header.strip() for header in lines[0]]
            data = {headers[i]: [float(cell.strip()) if i > 0 else cell.strip() for cell in row]
                    for i, row in enumerate(lines[1:])}
            df = pd.DataFrame(data, columns=headers)
            df['Epoch'] = epoch
            results.append(df)
    return pd.concat(results)


def plot_results(df):
    categories = df['category'].unique()

    fig, axs = plt.subplots(len(categories), 3, figsize=(15, len(categories) * 5))

    for i, category in enumerate(categories):
        category_df = df[df['category'] == category]

        axs[i, 0].plot(category_df['Epoch'], category_df['mAP'], label=f'{category} mAP')
        axs[i, 0].set_title(f'{category} mAP')
        axs[i, 0].set_xlabel('Epoch')
        axs[i, 0].set_ylabel('mAP')
        axs[i, 0].legend()

        axs[i, 1].plot(category_df['Epoch'], category_df['mAP_50'], label=f'{category} mAP_50')
        axs[i, 1].set_title(f'{category} mAP_50')
        axs[i, 1].set_xlabel('Epoch')
        axs[i, 1].set_ylabel('mAP_50')
        axs[i, 1].legend()

        axs[i, 2].plot(category_df['Epoch'], category_df['mAP_75'], label=f'{category} mAP_75')
        axs[i, 2].set_title(f'{category} mAP_75')
        axs[i, 2].set_xlabel('Epoch')
        axs[i, 2].set_ylabel('mAP_75')
        axs[i, 2].legend()

    plt.tight_layout()
    plt.show()

def count_step_individual_mAP(category_result, pattern='70+10'):
    categories_nums = pattern.split('+')
    for key in category_result:
        print(key)

if __name__ == '__main__':
    log_file = r'F:\PYTHON\ERD10184\work_dirs\increse-70-10-shuffle-v1-on-pretrained\20240826_162141\20240826_162141.log'
    category_result = extract_lines_from_log(log_file)
    count_step_individual_mAP(category_result)