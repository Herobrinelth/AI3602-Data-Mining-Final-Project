import pandas as pd
from datetime import datetime

def convert_csv_to_dat(input_csv, output_dat):
    # 读取CSV文件
    df = pd.read_csv(input_csv)
    
    # 提取前4列
    df_extracted = df.iloc[:, :4]
    
    # 将日期转换为时间戳
    df_extracted['date'] = df_extracted['date'].apply(
        lambda x: int(datetime.strptime(x, '%Y-%m-%d').timestamp())
    )
    
    # 重新排列列的顺序
    df_extracted = df_extracted[['user_id', 'recipe_id', 'rating', 'date']]
    
    # 保存到DAT文件
    # df_extracted.to_csv(output_dat, index=False, header=False, sep='::')
    
    # 将数据框转换为字符串，并用 "::" 分隔
    data_string = df_extracted.to_string(index=False, header=False)
    data_string = '\n'.join(['::'.join(line.split()) for line in data_string.split('\n')])
    
    # 写入DAT文件
    with open(output_dat, 'w') as file:
        file.write(data_string)

# 示例使用
convert_csv_to_dat('RAW_interactions.csv', 'processed_interactions.dat')
convert_csv_to_dat('interactions_test.csv', 'test_set.dat')
convert_csv_to_dat('interactions_train.csv', 'train_set.dat')
