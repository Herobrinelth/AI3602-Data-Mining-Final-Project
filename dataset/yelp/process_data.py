import json
import time

def convert_to_timestamp(date_str):
    """
    将日期字符串转换为时间戳。
    """
    # 定义日期格式
    date_format = "%Y-%m-%d %H:%M:%S"
    # 将日期字符串转换为时间元组
    time_tuple = time.strptime(date_str, date_format)
    # 将时间元组转换为时间戳
    timestamp = int(time.mktime(time_tuple))
    return timestamp

def process_yelp_reviews(input_file, output_file):
    """
    处理 Yelp 评论数据集，将所需字段提取并转换后存储到新的 .dat 文件中。
    
    参数：
    input_file (str): 输入的 JSON 文件路径。
    output_file (str): 输出的 .dat 文件路径。
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                review = json.loads(line)
                user_id = hash(review['user_id'])  # 将 user_id 转换为整数
                business_id = hash(review['business_id'])  # 将 business_id 转换为整数
                stars = review['stars']
                date = convert_to_timestamp(review['date'])  # 将日期转换为时间戳

                # 构造输出字符串，并以 '::' 分隔字段
                output_line = f"{user_id}::{business_id}::{stars}::{date}\n"
                outfile.write(output_line)
    except FileNotFoundError:
        print(f"文件 {input_file} 未找到。")
    except json.JSONDecodeError:
        print(f"文件 {input_file} 不是有效的 JSON 文件。")

# 示例用法
process_yelp_reviews('yelp_academic_dataset_review.json', 'yelp_reviews.dat')
