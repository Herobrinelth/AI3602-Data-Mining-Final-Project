import json

def preview_yelp_reviews(file_path, num_records=5):
    """
    预览 Yelp 评论数据集的前几条记录。
    
    参数：
    file_path (str): JSON 文件的路径。
    num_records (int): 要预览的记录数量，默认为 5。
    
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # 打开文件并逐行读取
            for i, line in enumerate(file):
                if i >= num_records:
                    break
                # 将 JSON 记录转换为 Python 字典
                review = json.loads(line)
                print(json.dumps(review, indent=4, ensure_ascii=False))
                print("\n" + "-"*80 + "\n")
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
    except json.JSONDecodeError:
        print(f"文件 {file_path} 不是有效的 JSON 文件。")

# 示例用法
preview_yelp_reviews('yelp_academic_dataset_review.json', num_records=3)
