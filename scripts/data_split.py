import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 读取 CSV 文件
file_path = '/home/rxb/Explanation_MPP/FP-GNN_data/MoleculeNet/lipo.csv'
data = pd.read_csv(file_path)

# 设置随机种子以保证结果可复现
random_seed = 0

# 按照 0.8、0.1、0.1 的比例划分数据集
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=random_seed)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=random_seed)

# 验证划分结果
print(f"Train size: {len(train_data)}")
print(f"Validation size: {len(val_data)}")
print(f"Test size: {len(test_data)}")

# 创建保存结果的文件夹
output_folder = '/home/rxb/Explanation_MPP/FP-GNN_data/MoleculeNet/lipo'
os.makedirs(output_folder, exist_ok=True)  # 确保文件夹存在

# 保存划分后的数据集到指定文件夹中的 CSV 文件
train_data.to_csv(os.path.join(output_folder, 'train.csv'), index=False)
val_data.to_csv(os.path.join(output_folder, 'val.csv'), index=False)
test_data.to_csv(os.path.join(output_folder, 'test.csv'), index=False)

print(f"Data saved to {output_folder}")