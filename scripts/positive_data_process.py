import csv

# 输入文件名
input_file = "/home/rxb/Explanation_MPP/FP-GNN_data/MoleculeNet/bace/test.csv"  # 替换为您的输入文件名
output_file = "/home/rxb/Explanation_MPP/FP-GNN_data/MoleculeNet/bace/positive_bace.csv"  # 输出文件名

# 打开输入文件并提取positive属性的分子
with open(input_file, "r") as infile, open(output_file, "w", newline="") as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # 写入输出文件的表头
    header = next(reader)  # 读取表头
    writer.writerow(header)  # 输出文件保留原始表头

    # 遍历文件内容
    for row in reader:
        smiles, classification = row[0], row[1]  # 提取smiles和属性列
        if classification == "1":  # 判断是否为positive属性
            writer.writerow(row)  # 写入输出文件

print(f"Positive SMILES 分子及其标签已保存至 {output_file}")