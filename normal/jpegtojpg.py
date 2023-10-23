import os

# 数据文件夹的路径
data_folder = "Dentaldata"

# 要更改的目标文件扩展名
target_extension = ".jpg"

# 遍历每个类别文件夹
for class_label in os.listdir(data_folder):
    class_folder = os.path.join(data_folder, class_label)
    
    # 检查文件夹是否存在
    if os.path.isdir(class_folder):
        for filename in os.listdir(class_folder):
            # 检查文件是否以 ".jpeg" 结尾
            if filename.endswith(".jpeg"):
                # 构建完整的文件路径
                file_path = os.path.join(class_folder, filename)
                
                # 构建新的文件名，将 ".jpeg" 替换为 ".jpg"
                new_filename = filename.replace(".jpeg", ".jpg")
                
                # 构建新的文件路径
                new_file_path = os.path.join(class_folder, new_filename)
                
                # 重命名文件
                os.rename(file_path, new_file_path)
                print(f"Renamed: {file_path} -> {new_file_path}")
