import os

# 資料夾路徑
folder_path = "/home/ethan/original_videos_h264/"

# 讀取 txt 檔案
with open("labels_ori.txt", "r") as file:
    lines = file.readlines()

# 檢查每個檔案是否存在於資料夾中
existing_files = []
for line in lines:
    file_name = line.split(",")[0].strip()
    file_path = os.path.join(folder_path, file_name)
    if os.path.exists(file_path):
        existing_files.append(line)

# 將存在的檔案名稱寫回新的 txt 檔案
with open("labels.txt", "w") as file:
    file.writelines(existing_files)

print("檔案篩選完成")