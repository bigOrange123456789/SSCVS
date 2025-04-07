import csv
def create_csv(path, csv_head):
    with open(path, 'w', newline='') as f:
        csv_write = csv.writer(f)
        # csv_head = ["good","bad"]
        csv_write.writerow(csv_head)

def write_csv(path, data_row):
    # path  = "aa.csv"
    with open(path, 'a+', newline='') as f:
        csv_write = csv.writer(f)
        # data_row = ["1","2"]
        csv_write.writerow(data_row)

import os
from config import config
def getPath_csv():
    folder_path = os.path.join('logs', config.logname + '.log')
    try:# 使用 os.makedirs 创建文件夹，如果父文件夹不存在也会一并创建
        os.makedirs(folder_path, exist_ok=True)  # exist_ok=True 表示如果文件夹已存在，不会抛出异常
    except OSError as error:
        print(f"创建文件夹 '{folder_path}' 时出错: {error}")
    return folder_path