# 方法一
import os
import re
source = '/home/ubuntu/work_space/6DRepNet_Experiment/sixdrepnet/output/result' # 源文件夹
target = '/home/ubuntu/work_space/6DRepNet_Experiment/sixdrepnet/output/post_result' # 目标文件夹
if not os.path.exists(target): # 如果目标文件夹不存在，就创建一个
    os.mkdir(target)
for file in os.listdir(source): # 遍历源文件夹下的所有文件
    if file.endswith('.txt'): # 判断文件是否是文本文件
        with open(os.path.join(source, file), 'r') as f1, open(os.path.join(target, file), 'w') as f2: # 打开源文件和目标文件
            for line in f1: # 读取源文件的每一行
                if line.startswith('Yaw:'): # 判断是否以Yaw:开头
                    f2.write(line) # 写入目标文件

"""
# 方法二
import pathlib
import re
source = pathlib.Path('source') # 源文件夹
target = pathlib.Path('target') # 目标文件夹
if not target.exists(): # 如果目标文件夹不存在，就创建一个
    target.mkdir()
for file in source.iterdir(): # 遍历源文件夹下的所有文件
    if file.suffix == '.txt': # 判断文件是否是文本文件
        with file.open('r') as f1, (target / file.name).open('w') as f2: # 打开源文件和目标文件
            for line in f1: # 读取源文件的每一行
                if line.startswith('changeset:'): # 判断是否以changeset:开头
                    f2.write(line) # 写入目标文件
                    
"""