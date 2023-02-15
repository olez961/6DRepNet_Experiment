import os
import subprocess

# 指定文件夹的路径
folder_path = './output/snapshots/SixDRepNet_1676388190_bs80_Pose_300W_LP_GeodesicLoss_Convnext'

# 获取文件夹中的所有文件名
files = os.listdir(folder_path)

# 对文件名进行排序
files.sort(key=lambda x : int(x.split('.')[0].split('_')[-1]))

results = folder_path[folder_path.rfind('/') + 1:]

# 打开文本文件，用于保存执行结果
# a模式是追加写入，而w模式当文件中已存在内容时会清空原有内容
with open('./output/result/' + results +'.txt', 'w', buffering=128) as f:
    for file in files:
        # 构建完整的文件路径
        file_path = os.path.join(folder_path, file)
        # 调用subprocess.run函数执行该文件
        result = subprocess.run(['python3', 'test_pth_AFLW2000.py', '--snapshot', file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # 将执行结果写入文本文件中
        print(f'{file}: {result.stdout.decode()}\n')
        f.write(f'{file}: {result.stdout.decode()}\n')
        # 每次处理完一个文件便将结果写入输出文件中，避免最后写入导致错误
        f.flush()
    f.close()