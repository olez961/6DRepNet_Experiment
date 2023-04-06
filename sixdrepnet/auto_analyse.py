import os
import subprocess
import argparse

# 使用示例
'''
python auto_analyse.py  --folder_path /home/ubuntu/work_space/6DRepNet_Experiment/sixdrepnet/output/snapshots/SixDRepNet_1680017093_bs100_Pose_300W_LP_GeodesicLoss \
                        --test_file test_pth_AFLW2000.py \
                        --other_information None
或
CUDA_VISIBLE_DEVICES=1 \
python auto_analyse.py  --folder_path /home/ubuntu/work_space/6DRepNet_Experiment/sixdrepnet/output/snapshots/SixDRepNet_1680017093_bs100_Pose_300W_LP_GeodesicLoss \
                        --test_file test_pth_BIWI.py \
                        --other_information None
'''

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the 6DRepNet.')
    # 下面这段代码在用的时候这个参数可能被当作一个bool值
    # 所以我把原代码的默认值0改成1了
    parser.add_argument(
        '--folder_path', dest='folder_path',
        help='Path to the folder you save the snapshots.',
        default='/home/ubuntu/work_space/6DRepNet_Experiment/sixdrepnet/output/snapshots/SixDRepNet_1680017093_bs100_Pose_300W_LP_GeodesicLoss', type=str) 
    parser.add_argument(
        '--test_file', dest='test_file',
        help='The test file you want to use.',
        default='test_pth_AFLW2000.py', type=str) # test_pth_BIWI.py
    parser.add_argument(
        '--other_information', dest='other_information', help='Other information for marking.',
        default='', type=str) # 这里可以加一些信息，比如用300w-lp训练的模型测试biwi准确度的

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    # 指定文件夹的路径
    folder_path = args.folder_path
    test_file =  args.test_file
    other_information = args.other_information

    # 获取文件夹中的所有文件名
    files = os.listdir(folder_path)

    files = [x for x in files if 'SixDRepNet' not in x]

    # 对文件名进行排序
    files.sort(key=lambda x : int(x.split('.')[0].split('_')[-1]))

    results = folder_path[folder_path.rfind('/') + 1:] + other_information

    # 打开文本文件，用于保存执行结果
    # a模式是追加写入，而w模式当文件中已存在内容时会清空原有内容
    with open('./output/result/' + results +'.txt', 'w', buffering=128) as f:
        for file in files:
            # 构建完整的文件路径
            file_path = os.path.join(folder_path, file)
            # 调用subprocess.run函数执行该文件
            result = subprocess.run(['python3', test_file, '--snapshot', file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # 将执行结果写入文本文件中
            print(f'{file}: {result.stdout.decode()}\n')
            f.write(f'{file}: {result.stdout.decode()}\n')
            # 每次处理完一个文件便将结果写入输出文件中，避免最后写入导致错误
            f.flush()
        f.close()