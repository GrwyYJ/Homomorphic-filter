""" 基于小波变换的同态滤波器参数测试脚本  

设置同态滤波参数
1. `alpha`：`alpha`     参数控制高通滤波器的增益，
2. `beta`：`beta`       参数也用于调整高通滤波器的增益。
3. `gamma`：`gamma`     参数控制滤波器的带宽
4. `cutoff`：`cutoff`   参数是滤波器的截止频率

create_date： 2023.10.12
update_date： 2023.10.14
"""
import numpy as np
from Homomorphicfilter import *

# 设置是否使用小波变换
isWaveTransform = False
# 滤波器说明
filter_doc = 'H(i, j) = alpha + beta * (1 - exp(-gamma * d^2 / cutoff^2))'
# 测试图片数量
num = 2
# 路径设置
data_root = '/home/grey/PycharmProjects/pythonProject/datasets/cataract'
classification = ['dark', 'cover', 'uneven']
result_root = '/home/grey/PycharmProjects/pythonProject/Homomorphic filter/result'
experinment_name = 'demo'


class MYSetting(MySetting):
    def __init__(self, testnum, filter_doc) -> None:
        self.testnum = testnum
        self.filter = filter_doc
    def setting(self, alpha, beta, gamma, cutoff):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.cutoff = cutoff


def test(isWaveTransform, device):
    i = 0
    setting = MYSetting(num,filter_doc)
    for classNum in range(0, 3):
        path = MyPath(data_root, classification[classNum],
                      result_root, experinment_name)
        for alpha in np.arange(0, 0.1, 0.05):
            for beta in np.arange(0.0, 1.5, 0.1):
                for gamma in np.arange(1000, 2000, 600000):
                    for cutoff in np.arange(100, 200, 20):
                        setting.setting(alpha=alpha, beta=beta, gamma=gamma, cutoff=cutoff)
                        # 保存路径:root + experinment_name + img.jpg
                        # 增加子目录，修改experinment_name
                        floder_path = os.path.join(
                            path.result_root, path.experinmetname, classification[classNum], 'test_'+str(i))
                        path.setExperinmetfloder(floder_path)

                        myprocessing(setting, path, isWaveTransform, device)
                        i = i + 1
        print('执行次数：', i)



if __name__ == '__main__':

    if isWaveTransform:
        print('Using WavesTransform')
    # 检查GPU是否可用
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU可用")
    else:
        device = torch.device("cpu")
        print("GPU不可用")
    test(isWaveTransform, device)
