""" 彩色图片的基于小波变换的同态滤波的实现
同态滤波波器系统函数：
参考：融合同态滤波和小波变换的图像去雾算法研究
H(i, j) = alpha + beta * (1 - exp(-gamma * d^2 / cutoff^2))
H(u, v) = (HH-HL) {1-exp[-c (D2 (u, v) /D02) ]}+HL 
以上两者无异

update: 1.增加与图片的对比，保存设置参数
        2.增加class以适应测试脚本
        3.增加pytorch使用cuda加速滤波器计算
        4.增加小波变换
        5.为小波变换增加cuda配置
        6.优化cuda配置
        7.重新设计API

create_date： 2023.10.12
update_date： 2023.10.13
"""
import os
import cv2
import numpy as np
import time
import torch
import pywt


def homomorphic_filter_wavetransform(image, alpha, beta, gamma, cutoff, device):
    # 转换图像为自然对数域
    img_log = np.log1p(np.array(image, dtype="float"))

    # 将彩色图像拆分为通道
    channels = cv2.split(img_log)

    # 对每个通道进行同态滤波
    filtered_channels = []
    for channel in channels:
        # 使用小波变换将图像分解为低频和高频部分
        channel = torch.from_numpy(channel)
        LL, (LH, HL, HH) = pywt.dwt2(channel.numpy(), 'bior1.3')
        LL = torch.from_numpy(LL).to(device)

        # 对低频部分进行同态滤波
        LL_filtered = apply_homomorphic_filter_wavetransform(
            LL, alpha, beta, gamma, cutoff, device)
        LL_filtered = LL_filtered.cpu().detach().numpy()

        # 重构过滤后的通道
        coeffs_filtered = (LL_filtered, (LH, HL, HH))
        channel_filtered = pywt.idwt2(coeffs_filtered, 'bior1.3')

        # 反向指数变换
        channel_filtered = np.expm1(channel_filtered)
        channel_filtered = np.uint8(np.clip(channel_filtered, 0, 255))

        filtered_channels.append(channel_filtered)

    # 合并通道以获得输出图像
    img_filtered = cv2.merge(filtered_channels)

    return img_filtered


def apply_homomorphic_filter_wavetransform(LL, alpha, beta, gamma, cutoff, device):
    # 构建滤波器
    rows, cols = LL.shape
    center_row, center_col = rows // 2, cols // 2

    # 创建一个网格，以便计算所有像素点的d2
    x = torch.arange(cols).to(device) - center_col
    y = torch.arange(rows).to(device) - center_row
    X, Y = torch.meshgrid(y, x, indexing='ij')
    d2 = X**2 + Y**2

    H = alpha + beta * (1 - torch.exp(-gamma * d2 / (cutoff ** 2)))

    # 应用滤波器到图像
    img_filtered = LL * H

    return img_filtered


def homomorphic_filter(image, alpha, beta, gamma, cutoff, device):
    """ 在同态滤波
    H是频域滤波器，它的形式通常是高斯函数或某种低通滤波器。
    H的目的是控制哪些频率成分将被增强，哪些将被抑制。
    在下述示例中，H的形式是：
    H(i, j) = alpha + beta * (1 - exp(-gamma * d^2 / cutoff^2))
    其中：
    - `alpha` 控制低频部分的增强。
    - `beta` 控制高频部分的增强。
    - `gamma` 是一个参数，用于调整频域响应的形状。
    - `d` 是频域中像素 `(i, j)` 到图像中心的距离。
    - `cutoff` 是一个截止频率，控制滤波器的截止频率。
    """

    # 拆分图像为红色、绿色和蓝色通道
    r, g, b = cv2.split(image)

    # 对每个通道应用同态滤波
    # print('Using CPU')
    r_filtered = apply_homomorphic_filter_gpu(
        r, alpha, beta, gamma, cutoff, device)
    g_filtered = apply_homomorphic_filter_gpu(
        g, alpha, beta, gamma, cutoff, device)
    b_filtered = apply_homomorphic_filter_gpu(
        b, alpha, beta, gamma, cutoff, device)

    # 合并通道以生成彩色图像
    filtered_image = cv2.merge((r_filtered, g_filtered, b_filtered))

    return filtered_image


def apply_homomorphic_filter_gpu(channel, alpha, beta, gamma, cutoff, device):
    # 转换通道到对数域
    log_channel = torch.log1p(torch.tensor(
        channel, dtype=torch.float64).to(device))

    # 傅立叶变换
    f_transform = torch.fft.fft2(log_channel)  # 形状 高*宽

    # 构建滤波器
    rows, cols = channel.shape
    center_row, center_col = rows // 2, cols // 2
    H = torch.zeros((rows, cols), dtype=torch.complex64).to(device)

    x = torch.arange(cols, dtype=torch.float64).to(device) - center_col
    y = torch.arange(rows, dtype=torch.float64).to(device) - center_row
    # X, Y = torch.meshgrid(x, y)  # 形状 宽*高
    X, Y = torch.meshgrid(y, x, indexing='ij')
    d2 = X**2 + Y**2

    H = alpha + beta * (1 - torch.exp(-gamma * d2 / (cutoff ** 2)))

    # 应用滤波器
    filtered_channel = torch.real(torch.fft.ifft2(f_transform * H))

    # 反变换
    filtered_channel = torch.exp(filtered_channel) - 1

    # 规范化到0-255范围
    filtered_channel = (filtered_channel - torch.min(filtered_channel)) / \
        (torch.max(filtered_channel) - torch.min(filtered_channel)) * 255

    return filtered_channel.cpu().numpy().astype(np.uint8)


def contactPic(img1, img2):
    height1, width1, _ = img1.shape
    height2, width2, _ = img2.shape

    new_width = width1 + width2
    new_height = max(height1, height2)

    new_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    new_image[0:height1, 0:width1] = img1
    cv2.putText(new_image, 'ORIGINAL', (0+10, height1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 3)
    new_image[0:new_height, width1:new_width] = img2
    cv2.putText(new_image, 'ENHANCED', (width1+10, height2-5),
                cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 3)
    return new_image


def get_flienames(root):
    name = []
    path = []
    for filename in os.listdir(root):
        if os.path.isfile(os.path.join(root, filename)):
            path.append(os.path.join(root, filename))
            name.append(filename)
    return name, path


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


class MyPath:
    path = ''

    def __init__(self, img_root_path, img_floder_name, result_root_path, experinmet_name) -> None:
        self.root = img_root_path
        self.classification = img_floder_name
        self.imgfloder = os.path.join(img_root_path, img_floder_name)
        self.result_root = result_root_path
        self.experinmetname = experinmet_name
        self.experinmetfloder = os.path.join(
            self.result_root, self.experinmetname)
        self.imgname, self.imgpath = get_flienames(self.imgfloder)
        create_folder(self.experinmetfloder)

    def setExperinmetfloder(self, path):
        self.experinmetfloder = path
        create_folder(self.experinmetfloder)
        print(f'create folder{path}')

    def setPath(self, path):
        self.path = path


class MySetting:

    testnum = 0
    runTime = 0
    filter = 'No description'

    def __init__(self, alpha, beta, gamma, cutoff) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.cutoff = cutoff

    def setFilterDoc(self, filter):
        self.filter = filter

    def setTestnum(self, num):
        self.testnum = num

    def setTime(self, runtime):
        self.runTime = runtime

    def save(self, root):
        if not os.path.exists(root):
            os.makedirs(root)
        with open(os.path.join(root, 'saveData.txt'), 'w') as file:
            file.write(f'Process image number = {self.testnum}\n')
            file.write(f'Running time = {self.runTime}\n')
            file.write(f'Filter description : {self.filter}\n')
            file.write(f'alpha = {self.alpha}\n')
            file.write(f'beta = {self.beta}\n')
            file.write(f'gamma = {self.gamma}\n')
            file.write(f'cutoff = {self.cutoff}\n')


def myprocessing(mySetting, myPath, isWaveTransform, device):
    start = time.perf_counter()
    for index, imgname in enumerate(myPath.imgname):
        if index < mySetting.testnum:
            print(f'processing img_num : {index} \t img_name : {imgname}')
            image = cv2.imread(os.path.join(myPath.imgfloder, imgname))
            torch.from_numpy(image).to(device)
            if isWaveTransform:
                image_new = homomorphic_filter_wavetransform(image,
                                                             mySetting.alpha,
                                                             mySetting.beta,
                                                             mySetting.gamma,
                                                             mySetting.cutoff, device)
            else:
                image_new = homomorphic_filter(image,
                                               mySetting.alpha,
                                               mySetting.beta,
                                               mySetting.gamma,
                                               mySetting.cutoff, device)

            new_image = contactPic(image, image_new)
            myPath.path = os.path.join(myPath.experinmetfloder, imgname)
            cv2.imwrite(myPath.path, new_image)

    end = time.perf_counter()
    runTime = end - start
    print("运行时间：", runTime)
    mySetting.setTime(runTime)
    mySetting.save(myPath.experinmetfloder)


if __name__ == '__main__':
    print('Homomorphicfilter')
    # 设置是否使用小波变换
    isWaveTransform = False
    if isWaveTransform:
        print('Using WaveTransform')


    # 路径设置
    data_root = '/home/grey/PycharmProjects/pythonProject/Cataracts_2/code/results/arcnet/test_latest_0_filter'
    classification = 'images'
    result_root = '/home/grey/PycharmProjects/pythonProject/Cataracts_2/code/results/arcnet/test_latest_0_filter'
    experinment_name = 'target'
    # 滤波器参数设置
    filter = 'H(i, j) = alpha + beta * (1 - exp(-gamma * d^2 / cutoff^2)) \n H (u, v) = (HH-HL) {1-exp[-c (D2 (u, v) /D02) ]}+HL   '
    alpha = 0.0         # 调节亮度，越大越亮
    beta = 1.0          # 调节亮度，越大越亮
    gamma = 1.6      # 中间空洞大小，越小越大
    cutoff = 51
    # 测试图片数量
    num = 72

    path = MyPath(data_root, classification, result_root, experinment_name)
    setting = MySetting(alpha=alpha, beta=beta, gamma=gamma, cutoff=cutoff)
    setting.setFilterDoc(filter)
    setting.setTestnum(num)

    # 检查GPU是否可用
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    # device = torch.device("cpu")
    print('Start')
    myprocessing(setting, path, isWaveTransform, device)
    print('End')
