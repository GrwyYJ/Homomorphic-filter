from Homomorphicfilter import *

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

            # new_image = contactPic(image, image_new)
            myPath.path = os.path.join(myPath.experinmetfloder, imgname)
            cv2.imwrite(myPath.path, image_new)

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
