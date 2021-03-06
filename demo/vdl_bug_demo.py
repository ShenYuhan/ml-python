# coding=utf-8
import numpy as np
from visualdl import LogWriter
from PIL import Image
import time
def random_crop(img):
    '''
    此函数用于获取图片数据 img 的 100*100 的随机分块
    '''
    img = Image.open(img)
    w, h = img.size
    random_w = np.random.randint(0, w - 400)
    random_h = np.random.randint(0, h - 400)
    return img.crop((random_w, random_h, random_w + 400, random_h + 400))
# 创建 LogWriter 对象
log_writer = LogWriter("./log", sync_cycle=10)
# 创建 image 组件，模式为train, 采样数设为 ns
ns = 1
with log_writer.mode("train") as logger:
    input_image = logger.image(tag="test", num_samples=ns)
# 一般要设置一个变量 sample_num，用于记录当前已采样了几个 image 数据
sample_num = 0
for step in range(1000000):
    time.sleep(0.01)
    print(step)
    # 设置start_sampling() 的条件，满足条件时，开始采样
    if sample_num == 0:
        input_image.start_sampling()
    # 获取idx
    idx = input_image.is_sample_taken()
    # 如果 idx != -1，采样，否则跳过
    if idx != -1:
        # 获取图片数据
        image_path = "doge_big.jpeg"
        image_data = np.array(random_crop(image_path))
        # 使用 set_sample() 函数添加数据
        # flatten() 用于把 ndarray 由矩阵变为行向量
        #for index in range(10000):
        #    time.sleep(0.5)
        input_image.set_sample(idx, image_data.shape, image_data.flatten())
        sample_num += 1
        # 如果完成了当前轮的采样，则调用finish_sample()
        if sample_num % ns == 0:
            input_image.finish_sampling()
            sample_num = 0
