import os
import time
import numpy as np

class Config(object):
    def __init__(self):

        # General
        self.epochs = 100        # 向前和向后传播中所有批次的单次训练迭代
        self.batch_size = 12   # 4
        self.date = 'FD4MM'      # FoCR
        self.numdata = 100000  # 100000
        self.workers = 16  # 32
        # Data
        self.test_batch_size = 1   # 一次训练所选取的样本数
        self.test_workers = 16
        self.numtestdata = 600  # 100000
        # Data
        self.data_dir = './datasets'
        self.dir_train = os.path.join(self.data_dir, 'train')
        # Setting
        self.frames_train = 'coco100000'        # you can adapt 100000 to a smaller number to train
        self.cursor_end = int(self.frames_train.split('coco')[-1])  #将字符串以'coco'开割形成一个字符串数组，然后再通过索引[-1]取出所得数组中的第一个元素的值
        self.coco_amp_lst = np.loadtxt(os.path.join(self.dir_train, 'train_mf.txt'))[:self.cursor_end]
        self.videos_train = []
        self.load_all = False        # Don't turn it on, unless you have such a big mem.
                                     # On coco dataset, 100, 000 sets -> 850G
        # Training
        self.lr = 1e-4               # 学习率
        self.betas = (0.9, 0.999)    # 用于计算梯度以及梯度平方的运行平均值的系数
        self.weight_decay=0.0
        self.batch_size_test = 1     # 测试训练的所选取的样本数
        self.preproc = ['poisson']   # ['poisson','resize', ]
        self.pretrained_weights = ''

        # Callbacks
        self.num_val_per_epoch = 1000  # 每多少print一下  x / all_num中的x
        self.save_dir = 'weights_date{}'.format(self.date)
        self.time_st = time.time()
        self.losses = []
        #### amp test
        ##################

        self.dir_amp0 = os.path.join(self.data_dir, 'systest/amp0/000000')
        self.dir_amp1 = os.path.join(self.data_dir, 'systest/amp0/000001')
        self.dir_amp2 = os.path.join(self.data_dir, 'systest/amp0/000002')
        self.dir_amp3 = os.path.join(self.data_dir, 'systest/amp0/000003')
        self.dir_amp4 = os.path.join(self.data_dir, 'systest/amp0/000004')
        self.dir_amp5 = os.path.join(self.data_dir, 'systest/amp0/000005')
        self.dir_amp6 = os.path.join(self.data_dir, 'systest/amp0/000006')
        self.dir_amp7 = os.path.join(self.data_dir, 'systest/amp0/000007')
        self.dir_amp8 = os.path.join(self.data_dir, 'systest/amp0/000008')
        self.dir_amp9 = os.path.join(self.data_dir, 'systest/amp0/000009')
        #####5amp
        self.dir_5amp0 = os.path.join(self.data_dir, 'systest/amp5/000000')
        self.dir_5amp1 = os.path.join(self.data_dir, 'systest/amp5/000001')
        self.dir_5amp2 = os.path.join(self.data_dir, 'systest/amp5/000002')
        self.dir_5amp3 = os.path.join(self.data_dir, 'systest/amp5/000003')
        self.dir_5amp4 = os.path.join(self.data_dir, 'systest/amp5/000004')
        self.dir_5amp5 = os.path.join(self.data_dir, 'systest/amp5/000005')
        self.dir_5amp6 = os.path.join(self.data_dir, 'systest/amp5/000006')
        self.dir_5amp7 = os.path.join(self.data_dir, 'systest/amp5/000007')
        self.dir_5amp8 = os.path.join(self.data_dir, 'systest/amp5/000008')
        self.dir_5amp9 = os.path.join(self.data_dir, 'systest/amp5/000009')
        ####10amp
        self.dir_10amp0 = os.path.join(self.data_dir, 'systest/amp10/000000')
        self.dir_10amp1 = os.path.join(self.data_dir, 'systest/amp10/000001')
        self.dir_10amp2 = os.path.join(self.data_dir, 'systest/amp10/000002')
        self.dir_10amp3 = os.path.join(self.data_dir, 'systest/amp10/000003')
        self.dir_10amp4 = os.path.join(self.data_dir, 'systest/amp10/000004')
        self.dir_10amp5 = os.path.join(self.data_dir, 'systest/amp10/000005')
        self.dir_10amp6 = os.path.join(self.data_dir, 'systest/amp10/000006')
        self.dir_10amp7 = os.path.join(self.data_dir, 'systest/amp10/000007')
        self.dir_10amp8 = os.path.join(self.data_dir, 'systest/amp10/000008')
        self.dir_10amp9 = os.path.join(self.data_dir, 'systest/amp10/000009')
        ####20amp
        self.dir_20amp0 = os.path.join(self.data_dir, 'systest/amp20/000000')
        self.dir_20amp1 = os.path.join(self.data_dir, 'systest/amp20/000001')
        self.dir_20amp2 = os.path.join(self.data_dir, 'systest/amp20/000002')
        self.dir_20amp3 = os.path.join(self.data_dir, 'systest/amp20/000003')
        self.dir_20amp4 = os.path.join(self.data_dir, 'systest/amp20/000004')
        self.dir_20amp5 = os.path.join(self.data_dir, 'systest/amp20/000005')
        self.dir_20amp6 = os.path.join(self.data_dir, 'systest/amp20/000006')
        self.dir_20amp7 = os.path.join(self.data_dir, 'systest/amp20/000007')
        self.dir_20amp8 = os.path.join(self.data_dir, 'systest/amp20/000008')
        self.dir_20amp9 = os.path.join(self.data_dir, 'systest/amp20/000009')
        ####50amp
        self.dir_50amp0 = os.path.join(self.data_dir, 'systest/amp50/000000')
        self.dir_50amp1 = os.path.join(self.data_dir, 'systest/amp50/000001')
        self.dir_50amp2 = os.path.join(self.data_dir, 'systest/amp50/000002')
        self.dir_50amp3 = os.path.join(self.data_dir, 'systest/amp50/000003')
        self.dir_50amp4 = os.path.join(self.data_dir, 'systest/amp50/000004')
        self.dir_50amp5 = os.path.join(self.data_dir, 'systest/amp50/000005')
        self.dir_50amp6 = os.path.join(self.data_dir, 'systest/amp50/000006')
        self.dir_50amp7 = os.path.join(self.data_dir, 'systest/amp50/000007')
        self.dir_50amp8 = os.path.join(self.data_dir, 'systest/amp50/000008')
        self.dir_50amp9 = os.path.join(self.data_dir, 'systest/amp50/000009')
        ####100amp
        self.dir_100amp0 = os.path.join(self.data_dir, 'systest/amp100/000000')
        self.dir_100amp1 = os.path.join(self.data_dir, 'systest/amp100/000001')
        self.dir_100amp2 = os.path.join(self.data_dir, 'systest/amp100/000002')
        self.dir_100amp3 = os.path.join(self.data_dir, 'systest/amp100/000003')
        self.dir_100amp4 = os.path.join(self.data_dir, 'systest/amp100/000004')
        self.dir_100amp5 = os.path.join(self.data_dir, 'systest/amp100/000005')
        self.dir_100amp6 = os.path.join(self.data_dir, 'systest/amp100/000006')
        self.dir_100amp7 = os.path.join(self.data_dir, 'systest/amp100/000007')
        self.dir_100amp8 = os.path.join(self.data_dir, 'systest/amp100/000008')
        self.dir_100amp9 = os.path.join(self.data_dir, 'systest/amp100/000009')
        ########
        self.dir_001noise0 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.01/000000')
        self.dir_001noise1 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.01/000001')
        self.dir_001noise2 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.01/000002')
        self.dir_001noise3 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.01/000003')
        self.dir_001noise4 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.01/000004')
        self.dir_001noise5 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.01/000005')
        self.dir_001noise6 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.01/000006')
        self.dir_001noise7 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.01/000007')
        self.dir_001noise8 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.01/000008')
        self.dir_001noise9 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.01/000009')
        ########
        self.dir_005noise0 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.05/000000')
        self.dir_005noise1 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.05/000001')
        self.dir_005noise2 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.05/000002')
        self.dir_005noise3 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.05/000003')
        self.dir_005noise4 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.05/000004')
        self.dir_005noise5 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.05/000005')
        self.dir_005noise6 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.05/000006')
        self.dir_005noise7 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.05/000007')
        self.dir_005noise8 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.05/000008')
        self.dir_005noise9 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.05/000009')
        ############
        ########
        self.dir_01noise0 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.1/000000')
        self.dir_01noise1 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.1/000001')
        self.dir_01noise2 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.1/000002')
        self.dir_01noise3 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.1/000003')
        self.dir_01noise4 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.1/000004')
        self.dir_01noise5 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.1/000005')
        self.dir_01noise6 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.1/000006')
        self.dir_01noise7 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.1/000007')
        self.dir_01noise8 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.1/000008')
        self.dir_01noise9 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.1/000009')
        ############
        ########
        self.dir_02noise0 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.2/000000')
        self.dir_02noise1 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.2/000001')
        self.dir_02noise2 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.2/000002')
        self.dir_02noise3 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.2/000003')
        self.dir_02noise4 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.2/000004')
        self.dir_02noise5 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.2/000005')
        self.dir_02noise6 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.2/000006')
        self.dir_02noise7 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.2/000007')
        self.dir_02noise8 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.2/000008')
        self.dir_02noise9 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.2/000009')
        ############
        ########
        self.dir_05noise0 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.5/000000')
        self.dir_05noise1 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.5/000001')
        self.dir_05noise2 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.5/000002')
        self.dir_05noise3 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.5/000003')
        self.dir_05noise4 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.5/000004')
        self.dir_05noise5 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.5/000005')
        self.dir_05noise6 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.5/000006')
        self.dir_05noise7 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.5/000007')
        self.dir_05noise8 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.5/000008')
        self.dir_05noise9 = os.path.join(self.data_dir, 'noise_sysamp20/noise0.5/000009')
        ############

        self.dir_baby = os.path.join(self.data_dir, 'train/train_vid_frames/val_baby')



# def mse(imageA, imageB):
#     # the 'Mean Squared Error' between the two images is the sum of the squared difference between the two images
#     mse_error = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
#     mse_error /= float(imageA.shape[0] * imageA.shape[1] * 255 )
#     mse_error /= (np.mean((imageA.astype("float"))))**2
#     # return the MSE. The lower the error, the more "similar" the two images are.
#     return mse_error

# def mae(imageA, imageB):
#     mae = np.sum(np.absolute((imageB.astype("float") - imageA.astype("float"))))
#     mae /= float(imageA.shape[0] * imageA.shape[1] * 255)
#     if (mae < 0):
#         return mae * -1
#     else:
#         return mae

from skimage.metrics import mean_squared_error as compare_mse
from sklearn.metrics import mean_absolute_error as compare_mae
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
import math
from cmath import sqrt

# def calc_mae(img1, img2):
#     mae_score = compare_mae(img1, img2)
#     return mae_score

def calc_mse(img1, img2):
    mse_score = np.mean((img1 / 255. - img2 / 255.) ** 2)
    return mse_score

def calc_rmse(img1, img2):
    mse_score = np.mean((img1 / 255. - img2 / 255.) ** 2)
    rmse_score = sqrt(mse_score)
    return rmse_score

def calc_psnr(img1, img2): #这里输入的是（0,255）的灰度或彩色图像，如果是彩色图像，则numpy.mean相当于对三个通道计算的结果再求均值
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10: # 如果两图片差距过小代表完美重合
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse)) # 将对数中pixel_max的平方放了下来


# def calc_psnr(img1, img2):

#     # img1 = Image.open(img1_path)
#     # img2 = Image.open(img2_path)
#     # img2 = img2.resize(img1.size)
#     # img1, img2 = np.array(img1), np.array(img2)
#     # 此处的第一张图片为真实图像，第二张图片为测试图片
#     # 此处因为图像范围是0-255，所以data_range为255，如果转化为浮点数，且是0-1的范围，则data_range应为1
#     psnr_score = psnr(img1, img2, data_range=255)
#     return psnr_score

def calc_ssim(img1, img2):

    # img1 = Image.open(img1_path).convert('L')
    # img2 = Image.open(img2_path).convert('L')
    # img2 = img2.resize(img1.size)
    # img1, img2 = np.array(img1), np.array(img2)
    # 此处因为转换为灰度值之后的图像范围是0-255，所以data_range为255，如果转化为浮点数，且是0-1的范围，则data_range应为1
    ssim_score = ssim(img1, img2, data_range=255 , multichannel=True)
    return ssim_score


import json

""" configuration json """
class Configjson(dict): 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Configjson(config)