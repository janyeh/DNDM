#!/usr/bin/python3
import csv
import argparse
import itertools
import math
import numpy as np
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torchvision.models import vgg16
from perceptual import LossNetwork
from datasets2 import  TrainDatasetFromFolder4,TrainDatasetFromFolder2,TestDatasetFromFolder1

from ECLoss import DCLoss
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ski_ssim
from CAPLOSS import *
from GFN20 import *
# from model11242 import *
from FFA02 import ffa
from Labloss import LabLoss
from utils21 import *
import os
from torch.utils.checkpoint import checkpoint
import torch.backends.cudnn
from typing import Optional, Tuple, Union

# JanYeh DEBUG BEGIN
#torch.backends.cudnn.benchmark = True
torch.backends.cuda.max_memory_allocated = 4294967296  # 4GB limit
# Use deterministic algorithms
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# JanYeh DEBUG END

# --- TRAINING STABILITY PARAMETERS ---
TOTAL_EPOCHS = 20  # 總訓練回合數

# --- STABILITY CONFIG ---
STABILITY_CONFIG = {
    'gradient_clip_norm': 1.0,          # 梯度裁剪的最大範數值
    'loss_scale_threshold': 100.0,      # 縮小大型損失的閾值
    'loss_scale_factor': 0.01,          # 超過閾值時的損失縮小係數
    'tensor_value_clip': (-1.0, 1.0),   # 張量值的安全範圍
    'max_grad_value': 1.0,              # 梯度值裁剪的最大值
    'enable_anomaly_detection': True,    # 啟用自動梯度異常檢測
}

# --- 記憶體管理 ---
MEMORY_CONFIG = {
    'max_gpu_memory': 4294967296,       # GPU記憶體限制(4GB)
    'enable_cuda_benchmark': False,      # 停用CUDA基準測試以提高穩定性
    'deterministic': True,              # 啟用確定性訓練
    'enable_cudnn_benchmark': False,     # 停用cuDNN基準測試以保持一致性
    'batch_size': 1,                    # 小批次大小以保持穩定性
    'pin_memory': True,                 # 啟用固定記憶體以加速數據傳輸
}

# --- OPTIMIZER CONFIG ---
OPTIMIZER_CONFIG = {
    'learning_rate': 0.00001,           # 降低學習率以提高穩定性
    'adam_betas': (0.5, 0.999),         # Adam優化器的beta參數
    'adam_eps': 1e-8,                   # Adam優化器的epsilon值(數值穩定性)
    'scheduler_t_max': 100,             # 餘弦退火調度器週期
    'weight_decay': 1e-4,               # L2正則化係數
    'warmup_epochs': 2,                 # 熱身訓練期的回合數
}

# --- LOSS WEIGHTS ---
LOSS_WEIGHTS = {
    'content_loss': 1.0,                # 內容損失權重
    'perceptual_loss': 0.04,            # 感知損失權重
    'dehaze_loss': 10.0,                # 去霧損失權重
    'dc_loss': 0.01,                    # 暗通道損失權重
    'tv_loss': 2e-7,                    # 全變分損失權重
    'cap_loss': 0.001,                  # CAP損失權重
    'lab_loss': 0.0001,                 # Lab顏色空間損失權重
    'recover_loss': 1.0,                # 恢復損失權重
    'mask_loss': 1.0,                   # 遮罩損失權重
    'haze_loss': 1.0,                   # 霧化損失權重
    'cycle_loss': 1.0,                  # 循環一致性損失權重
    'identity_loss': 1.0,               # 身份損失權重
}

# --- MODEL ARCHITECTURE PARAMETERS ---
MODEL_CONFIG = {
    'ffa_groups': 3,                    # FFA模型的分組數
    'ffa_blocks': 5,                    # FFA模型的區塊數
    'init_type': 'kaiming',             # 權重初始化方法
    'init_gain': 0.02,                  # 初始化縮放因子
}

# --- DATA PROCESSING PARAMETERS ---
DATA_CONFIG = {
    'crop_size': 128,                   # 訓練用圖像裁剪大小
    'normalize_range': (-1, 1),         # 輸入標準化範圍
    'num_workers': 1,                   # 數據加載器的工作進程數
    'pin_memory': True,                 # 啟用固定記憶體以加速傳輸
}

def apply_training_configs():
    # 設置記憶體和CUDA配置
    torch.backends.cuda.max_memory_allocated = MEMORY_CONFIG['max_gpu_memory']
    torch.backends.cudnn.deterministic = MEMORY_CONFIG['deterministic']
    torch.backends.cudnn.benchmark = MEMORY_CONFIG['enable_cudnn_benchmark']
    
    # 啟用自動梯度異常檢測
    if STABILITY_CONFIG['enable_anomaly_detection']:
        torch.autograd.set_detect_anomaly(True)
    
    # 創建參數解析器並設置優化後的默認值
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=TOTAL_EPOCHS)
    parser.add_argument('--batchSize', type=int, default=MEMORY_CONFIG['batch_size'])
    parser.add_argument('--lr', type=float, default=OPTIMIZER_CONFIG['learning_rate'])
    parser.add_argument('--input_nc', type=int, default=3)
    parser.add_argument('--output_nc', type=int, default=3)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--n_cpu', type=int, default=DATA_CONFIG['num_workers'])
    
    return parser.parse_args()

# Utility function for safe tensor operations
class SafeOps:
    def safe_tensor_ops(self, tensor, name=""):
        if tensor is None:
            return None
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"Warning: Non-finite values in {name}, clamping")
            tensor = torch.clamp(tensor, \
                    STABILITY_CONFIG['tensor_value_clip'][0], \
                    STABILITY_CONFIG['tensor_value_clip'][1])
        return tensor

def get_loss_name(loss_fn):
    """Get a readable name for the loss function or class."""
    if hasattr(loss_fn, '__name__'):
        return loss_fn.__name__
    return loss_fn.__class__.__name__

def compute_loss_safely(loss_fn, *args, **kwargs):
    try:
        loss = loss_fn(*args, **kwargs)
        if not torch.isfinite(loss).all():
            print(f"Non-finite loss in {get_loss_name(loss_fn)}")
            return torch.tensor(0.0, requires_grad=True).cuda()
        return loss
    except Exception as e:
        print(f"Error in {get_loss_name(loss_fn)}: {str(e)}")
        return torch.tensor(0.0, requires_grad=True).cuda()

def safe_clamp_tuple(tuple_tensor, name="", min=-1e8, max=1e8):
    """
    Safely clamp tuple of tensors, handling None values and providing warnings
    Returns tuple of clamped tensors
    """
    if not isinstance(tuple_tensor, tuple):
        return tuple_tensor
    result = []
    for i, tensor in enumerate(tuple_tensor):
        if isinstance(tensor, torch.Tensor):
            if not torch.isfinite(tensor).all():
                print(f"Warning: Non-finite values detected in {name}[{i}]. Clamping values.")
            result.append(torch.clamp(tensor, \
                    STABILITY_CONFIG['tensor_value_clip'][0], \
                    STABILITY_CONFIG['tensor_value_clip'][1]))  # min=-1.0, max=1.0
        else:
            result.append(tensor)
    return tuple(result)

def check_tensor(tensor, name):
    if tensor is None:
        return False
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"NaN or Inf detected in {name}")
        return False
    print(f"{name} shape: {tensor.shape}, min: {tensor.min().item()}, max: {tensor.max().item()}, mean: {tensor.mean().item()}")
    return True

opt = apply_training_configs()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

torch.autograd.set_detect_anomaly(True) # enable to detect an error
# JanYeh: Set smaller learning rate
opt.lr = 0.00001  # Reduced from 0.0001

###### Definition of variables ######
# Networks
netG_content=Net_content()
netG_haze = Net_hazy()
net_dehaze  = ffa(MODEL_CONFIG['ffa_groups'], MODEL_CONFIG['ffa_blocks']) # ffa(3,5)
net_G= Net_G()

# JanYeh: Initialize weights properly
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0.01)

for net in [netG_content, netG_haze, net_dehaze, net_G]:
    net.apply(init_weights)

netG_content.cuda()
netG_haze.cuda()
net_dehaze.cuda()
net_G.cuda()


# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.cuda()
for param in vgg_model.parameters():
     param.requires_grad = False

loss_network = LossNetwork(vgg_model).cuda()
loss_network.eval()


optimizer_G= torch.optim.Adam(itertools.chain(netG_content.parameters() ,net_dehaze.parameters(),netG_haze.parameters(),net_G.parameters()), \
        lr=opt.lr, betas=OPTIMIZER_CONFIG['adam_betas'], eps=OPTIMIZER_CONFIG['adam_eps'], \
        weight_decay=OPTIMIZER_CONFIG['weight_decay'])
        #lr=opt.lr, betas=(0.5, 0.999), eps=1e-8)

lr_scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=OPTIMIZER_CONFIG['scheduler_t_max']) # torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=100)
dataloader1 = DataLoader(TrainDatasetFromFolder2('trainset/trainA_new', \
        'trainset/trainB_new',  'trainset/trainB_newsize_128', \
        crop_size=DATA_CONFIG['crop_size']),
        batch_size=MEMORY_CONFIG['batch_size'],
        shuffle=True,
        pin_memory=DATA_CONFIG['pin_memory']
        )
        #crop_size= 128), batch_size=opt.batchSize,shuffle=True )  #SIDMS   /home/omnisky/volume/ITSV2/clear
# dataloader2 = DataLoader(TrainDatasetFromFolder4('/home/omnisky/4t/RESIDE/OTS_BETA/clear/clear_newsize',
#                                              '/home/omnisky/4t/RESIDE/OTS_BETA/haze/hazy7',  '/home/omnisky/4t/realWorldHazeDataSet/trainA_newsize_128', crop_size=128), batch_size=opt.batchSize,shuffle=True )  #SIDMS   /home/omnisky/volume/ITSV2/clear


val_data_loader = DataLoader(TestDatasetFromFolder1('testdataset'), \
        batch_size=MEMORY_CONFIG['batch_size'], shuffle=False, num_workers=DATA_CONFIG['num_workers'], \
        pin_memory=DATA_CONFIG['pin_memory']) # batch_size=opt.batchSize, num_workers=opt.n_cpu

logger1 = Logger(opt.n_epochs, len(dataloader1))
# logger2 = Logger(opt.n_epochs, len(dataloader1))
###################################
if not os.path.exists('output'):
    os.makedirs('output')
if not os.path.exists('./results'):
    os.makedirs('./results/Inputs')
    os.makedirs('./results/Outputs')
    os.makedirs('./results/Targets')
###### Training ######
safe_ops = SafeOps()
for epoch in range(opt.epoch, opt.n_epochs):

    # if not epoch%2 :
    #     dataloader = dataloader1
    # else :
    #     dataloader = dataloader2
    
    dataloader = dataloader1
    

    ite = 0
    adjust_learning_rate(optimizer_G, epoch)

    # Jan - debug 
    max_debug_iterations = 10

    # Add memory debug info
    torch.backends.cudnn.benchmark = True
    print(f"Initial memory allocated: {torch.cuda.memory_allocated()} bytes")
    print(f"Initial max memory allocated: {torch.cuda.max_memory_allocated()} bytes")
    
    # JanYeh: Safe loss calculation

    # In training loop
    for i, batch in enumerate(dataloader):
        # Jan - debug for shorter training
        if i >= max_debug_iterations:
            break

        # Set model input
        real_A = Variable(batch['A']).cuda(0)#clear
        real_B = Variable(batch['B']).cuda(0)
        real_R = Variable(batch['R']).cuda(0)

        if real_A.size(1) == 3 and real_B.size(1) == 3:
        
            check_tensor(real_A, "real_A")
            check_tensor(real_B, "real_B")
            check_tensor(real_R, "real_R")

            ite += 1
            optimizer_G.zero_grad()
            content_B,con_B = safe_ops.safe_tensor_ops(netG_content(real_B), "netG_content(real_B)")
            haze_mask_B,mask_B = safe_clamp_tuple(netG_haze(real_B), "netG_haze(real_B)")

            content_R,con_R = safe_ops.safe_tensor_ops(netG_content(real_R), "netG_content(real_R)")
            haze_mask_R,mask_R = safe_clamp_tuple(netG_haze(real_R), "netG_haze(real_R)")

            recover_R = safe_ops.safe_tensor_ops(net_G(content_R, haze_mask_R), "net_G(content_R, haze_mask_R)")
            recover_B = safe_ops.safe_tensor_ops(net_G(content_B, haze_mask_B), "net_G(content_B, haze_mask_B)")

            meta_B = cat([con_B,mask_B],1)
            meta_R = cat([con_R, mask_R], 1)

            dehaze_B = safe_ops.safe_tensor_ops(net_dehaze(real_B, meta_B), "net_dehaze(real_B, meta_B)")
            dehaze_R = safe_ops.safe_tensor_ops(net_dehaze(real_R, meta_R), "net_dehaze(real_R, meta_R)")

            # Check tensor shapes and channels
            print(f"dehaze_R shape before content: {dehaze_R.shape}")
            # Ensure dehaze_R has correct number of channels (3) before passing to netG_content
            if dehaze_R.size(1) != 3:
                print(f"Warning: Incorrect channel count in dehaze_R: {dehaze_R.size(1)}, reshaping...")
                dehaze_R = dehaze_R[:,:3,:,:]
            if dehaze_B.size(1) != 3:
                print(f"Warning: Incorrect channel count in dehaze_B: {dehaze_B.size(1)}, reshaping...")
                dehaze_B = dehaze_B[:,:3,:,:]

            content_dehaze_R,con_dehaze_R = safe_clamp_tuple(netG_content(dehaze_R), "netG_content(dehaze_R)")
            content_dehaze_B,con_dehaze_B = safe_clamp_tuple(netG_content(dehaze_B), "netG_content(dehaze_B)")

            haze_mask_dehaze_R,mask_dehaze_B = safe_clamp_tuple(netG_haze(dehaze_R), "netG_haze(dehaze_R)")
            haze_mask_dehaze_B,mask_dehaze_R = safe_clamp_tuple(netG_haze(dehaze_B), "netG_haze(dehaze_B)")


            recover_dehaze_R = safe_ops.safe_tensor_ops(net_G(content_dehaze_R, haze_mask_dehaze_R), "net_G(content_dehaze_R, haze_mask_dehaze_R)")
            recover_dehaze_B = safe_ops.safe_tensor_ops(net_G(content_dehaze_B, haze_mask_dehaze_B), "net_G(content_dehaze_B, haze_mask_dehaze_B)")

        
            content_A ,con_A= safe_clamp_tuple(netG_content(real_A), "netG_content(real_A)")
            haze_mask_A,mask_A = safe_clamp_tuple(netG_haze(real_A), "netG_haze(real_A)")

            meta_A = cat([con_A, mask_A],1)
 
            # JanYeh: Check for NaN or Inf before passing to net_G
            if not torch.isfinite(content_A).all() or not torch.isfinite(haze_mask_B).all():
               print("NaN or Inf detected in content_A or haze_mask_B, skipping iteration")
               continue

            dehaze_A = safe_ops.safe_tensor_ops(net_dehaze(real_A, meta_A ), "net_dehaze(real_A, meta_A )")
            recover_A = safe_ops.safe_tensor_ops(net_G(content_A, haze_mask_A), "net_G(content_A, haze_mask_A)")

            fake_hazy_A = safe_ops.safe_tensor_ops(net_G(content_A,haze_mask_B), "net_G(content_A,haze_mask_B)")
            # JanYeh: Check for NaN or Inf after computing fake_hazy_A
            if not torch.isfinite(fake_hazy_A).all():
                print("NaN or Inf detected in fake_hazy_A, skipping iteration")
                continue
            check_tensor(fake_hazy_A, "fake_hazy_A")

            content_fake_hazy_A, con_fake_hazy_A  = safe_clamp_tuple(netG_content(fake_hazy_A ), "netG_content(fake_hazy_A )")
            haze_mask_fake_hazy_A,mask_fake_hazy_A = safe_clamp_tuple(netG_haze(fake_hazy_A ), "netG_haze(fake_hazy_A )")

            meta_fake_hazy_A = torch.clamp(cat([con_fake_hazy_A,mask_fake_hazy_A],1), min=-1.0, max=1.0)
            # Jan - debug BEGIN
            if not torch.isfinite(fake_hazy_A).all() or not torch.isfinite(meta_fake_hazy_A).all():
                print("NaN or Inf detected in fake_hazy_A or meta_fake_hazy_A, skipping iteration")
                continue
            if not check_tensor(fake_hazy_A, "fake_hazy_A") or not check_tensor(meta_fake_hazy_A, "meta_fake_hazy_A"):
                print("Skipping iteration due to NaN or Inf")
                continue
            # Jan - debug END

            # print(f"fake_hazy_A shape: {fake_hazy_A.shape}")
            # print(f"meta_fake_hazy_A shape: {meta_fake_hazy_A.shape}")
            # print(f"content_A shape: {content_A.shape}")
            # print(f"haze_mask_B shape: {haze_mask_B.shape}")
            # print(f"content_fake_hazy_A shape: {content_fake_hazy_A.shape}")
            # print(f"con_fake_hazy_A shape: {con_fake_hazy_A.shape}")
            # print(f"mask_fake_hazy_A shape: {mask_fake_hazy_A.shape}")
            try:
                dehaze_fake_hazy_A = safe_ops.safe_tensor_ops(net_dehaze(fake_hazy_A, meta_fake_hazy_A ), "net_dehaze(fake_hazy_A, meta_fake_hazy_A )")
            except Exception as e:
                print(f"Error in net_dehaze: {e}")
                continue
            check_tensor(meta_fake_hazy_A, "meta_fake_hazy_A")
        
            loss_components = []

            # JanYeh: Replace direct loss calculations with safe versions
            loss_haze = F.smooth_l1_loss(fake_hazy_A, real_B)
            loss_components = []
            loss_haze = compute_loss_safely(F.smooth_l1_loss, fake_hazy_A, real_B)
            if check_tensor(loss_haze, "loss_haze"):
                loss_components.append(loss_haze)
            # Add gradient clipping
            max_grad_norm = STABILITY_CONFIG['gradient_clip_norm'] # 1.0
            torch.nn.utils.clip_grad_norm_(
                itertools.chain(netG_content.parameters(), \
                        netG_haze.parameters(), \
                        net_dehaze.parameters(), \
                        net_G.parameters()), \
                        max_grad_norm
            ) # max_grad_norm was 1.0
            # JanYeh: End of safe loss calculations

            # loss_haze =  F.smooth_l1_loss(fake_hazy_A , real_B)  + loss_network(fake_hazy_A , real_B) * 0.04
            loss_haze = compute_loss_safely(F.smooth_l1_loss, fake_hazy_A , real_B)  + compute_loss_safely(loss_network, fake_hazy_A , real_B) * LOSS_WEIGHTS['perceptual_loss'] #0.04
            # Jan - debug
            if check_tensor(loss_haze, "loss_haze"):
                loss_components.append(loss_haze)

            # loss_dehaze = F.smooth_l1_loss(dehaze_B, real_A)  + loss_network(dehaze_B, real_A) * 0.04 \
            #               + F.smooth_l1_loss(dehaze_A, real_A)  + loss_network(dehaze_A, real_A) * 0.04 \
            #                + F.smooth_l1_loss( dehaze_fake_hazy_A, real_A) + loss_network( dehaze_fake_hazy_A, real_A) * 0.04\
            loss_dehaze = compute_loss_safely(F.smooth_l1_loss, dehaze_B, real_A)  + compute_loss_safely(loss_network, dehaze_B, real_A) * 0.04 \
                            + compute_loss_safely(F.smooth_l1_loss, dehaze_A, real_A)  + compute_loss_safely(loss_network, dehaze_A, real_A) * 0.04 \
                            + compute_loss_safely(F.smooth_l1_loss, dehaze_fake_hazy_A, real_A) + compute_loss_safely(loss_network, dehaze_fake_hazy_A, real_A) * 0.04\
            # Jan - debug
            if check_tensor(loss_dehaze, "loss_dehaze"):
                loss_components.append(loss_dehaze)
            
            # loss_content = F.smooth_l1_loss(content_A, content_B)  + F.smooth_l1_loss(content_dehaze_B, content_B) + F.smooth_l1_loss(content_dehaze_R, content_R)\
            #                 +F.smooth_l1_loss(content_fake_hazy_A ,content_A) # +F.smooth_l1_loss(content_A,content_dehaze_fake_B)\
            loss_content = compute_loss_safely(F.smooth_l1_loss, content_A, content_B)  \
                            + compute_loss_safely(F.smooth_l1_loss, content_dehaze_B, content_B) + compute_loss_safely(F.smooth_l1_loss, content_dehaze_R, content_R)\
                            + compute_loss_safely(F.smooth_l1_loss, content_fake_hazy_A ,content_A) # +F.smooth_l1_loss(content_A,content_dehaze_fake_B)\
            # Jan - debug
            if check_tensor(loss_content, "loss_content"):
                loss_components.append(loss_content)

            #loss_mask = F.smooth_l1_loss(haze_mask_dehaze_B, haze_mask_A) + F.smooth_l1_loss(haze_mask_fake_hazy_A , haze_mask_B)#+ F.smooth_l1_loss(haze_mask_fake_hazy_A, haze_mask_A)
            loss_mask = compute_loss_safely(F.smooth_l1_loss, haze_mask_dehaze_B, haze_mask_A) \
                        + compute_loss_safely(F.smooth_l1_loss, haze_mask_fake_hazy_A, haze_mask_B)#+ F.smooth_l1_loss(haze_mask_fake_hazy_A, haze_mask_A)
            # Jan - debug
            if check_tensor(loss_mask, "loss_mask"):
                loss_components.append(loss_mask)


            # loss_recover = F.smooth_l1_loss(recover_B, real_B) + loss_network(recover_B, real_B) * 0.04 + \
            #                F.smooth_l1_loss(recover_A, real_A) + loss_network(recover_A, real_A) * 0.04 + \
            #                F.smooth_l1_loss(recover_R, real_R) + loss_network(recover_R, real_R) * 0.04 + \
            #                F.smooth_l1_loss(recover_dehaze_B, real_A) + loss_network(recover_dehaze_B, real_A) * 0.04

            loss_recover = compute_loss_safely(F.smooth_l1_loss, recover_B, real_B) + compute_loss_safely(loss_network, recover_B, real_B) * 0.04 + \
                            compute_loss_safely(F.smooth_l1_loss, recover_A, real_A) + compute_loss_safely(loss_network, recover_A, real_A) * 0.04 + \
                            compute_loss_safely(F.smooth_l1_loss, recover_R, real_R) + compute_loss_safely(loss_network, recover_R, real_R) * 0.04 + \
                            compute_loss_safely(F.smooth_l1_loss, recover_dehaze_B, real_A) + compute_loss_safely(loss_network, recover_dehaze_B, real_A) * 0.04
            # Jan - debug
            if check_tensor(loss_recover, "loss_recover"):
                loss_components.append(loss_recover)            


            y = dehaze_R
            z = dehaze_B
            tv_loss = (torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) +
                    torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))+ \
                    (torch.sum(torch.abs(z[:, :, :, :-1] - z[:, :, :, 1:])) +
                    torch.sum(torch.abs(z[:, :, :-1, :] - z[:, :, 1:, :])))

            loss_DC_A = DCLoss((dehaze_R + 1) / 2, 16) + DCLoss((dehaze_B + 1) / 2, 16)  + DCLoss((dehaze_A  + 1) / 2, 16) + DCLoss((dehaze_fake_hazy_A  + 1) / 2, 16)
            loss_CAP = CAPLoss(dehaze_R)+CAPLoss(dehaze_B) + CAPLoss(dehaze_A) + CAPLoss(dehaze_fake_hazy_A)
            loss_Lab = LabLoss(dehaze_R, real_R)*0.01+LabLoss(dehaze_B,real_A)+LabLoss(dehaze_fake_hazy_A ,real_A)
            loss_Lab = loss_Lab.float()

            # Scale large losses before combining
            if loss_recover > 1000:
                print(f"Scaling down large loss_recover: {loss_recover}")
                loss_recover = torch.clamp(loss_recover, max=1000)
            # Similar checks for other large losses

            # Total loss
            # Add the remaining loss components
            #loss_components.extend([10*loss_dehaze, 0.01 * loss_DC_A, 2*1e-7*tv_loss, 0.001 *loss_CAP, 0.0001*loss_Lab])
            loss_components.extend([ \
                    LOSS_WEIGHTS['dehaze_loss'] * loss_dehaze, \
                    LOSS_WEIGHTS['dc_loss'] * loss_DC_A, \
                    LOSS_WEIGHTS['tv_loss'] * tv_loss, \
                    LOSS_WEIGHTS['cap_loss'] * loss_CAP, \
                    LOSS_WEIGHTS['lab_loss'] * loss_Lab \
            ])
            # Jan - debug BEGIN
            if loss_components:
                loss_G = sum(loss_components)
                if check_tensor(loss_G, "loss_G"):

                    # Scale down large loss values
                    if loss_G.item() > STABILITY_CONFIG['loss_scale_threshold']: # 100:
                        # scale_factor = min(100.0 / loss_G.item(), 0.1)  # Cap scaling factor
                        # loss_G = loss_G * scale_factor
                        #scale_factor = min(100.0 / loss_G.item(), 0.01)  # More aggressive scaling
                        scale_factor = min(STABILITY_CONFIG['loss_scale_threshold'] / loss_G.item(), \
                                STABILITY_CONFIG['loss_scale_factor'])
                        loss_G = loss_G * scale_factor 

                        print(f"Scaling loss by factor {scale_factor}")
                    
                    try:
                        # Do backward pass and optimization
                        loss_G.backward()
                        # Clip gradients before optimizer step
                        torch.nn.utils.clip_grad_norm_(
                            itertools.chain(
                                netG_content.parameters(),
                                netG_haze.parameters(), 
                                net_dehaze.parameters(),
                                net_G.parameters()
                            ),
                            max_norm=1.0
                        )
                        # Check and clamp any remaining bad gradients
                        for name, param in itertools.chain(
                            netG_content.named_parameters(),
                            netG_haze.named_parameters(),
                            net_dehaze.named_parameters(),
                            net_G.named_parameters()):
                            if param.grad is not None:
                                param.grad.data.clamp_(-1, 1)
                        optimizer_G.step()
                        print("Optimization step completed")
                    except Exception as e:
                        print(f"Error in backward pass: {e}")
                        continue
                    
                else:
                    print("Skipping backward pass due to invalid loss_G")
            else:
                print("No valid loss components to compute loss_G")
            # Jan - debug END

            ###################################

            # if not epoch % 2:
            #     logger = logger1
            # else :
            #     logger = logger2
            
            logger = logger1
            logger.log({'loss_G': loss_G,  'loss_recover': loss_recover,  'loss_content': loss_content, 'loss_mask': loss_mask, 'loss_haze': loss_haze, 'loss_dehaze': loss_dehaze,'tv_loss': tv_loss,'loss_DC_A': loss_DC_A, 'loss_CAP': loss_CAP, 'loss_Lab': loss_Lab})
            if ite % 1000 == 0:

                vutils.save_image(recover_R.data, './recover_R.png' , normalize=True)
                vutils.save_image(recover_B.data, './recover_B.png', normalize=True)

            if ite % 100 == 0:
                vutils.save_image(real_A.data, './real_A.png' , normalize=True)
                vutils.save_image(real_B.data, './real_B.png', normalize=True)
                vutils.save_image(real_R.data, './real_R.png', normalize=True)
                vutils.save_image(dehaze_B.data, './dehaze_B.png', normalize=True)
                vutils.save_image(dehaze_R.data, './dehaze_R.png' , normalize=True)
            

    # Update learning rates

    lr_scheduler_G.step()


    torch.save(netG_content.state_dict(), 'output/netG_content_%d.pth' % int(epoch+1))
    torch.save(netG_haze.state_dict(), 'output/netG_haze_%d.pth' % int(epoch+1))
    torch.save(net_dehaze.state_dict(), 'output/net_dehaze_%d.pth' % int(epoch + 1))
    torch.save(net_G.state_dict(), 'output/net_G_%d.pth' % int(epoch + 1))

    if epoch % 1 == 0:
        with torch.no_grad():
            print('------------------------')
            test_psnr = 0
            test_ssim = 0
            eps = 1e-10
            test_ite = 0
            # for image_name,input, target in enumerate(self.val_loader):
            for i, batch in enumerate(val_data_loader):
                # Set model input
                if 'A' not in batch or 'B' not in batch:
                    print(f"Warning: Incomplete batch at index {i}, skipping")
                    continue
                if batch['A'] is None or batch['B'] is None:
                    print(f"Warning: None values in batch at index {i}, skipping")
                    continue

                real_A = Variable(batch['A']).cuda(0)  # clear
                real_B = Variable(batch['B']).cuda(0)


                content_B,con_B= netG_content(real_B)
                hazy_mask_B ,mask_B= netG_haze(real_B)

                meta_B = cat([con_B,mask_B],1)
                dehaze_B = net_dehaze(real_B, meta_B)

                # JanYeh: Check for None before saving images BEGIN
                if real_A is None:
                    real_A = torch.zeros_like(real_A)
                    print(f"ERROR: real_A is None, set to zero. Path=" + f"{'./results/Targets/%05d.png' % (int(i))}") 
                vutils.save_image(real_A.data, './results/Targets/%05d.png' % (int(i)), padding=0, normalize=True)  # False

                if real_B is None:
                    real_B = torch.zeros_like(real_B)
                    print(f"ERROR: real_B is None, set to zero. Path=" + f"{'./results/Inputs/%05d.png' % (int(i))}") 
                vutils.save_image(real_B.data, './results/Inputs/%05d.png' % (int(i)), padding=0, normalize=True)

                if dehaze_B is None:
                    dehaze_B = torch.zeros_like(dehaze_B)
                    print(f"ERROR: dehaze_B is None, set to zero. Path="+f"{'./results/Outputs/%05d.png' % (int(i))}") 
                vutils.save_image(dehaze_B.data, './results/Outputs/%05d.png' % (int(i)), padding=0, normalize=True)
                # JanYeh: Check for None before saving images END

                # Calculation of SSIM and PSNR values
                # print(output)
                output = dehaze_B.data.cpu().numpy()[0]
                output[output > 1] = 1
                output[output < 0] = 0
                output = output.transpose((1, 2, 0))
                hr_patch = real_A.data.cpu().numpy()[0]
                hr_patch[hr_patch > 1] = 1
                hr_patch[hr_patch < 0] = 0
                hr_patch = hr_patch.transpose((1, 2, 0))
                # SSIM
                test_ssim += ski_ssim(output, hr_patch, data_range=1, multichannel=True)
                # PSNR
                imdf = (output - hr_patch) ** 2
                mse = np.mean(imdf) + eps
                test_psnr += 10 * math.log10(1.0 / mse)
                test_ite += 1
            test_psnr /= (test_ite)
            test_ssim /= (test_ite)
            print('Valid PSNR: {:.4f}'.format(test_psnr))
            print('Valid SSIM: {:.4f}'.format(test_ssim))
            f = open('PSNR.txt', 'a')
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow([epoch, test_psnr, test_ssim])
            f.close()
            print('------------------------')

###################################
