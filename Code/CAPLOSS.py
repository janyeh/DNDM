import torch
import numpy as np
from PIL import Image

import torch.nn as nn
from torch.nn import L1Loss, MSELoss
from torch.autograd import Variable
from torchvision import transforms
import pdb
import cv2


def CAPLoss(img):
    """
    calculating dark channel of image, the image shape is of N*C*W*H
    修正批次處理問題 - 處理多批次輸入
    """
    if img.dim() == 4:  # 如果是批次輸入 (N, C, H, W)
        batch_size = img.size(0)
        total_loss = 0
        
        for i in range(batch_size):
            # 處理每個批次中的單個圖像
            single_img = img[i:i+1]  # 保持 4 維但批次大小為 1
            single_img = single_img.squeeze(0)  # 現在安全地移除批次維度
            
            try:
                unloader = transforms.ToPILImage()
                image = single_img.cpu().clone()
                haze = unloader(image)
                x = cv2.cvtColor(np.asarray(haze), cv2.COLOR_RGB2BGR)

                HSV_img = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
                image = np.asarray(HSV_img)
                H, S, V = cv2.split(image)
                
                totensor = transforms.ToTensor()
                S = totensor(S)
                V = totensor(V)
                l1loss = L1Loss()
                loss = l1loss(S, V)
                total_loss += loss
            except Exception as e:
                # 如果出現錯誤，返回零損失
                total_loss += torch.tensor(0.0, requires_grad=True).cuda()
        
        return total_loss / batch_size
    
    else:  # 原始單圖像處理邏輯
        unloader = transforms.ToPILImage()
        image = img.cpu().clone()
        image = image.squeeze(0)
        haze = unloader(image)
        x = cv2.cvtColor(np.asarray(haze), cv2.COLOR_RGB2BGR)

        HSV_img = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
        image = np.asarray(HSV_img)
        H, S, V = cv2.split(image)
        
        totensor = transforms.ToTensor()
        S = totensor(S)
        V = totensor(V)
        l1loss = L1Loss()
        loss = l1loss(S, V)
        return loss