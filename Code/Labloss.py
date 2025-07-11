
from skimage import color
from torch.nn import L1Loss, MSELoss
from torchvision import transforms
import cv2


def LabLoss(dehaze, hazy):
    """
    calculating Lab color space loss, the image shape is of N*C*W*H
    修正批次處理問題 - 支援多批次輸入
    """
    import torch
    
    if dehaze.dim() == 4:  # 如果是批次輸入 (N, C, H, W)
        batch_size = dehaze.size(0)
        total_loss = 0
        
        for i in range(batch_size):
            try:
                # 處理每個批次中的單個圖像
                single_dehaze = dehaze[i:i+1].squeeze(0)  # 移除批次維度
                single_hazy = hazy[i:i+1].squeeze(0)    # 移除批次維度
                
                unloader = transforms.ToPILImage()
                
                # 轉換為 PIL 圖像
                dehaze_pil = unloader(single_dehaze.cpu().clone())
                hazy_pil = unloader(single_hazy.cpu().clone())
                
                # 轉換為 Lab 色彩空間
                lab_dehaze = color.rgb2lab(dehaze_pil)
                ldehaze, a, b = cv2.split(lab_dehaze)
                lab_hazy = color.rgb2lab(hazy_pil)
                lhazy, a, b = cv2.split(lab_hazy)
                
                # 轉換回張量
                totensor = transforms.ToTensor()
                ldehaze = totensor(ldehaze)
                lhazy = totensor(lhazy)
                
                # 計算損失
                l1loss = L1Loss()
                loss = l1loss(ldehaze, lhazy)
                total_loss += loss
                
            except Exception as e:
                # 如果出現錯誤，返回零損失
                total_loss += torch.tensor(0.0, requires_grad=True).cuda()
        
        return total_loss / batch_size
    
    else:  # 原始單圖像處理邏輯
        unloader = transforms.ToPILImage()
        image = dehaze.cpu().clone()
        image = image.squeeze(0)
        dehaze_pil = unloader(image)

        unloader = transforms.ToPILImage()
        image = hazy.cpu().clone()
        image = image.squeeze(0)
        hazy_pil = unloader(image)
        
        totensor = transforms.ToTensor()
        
        # 轉換為 Lab 色彩空間
        lab_dehaze = color.rgb2lab(dehaze_pil)
        ldehaze, a, b = cv2.split(lab_dehaze)
        lab_hazy = color.rgb2lab(hazy_pil)
        lhazy, a, b = cv2.split(lab_hazy)
        ldehaze = totensor(ldehaze)
        lhazy = totensor(lhazy)
        l1loss = L1Loss()
        loss = l1loss(ldehaze, lhazy)
        
        return loss

