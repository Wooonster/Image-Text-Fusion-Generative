# infer.py

import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

from model import TextImageFusionModel
from train import contrastive_loss

# 如果要测试 GPU 或 MPS 的可用性
DEVICE = 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Inference device: {DEVICE}")

# ============== 1. 加载模型 ==============
model = TextImageFusionModel()
checkpoint_path = './checkpoints/model_epoch_30.pth'  # 你训练好的模型权重路径
model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ============== 2. 定义推理函数 ==============
def inference_single_image(model, image_path, text):
    """
    使用训练好的模型对单张图像 + 文本进行推理并返回重建图像结果。
    Args:
        model: 训练好的 TextImageFusionModel
        image_path: 待推理图像的路径
        text: 待推理文本（字符串）
    Returns:
        outputs_denorm: 重建后的反标准化图像（Tensor，形状 [1, 3, H, W]）
        img_emb: 图像特征向量
        txt_emb: 文本特征向量
    """
    # 让 pytorch 不计算梯度
    with torch.no_grad():
        # ============== 3. 预处理图像 ==============
        transform_infer = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # 与训练阶段保持一致的标准化参数
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # 读取并转换图像
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform_infer(image).unsqueeze(0)  # [1, 3, 224, 224]
        image_tensor = image_tensor.to(DEVICE)

        # ============== 4. 模型前向推断 ==============
        outputs, img_emb, txt_emb = model(image_tensor, [text])  # 文本传入为列表

        # ============== 5. 将重建图像反标准化，便于可视化或保存 ==============
        mean = torch.tensor([0.485, 0.456, 0.406]).to(DEVICE).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).to(DEVICE).view(1, 3, 1, 1)

        outputs_denorm = torch.clamp(outputs * std + mean, 0, 1)

    return outputs_denorm, img_emb, txt_emb

if __name__ == "__main__":
    # ============== 6. 推理示例 ==============
    test_image_path = "./image.png"    # 你的测试图像
    test_text = "The provided image is a cross-sectional view from a brain CT scan, displaying the cerebral hemispheres, ventricles, and various other intracranial structures. A region of interest, situated centrally in the lower-middle section of the image, exhibits abnormal density, indicating a pathological condition. This area's atypical appearance, distinct from the surrounding brain tissue, points to a possible intracranial hemorrhage. Its closeness to essential brain structures might suggest a potential mass effect or elevated intracranial pressure, potentially affecting the patient's neurological functions."  # 你的推理文本

    outputs_denorm, img_emb, txt_emb = inference_single_image(model, test_image_path, test_text)

    # 你可以将输出图像保存到本地，或者转成 numpy 显示
    # 以保存为例：
    recon_img = outputs_denorm.squeeze(0).cpu()  # [3, H, W]
    recon_img_pil = transforms.ToPILImage()(recon_img)
    recon_img_pil.save("reconstructed.jpg")
    
    print("推理完成，重建图像已保存为 reconstructed.jpg")
    print("图像 Embedding 向量形状: ", img_emb.shape)
    print("文本 Embedding 向量形状: ", txt_emb.shape)
