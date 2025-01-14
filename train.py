# train.py

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from model import TextImageFusionModel, contrastive_loss
from PIL import Image
import io
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# 数据集定义
class ImageCaptionDataset(Dataset):
    def __init__(self, pq_path):
        dataframe = pd.read_parquet(pq_path)
        self.dataframe = dataframe
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_data = self.dataframe.iloc[idx]['image']['bytes']
        caption = self.dataframe.iloc[idx]['caption']
        image = Image.open(io.BytesIO(img_data))

        # 检查图像数据是否有效
        if np.array(image).sum() == 0:
            raise ValueError(f"Image array at index {idx} is all zeros.")
        
        image = self.transform(image)
        return image, caption

if __name__ == '__main__':
    # 设置设备
    DEVICE = 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    # 加载数据
    pq_path = './train-00000-of-00010.parquet'
    dataset = ImageCaptionDataset(pq_path)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 加载模型并移动到设备
    model = TextImageFusionModel().to(DEVICE)

    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 定义学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # 定义对比损失
    criterion_contrastive = contrastive_loss

    # 定义重建损失
    criterion_reconstruction = F.mse_loss  # 均方误差损失

    # 设置重建损失的权重
    lambda_recon = 0.5  # 您可以根据需要调整这个值

    # 初始化损失记录列表
    loss_history = [0] * 30
    
    # 训练循环
    num_epochs = 30
    print('Start Training..., total epochs: ', num_epochs)
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        for images, captions in dataloader:
            images = images.to(DEVICE)
            # 确保 captions 是字符串列表
            captions = list(captions)  # 如果 captions 是 pandas Series
            captions = [str(caption) for caption in captions]  # 确保每个 caption 是字符串

            optimizer.zero_grad()
            outputs, img_emb, txt_emb = model(images, captions)  # 获取输出及嵌入

            # 确保嵌入在同一设备上
            img_emb = img_emb.to(DEVICE)
            txt_emb = txt_emb.to(DEVICE)

            # 计算对比损失
            loss_contrastive = criterion_contrastive(img_emb, txt_emb)

            # 计算重建损失
            # 需要对输出和输入图像进行反标准化，以便计算真实的重建误差
            # 反标准化公式: x = x * std + mean
            mean = torch.tensor([0.485, 0.456, 0.406]).to(DEVICE).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).to(DEVICE).view(1, 3, 1, 1)
            images_denorm = images * std + mean
            outputs_denorm = outputs * std + mean

            loss_reconstruction = criterion_reconstruction(outputs_denorm, images_denorm)

            # 组合损失
            total_loss = loss_contrastive + lambda_recon * loss_reconstruction

            # 反向传播
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        # 步进学习率调度器
        scheduler.step()

        epoch_loss = running_loss / len(dataloader)
        loss_history[epoch] = epoch_loss
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    # save checkpoints
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))

    # 绘制损失变化图
    plt.figure()
    plt.plot(range(1, num_epochs + 1), loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.show()