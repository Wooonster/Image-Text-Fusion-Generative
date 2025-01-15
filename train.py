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
from datasets import load_dataset


# 数据集定义
class ImageCaptionDataset(Dataset):
    def __init__(self, raw_dataset):
        # dataframe = pd.read_parquet(pq_path)
        # self.dataframe = dataframe

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
    # pq_path = './train-00000-of-00010.parquet'
    # dataset = ImageCaptionDataset(pq_path)
    # dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    base_url = "https://huggingface.co/datasets/jackyhate/text-to-image-2M/resolve/main/data_512_2M/data_000000.tar"
    num_shards = 46  # Number of webdataset tar files
    urls = [base_url.format(i=i) for i in range(num_shards)]
    raw_dataset = load_dataset("webdataset", data_files={"train": urls}, split="train", streaming=True)
    dataset = ImageCaptionDataset(raw_dataset)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 加载模型并移动到设备
    model = TextImageFusionModel().to(DEVICE)

    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

    # 定义学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # 定义对比损失
    criterion_contrastive = contrastive_loss

    # 定义重建损失
    criterion_reconstruction = F.mse_loss  # 均方误差损失

    # 设置重建损失的权重
    lambda_recon = 0.8

    # 初始化损失记录列表
    loss_history = []

    # 确保路径存在
    os.makedirs('loss_log', exist_ok=True)
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 训练循环
    num_epochs = 30
    print('Start Training...')
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        for images, captions in dataloader:
            images = images.to(DEVICE)

            # 确保 captions 是字符串列表
            captions = [str(caption) for caption in captions]

            optimizer.zero_grad()
            outputs, img_emb, txt_emb = model(images, captions)

            # 确保嵌入在同一设备上
            img_emb = img_emb.to(DEVICE)
            txt_emb = txt_emb.to(DEVICE)

            # 计算对比损失
            loss_contrastive = criterion_contrastive(img_emb, txt_emb)

            # 反标准化
            mean = torch.tensor([0.485, 0.456, 0.406]).to(DEVICE).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).to(DEVICE).view(1, 3, 1, 1)
            images_denorm = torch.clamp(images * std + mean, 0, 1)
            outputs_denorm = torch.clamp(outputs * std + mean, 0, 1)

            loss_reconstruction = criterion_reconstruction(outputs_denorm, images_denorm)

            # 组合损失
            total_loss = loss_contrastive + lambda_recon * loss_reconstruction

            # 反向传播
            total_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += total_loss.item()

        # 学习率调度器步进
        epoch_loss = running_loss / len(dataloader)
        loss_history.append(epoch_loss)
        scheduler.step(epoch_loss)  # 如果使用 ReduceLROnPlateau

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        # 保存模型
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth"))

    # 绘制损失变化图
    plt.figure()
    plt.plot(range(1, num_epochs + 1), loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.savefig('loss_log/loss_plot_1_14.png')
    plt.show()