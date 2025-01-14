import clip  # pip install git+https://github.com/openai/CLIP.git
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import io
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from clip.simple_tokenizer import SimpleTokenizer

# CLIP特征提取器
class CLIPFeatureExtract(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)  # 使用 CLIP 的 ViT-B/32 模型
        self.device = device
        self.context_length = 77  # CLIP 的上下文长度限制

    def truncate_texts(self, texts):
        """
        确保文本不会超过 CLIP 的最大上下文长度。
        """
        truncated_texts = []
        for text in texts:
            tokens = clip.tokenize([text], context_length=self.context_length).cpu().numpy()
            # 如果生成的 tokens 超过最大长度，截取前 context_length - 2 并加上特殊标记
            if len(tokens[0]) > self.context_length:
                truncated_text = text[:self.context_length - 2]  # 保留一些余地
                truncated_texts.append(truncated_text)
            else:
                truncated_texts.append(text)
        return truncated_texts

    def forward(self, images, texts):
        """
        images: 预处理后的图像张量，形状为 [batch_size, 3, 224, 224]
        texts: 文本列表，长度为 batch_size
        """
        # 裁剪文本
        texts = self.truncate_texts(texts)

        # 图像特征
        image_features = self.model.encode_image(images)  # [batch_size, 512]

        # 文本特征
        text_tokens = clip.tokenize(texts).to(self.device)  # 使用 CLIP 的 tokenizer
        text_features = self.model.encode_text(text_tokens)  # [batch_size, 512]

        # 返回 L2 归一化后的特征
        return F.normalize(image_features, p=2, dim=1), F.normalize(text_features, p=2, dim=1)
    
# 映射网络
class MappingNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.map = nn.Sequential(
            nn.Linear(input_dim, 1024),  # 增加隐藏层
            nn.ReLU(),
            nn.Linear(1024, output_dim),  # 输出为目标维度
        )
    
    def forward(self, x):
        x = self.map(x)
        return F.normalize(x, p=2, dim=1)


# 双路径编码器
class TwoPathEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义卷积层
        self.cnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ) for in_channels, out_channels in zip([1024, 32, 64, 128, 256], [32, 64, 128, 256, 512])
        ])
        
        # 定义用于调整 pooled 通道数的 1x1 卷积层
        self.pool_conv_layers = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels, out_channels in zip([1024, 32, 64, 128, 256], [32, 64, 128, 256, 512])
        ])

    def forward(self, x):
        pooled = x
        for layer, pool_conv in zip(self.cnn_layers, self.pool_conv_layers):
            cnn_out = layer(x)  # 输出形状 [batch, out_channels, h, w]
            # 使用自适应池化，使 pooled 的空间尺寸与 cnn_out 相同
            pooled = F.adaptive_max_pool2d(pooled, output_size=cnn_out.shape[2:])
            pooled = pool_conv(pooled)  # 调整通道数，输出形状 [batch, out_channels, h, w]
            x = cnn_out + pooled  # 相加，确保形状匹配
        return x


# 解码器
class Decoder(nn.Module):
    def __init__(self, encoded_dim=512, output_channels=3):
        super().__init__()
        self.fc = nn.Linear(1024, 7 * 7 * encoded_dim)
        self.deconv_layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ) for in_channels, out_channels in zip([512, 256, 128, 64, 32], [256, 128, 64, 32, 16])
        ])
        self.last_conv = nn.Conv2d(16, output_channels, kernel_size=1)
    
    def forward(self, z, encoded):
        z = self.fc(z).view(-1, 512, 7, 7)
        x = z + encoded
        for layer in self.deconv_layers:
            x = layer(x)
        x = self.last_conv(x)
        return x


# 融合模型
class TextImageFusionModel(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.clip_extract = CLIPFeatureExtract(device=device)  # 使用 CLIP 提取器
        self.img_map = MappingNetwork(512, 1024)  # CLIP 图像特征的输出维度是 512
        self.txt_map = MappingNetwork(512, 1024)  # CLIP 文本特征的输出维度是 512
        self.fusion_to_spatial = nn.Linear(1024, 1024 * 7 * 7)
        self.encoder = TwoPathEncoder()
        self.decoder = Decoder()

    def forward(self, images, texts):
        # Step 1: 使用 CLIP 提取图像和文本特征
        img_fea, txt_fea = self.clip_extract(images, texts)  # [batch_size, 512]

        # Step 2: 映射特征到共享空间
        img_mapped = self.img_map(img_fea)  # [batch_size, 1024]
        txt_mapped = self.txt_map(txt_fea)  # [batch_size, 1024]

        # Step 3: 融合特征
        fused = img_mapped + txt_mapped  # [batch_size, 1024]

        # Step 4: 映射到空间维度
        fused_spatial = self.fusion_to_spatial(fused)
        fused_spatial = fused_spatial.view(fused.size(0), 1024, 7, 7)

        # Step 5: 编码和解码
        encoded = self.encoder(fused_spatial)
        output = self.decoder(fused, encoded)

        return output, img_mapped, txt_mapped


# 对比损失
def contrastive_loss(img_emb, txt_emb, temperature=0.07):
    similarity_matrix = torch.matmul(img_emb, txt_emb.T) / temperature  # 添加温度缩放
    labels = torch.arange(similarity_matrix.size(0)).to(img_emb.device)
    img_loss = F.cross_entropy(similarity_matrix, labels)
    txt_loss = F.cross_entropy(similarity_matrix.T, labels)
    return (img_loss + txt_loss) / 2


# 数据集定义
class ImageCaptionDataset(Dataset):
    def __init__(self, pq_path, clip_preprocess):
        dataframe = pd.read_parquet(pq_path)
        self.dataframe = dataframe
        self.clip_preprocess = clip_preprocess

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_data = self.dataframe.iloc[idx]['image']['bytes']
        caption = self.dataframe.iloc[idx]['caption']
        image = Image.open(io.BytesIO(img_data))

        # 检查图像数据是否有效
        if np.array(image).sum() == 0:
            raise ValueError(f"Image array at index {idx} is all zeros.")
        
        image = self.clip_preprocess(image)
        return image, caption


# 主程序
if __name__ == '__main__':
    # 设置设备
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    # 加载 CLIP 模型
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)

    # 加载数据
    pq_path = './train-00000-of-00010.parquet'
    dataset = ImageCaptionDataset(pq_path, clip_preprocess)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 加载模型并移动到设备
    model = TextImageFusionModel(device=DEVICE).to(DEVICE)

    # 定义优化器和损失
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion_contrastive = contrastive_loss
    criterion_reconstruction = F.mse_loss

    # 设置重建损失的权重
    lambda_recon = 0.5

    # 初始化损失记录
    loss_history = []

    # 训练循环
    num_epochs = 20
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        for images, captions in dataloader:
            images = images.to(DEVICE)
            captions = [str(caption) for caption in captions]

            optimizer.zero_grad()
            outputs, img_emb, txt_emb = model(images, captions)

            # 计算损失
            loss_contrastive = criterion_contrastive(img_emb, txt_emb)
            loss_reconstruction = criterion_reconstruction(outputs, images)

            total_loss = loss_contrastive + lambda_recon * loss_reconstruction
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        # 学习率调度器步进
        scheduler.step()

        epoch_loss = running_loss / len(dataloader)
        loss_history.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    # 绘制损失变化图
    plt.figure()
    plt.plot(range(1, num_epochs + 1), loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.show()