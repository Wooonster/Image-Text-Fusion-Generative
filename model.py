# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models import resnet50, ResNet50_Weights
from transformers import AutoTokenizer, AutoModel


# 冻结 ResNet 的特征提取器参数
class ImageFeatureExtract(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad = False  # 冻结预训练参数
        self.feature_extract = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.feature_extract(x)
        return x.view(x.size(0), -1)
   
# 冻结 BERT 的特征提取器参数
class TextFeatureExtract(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False  # 冻结预训练参数
    
    def forward(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(self.bert.device) for key, value in inputs.items()}  # 将输入移到 bert 所在设备
        outputs = self.bert(**inputs)
        last_hidden_state = outputs.last_hidden_state
        return last_hidden_state

# 修改映射网络
class MappingNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.map = nn.Sequential(
            nn.Linear(input_dim, 512),  # 增加隐藏层
            nn.ReLU(),
            nn.Linear(512, output_dim),  # 输出为目标维度
        )
    
    def forward(self, x):
        x = self.map(x)
        return F.normalize(x, p=2, dim=1)  # L2 归一化

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
    def __init__(self, encoded_dim=512, output_channels=3):  # 修改 output_channels 为 3
        super().__init__()
        self.fc = nn.Linear(1024, 7 * 7 * encoded_dim)  # 将融合向量映射到空间表示
        self.deconv_layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ) for in_channels, out_channels in zip([512, 256, 128, 64, 32], [256, 128, 64, 32, 16])
        ])
        self.last_conv = nn.Conv2d(16, output_channels, kernel_size=1)  # 输出通道数为 3
    
    def forward(self, z, encoded):
        z = self.fc(z).view(-1, 512, 7, 7)
        x = z + encoded
        for layer in self.deconv_layers:
            x = layer(x)
        x = self.last_conv(x)
        return x

# 融合模型
class TextImageFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_extract = ImageFeatureExtract()
        self.txt_extract = TextFeatureExtract()
        self.img_map = MappingNetwork(2048, 1024)
        self.txt_map = MappingNetwork(768, 1024)
        self.fusion_to_spatial = nn.Linear(1024, 1024 * 7 * 7)  # 新增线性层
        self.encoder = TwoPathEncoder()
        self.decoder = Decoder()  # 已修改为输出 3 通道
    
    def forward(self, image, text_ids):
        # Step 1: 提取特征
        img_fea = self.img_extract(image)  # [batch_size, 2048]
        txt_fea = self.txt_extract(text_ids)  # [batch_size, seq_length, 768]
        txt_fea = txt_fea.mean(dim=1)  # 池化: [batch_size, 768]
    
        # Step 2: 映射特征
        img_mapped = self.img_map(img_fea)  # [batch_size, 1024]
        txt_mapped = self.txt_map(txt_fea)  # [batch_size, 1024]
    
        # Step 3: 融合特征
        fused = img_mapped + txt_mapped  # [batch_size, 1024]
    
        # Step 4: 映射到空间维度
        fused_spatial = self.fusion_to_spatial(fused)  # [batch_size, 1024*7*7]
        fused_spatial = fused_spatial.view(fused.size(0), 1024, 7, 7)  # [batch_size, 1024, 7, 7]
    
        # Step 5: 编码和解码
        encoded = self.encoder(fused_spatial)  # [batch_size, 512, h, w]
        output = self.decoder(fused, encoded)  # [batch_size, 3, 224, 224]
    
        # 返回解码器输出及图像和文本嵌入
        return output, img_mapped, txt_mapped

# Contrastive Loss
def contrastive_loss(img_emb, txt_emb, temperature=0.07):
    similarity_matrix = torch.matmul(img_emb, txt_emb.T) / temperature  # 添加温度缩放
    labels = torch.arange(similarity_matrix.size(0)).to(img_emb.device)
    img_loss = F.cross_entropy(similarity_matrix, labels)
    txt_loss = F.cross_entropy(similarity_matrix.T, labels)
    return (img_loss + txt_loss) / 2