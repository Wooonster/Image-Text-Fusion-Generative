import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from torchvision import models
# pip install git+https://github.com/openai/CLIP.git


# Extract image features
# use ResNet50
class ImageFeatureExtract(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)  # load pretrained ResNet50 from torchvision
        self.feature_extract = nn.Sequential(
            *list(
                resnet.children()  # iterator over the layers (modules) of the ResNet-50 architecture
                )[:-1]  # removes the last layer, the MLP 
            )  # chain the remaining ResNet-50 layers together

    def forward(self, x):
        # x: (batch_size, 2048, 1, 1) 
        # ResNet-50's final convolutional output is globally averaged to a single spatial position per channel
        x = self.feature_extract(x)
        return x.view(x.size(0), -1)


# Extract text features
# use BERT-base
class TextFeatureExtract(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
    
    def forward(self, input_ids, attention_mask):
        '''
        input_ids:      A tensor containing tokenized text sequences represented as token IDs. (batch_size, seq_length)
        attention_mask: A tensor of the same shape as input_ids. It indicates which tokens in input_ids should be attended to:
                        1: Attend to this token.
                        0: Ignore this token (padding).
        outputs.last_hidden_state[:, 0, :]:  Extract the CLS embedding from the bert results
        '''
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # CLS token embedding, shape: [batch_size, 768]
    

# Contrastive Learning for mapping text with image
class MappingNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.map = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        x = self.map(x)
        return F.normalize(x, p=2, dim=1)
    
# Two-Path VAE Encoder
class TwoPathEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ) for in_channels, out_channels in zip([1, 32, 64, 128, 256], [32, 64, 128, 256, 512])
        ])

    def forward(self, x):
        pooled = x  # init the pooling
        for layer in self.cnn_layers:
            cnn_out = layer(x)
            pooled = F.max_pool2d(pooled, kernel_size=2, stride=2)
            x = cnn_out + pooled
        return x
    
# Decoder
class Decoder(nn.Module):
    def __init__(self, encoded_dim=512, output_channels=1):
        super().__init__()
        self.fc = nn.Linear(1024, 7 * 7 * encoded_dim)  # Mapping fused vector to spatial representation
        self.deconv_layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ) for in_channels, out_channels in zip([512, 512, 512, 512, 512], [512, 256, 128, 64, 32])
        ])

        self.last_conv = nn.Conv2d(32, output_channels, kernel_size=1)

    def forward(self, z, encoded):
        z = self.fc(z).view(-1, 512, 7, 7)
        x = z + encoded
        for layer in self.deconv_layers:
            x = layer(x)
        return self.last_conv(x)
    

# Full Model
class TextImageFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_extract = ImageFeatureExtract()
        self.txt_extract = TextFeatureExtract()
        self.img_map = MappingNetwork(2048, 1024)
        self.txt_map = MappingNetwork(768, 1024)
        self.encoder = TwoPathEncoder()
        self.decoder = Decoder()

    def forward(self, image, text_ids, text_mask):
        # extract
        img_fea = self.img_extract(image)
        txt_fea = self.txt_extract(text_ids, text_mask)

        # map
        img_mapped = self.img_map(img_fea)
        txt_mapped = self.txt_map(txt_fea)

        # fused
        fused = img_mapped + txt_mapped

        encoded = self.encoder(fused)
        output = self.decoder(fused, encoded)
        return output
    

# Contrastive Loss
def contrastive_loss(img_emb, txt_emb):
    similarity_matrix = torch.matmul(img_emb, txt_emb.T)
    labels = torch.arange(similarity_matrix.size(0)).to(img_emb.device)
    img_loss = F.cross_entropy(similarity_matrix, labels)
    txt_loss = F.cross_entropy(similarity_matrix.T, labels)
    return (img_loss + txt_loss) / 2

model = TextImageFusionModel()
print(model)