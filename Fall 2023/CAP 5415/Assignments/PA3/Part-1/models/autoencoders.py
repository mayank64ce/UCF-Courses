import torch
import torch.nn as nn

class AutoencoderFC(nn.Module):
    def __init__(self, img_size=28, channels=1):
        super().__init__()
        self.img_size = img_size
        self.channels = channels

        self.flatten = nn.Flatten()
        self.encoder = nn.ModuleList([
            nn.Linear(self.img_size * self.img_size * self.channels, 256),
            nn.Linear(256, 128)
        ])
        self.decoder = nn.ModuleList([
            nn.Linear(128, 256),
            nn.Linear(256, self.img_size * self.img_size * self.channels)
        ])
        
    def forward(self, x):
        x_shape = x.shape
        x = self.flatten(x)

        for layer in self.encoder:
            x = layer(x)
        
        for layer in self.decoder:
            x = layer(x)
        
        x = x.reshape(x_shape)

        return x

class AutoencoderCNN(nn.Module):
    def __init__(self, img_size=28, channels=1):
        super().__init__()
        self.img_size = img_size
        self.channels = channels

        self.encoder = nn.ModuleList([
            nn.Conv2d(1, 16, 3, padding=1),  # 1st Convolutional Layer
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 1st Max-Pooling Layer
            nn.Conv2d(16, 32, 3, padding=1),  # 2nd Convolutional Layer
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 2nd Max-Pooling Layer
        ])
        self.decoder = nn.ModuleList([
            nn.Conv2d(32, 16, 3, padding=1),  # 1st Convolutional Layer
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),  # 1st Upsampling Layer
            nn.Conv2d(16, 8, 3, padding=1),  # 2nd Convolutional Layer
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),  # 2nd Upsampling Layer
            nn.Conv2d(8, 1, 3, padding=1),  # 3rd Convolutional Layer
        ])
        
    def forward(self, x):
        
        for layer in self.encoder:
            x = layer(x)
        
        for layer in self.decoder:
            x = layer(x)

        return x


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 32
    img_size = 28
    channels = 1
    x = torch.rand(batch_size,channels,img_size,img_size).to(device)

    # model = AutoencoderFC().to(device)
    model = AutoencoderCNN().to(device)
    out = model(x)
    print(out.shape)