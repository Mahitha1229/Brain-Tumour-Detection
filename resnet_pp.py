"""
ResNet++ Model for Brain Tumor Detection
Pure PyTorch implementation without torchvision dependency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """Basic ResNet block with residual connection"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class AttentionBlock(nn.Module):
    """Channel attention mechanism for ResNet++"""
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        channel_att = self.channel_attention(x)
        return x * channel_att

class ResNetPlusPlus(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNetPlusPlus, self).__init__()
        print("Building ResNet++ from Scratch...")
        
        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet layers with attention
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.attention1 = AttentionBlock(64)
        
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.attention2 = AttentionBlock(128)
        
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.attention3 = AttentionBlock(256)
        
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.attention4 = AttentionBlock(512)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.attention1(x)
        
        x = self.layer2(x)
        x = self.attention2(x)
        
        x = self.layer3(x)
        x = self.attention3(x)
        
        x = self.layer4(x)
        x = self.attention4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# Test the model
if __name__ == "__main__":
    print("Testing Pure PyTorch ResNet++ Implementation")
    print("=" * 60)
    
    model = ResNetPlusPlus(num_classes=4)
    print("✓ ResNet++ model created successfully!")
    
    # Test with sample input matching your data
    sample_input = torch.randn(4, 3, 128, 128)  # Batch of 4, 128x128 RGB images
    print(f"Input shape: {sample_input.shape}")
    
    output = model(sample_input)
    print(f"Output shape: {output.shape}")
    print("✓ Model is working correctly!")