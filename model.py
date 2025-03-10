import torch
import torch.nn as nn
import math

class BottleneckBlock(nn.Module):
    """
    Modified Bottleneck block for PyramidNet without expansion to keep parameter count low.
    """
    # expansion = 4  # No expansion to keep parameter count low
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
        
        out = self.bn1(x)
        out = self.relu1(out)
        
        if self.downsample is not None:
            residual = self.downsample(out)
        
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)
        
        out += residual
        
        return out

class BasicBlock(nn.Module):
    """
    Basic block for PyramidNet without bottleneck.
    """
    # expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
        
        out = self.bn1(x)
        out = self.relu1(out)
        
        if self.downsample is not None:
            residual = self.downsample(out)
        
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        
        out += residual
        
        return out

class PyramidNet(nn.Module):
    """
    PyramidNet with modified blocks to reduce parameter count
    """
    def __init__(self, block, num_blocks, alpha, num_classes=10):
        super(PyramidNet, self).__init__()
        self.inplanes = 16
        
        # Calculate the number of channels at each layer
        # alpha is the total growth rate (additional feature maps)
        n = sum(num_blocks)
        self.addrate = alpha / n
        
        # First layer
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        
        # Pyramid layers
        self.layer1 = self._make_pyramid_layer(block, num_blocks[0], stride=1)
        self.layer2 = self._make_pyramid_layer(block, num_blocks[1], stride=2)
        self.layer3 = self._make_pyramid_layer(block, num_blocks[2], stride=2)
        
        # Final batch norm and classifier
        self.bn_final = nn.BatchNorm2d(self.inplanes)
        self.relu_final = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.inplanes, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def _make_pyramid_layer(self, block, blocks, stride=1):
        layers = []
        
        # For the first block in each layer
        outplanes = int(math.floor(self.inplanes + self.addrate))
        
        # Always use downsample for the first block - even with stride=1
        # This is critical to match dimensions for the first residual connection
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, outplanes, 
                      kernel_size=1, stride=stride, bias=False)
        )
        
        # Add first block
        layers.append(block(self.inplanes, outplanes, stride, downsample))
        self.inplanes = outplanes
        
        # Remaining blocks in this layer
        for i in range(1, blocks):
            outplanes = int(math.floor(self.inplanes + self.addrate))
            
            # Downsample whenever channels change (which is every block in PyramidNet)
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, outplanes, 
                          kernel_size=1, stride=1, bias=False)
            )
            
            layers.append(block(self.inplanes, outplanes, 1, downsample))
            self.inplanes = outplanes
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.bn_final(x)
        x = self.relu_final(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def create_pyramidnet(num_blocks, use_bottleneck=True, alpha=270, num_classes=10):
    """
    Factory function to create a lightweight PyramidNet model
    
    Args:
        num_blocks: List specifying the number of blocks in each layer
        use_bottleneck: Whether to use bottleneck blocks (True) or basic blocks (False)
        alpha: The widening factor for the network
        num_classes: Number of output classes (10 for CIFAR-10)
        
    Returns:
        A PyramidNet model
    """
    block = BottleneckBlock if use_bottleneck else BasicBlock
    return PyramidNet(block, num_blocks, alpha, num_classes)