import torch.nn as nn
class ModelDiscriminate(nn.Module): # 由5个卷基层+4个relu构成
    def __init__(self, num_classes, ndf=64): # num_classes=1
        super(ModelDiscriminate, self).__init__()
        self.conv1 = nn.Conv2d(
            num_classes, ndf,
            kernel_size=4, stride=2, padding=1)
        # Conv2d(in_channels, out_channels,
        # kernel_size, stride=1, padding=0)[
        self.conv2 = nn.Conv2d(
            ndf, ndf*2,
            kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(
            ndf*2, ndf*4,
            kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(
            ndf*4, ndf*8,
            kernel_size=4, stride=2, padding=1)
        # self.pam = PAM_Module(512)
        self.classifier = nn.Conv2d(
            ndf*8, 1,
            kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # 如果x>0,f(x)=x 如果x<0,f(x)=0.2*x

    def forward(self, x): # [n,1,256**2]
        x = self.conv1(x) # [n,64,128**2]<-
        x = self.leaky_relu(x)
        x = self.conv2(x) # [n,2*64,64**2]<-
        x = self.leaky_relu(x)
        x = self.conv3(x) # [n,4*64,32**2]<-
        x = self.leaky_relu(x)
        x = self.conv4(x) # [n,8*64,16**2]<-
        x = self.leaky_relu(x)
        x = self.classifier(x) # [n,1,8**2]
        return x
