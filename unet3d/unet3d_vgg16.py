"""
3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation
Paper URL: https://arxiv.org/abs/1606.06650
Author: Amir Aghdam
"""

from torch import nn
from torchsummary import summary
import torch
import time


def init_weights(m):
    """ Same initialization as the segmentation-models-3D implementation: 'glorot_uniform', and zero bias """
    if isinstance(m, nn.Conv3d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)

class Conv3DBlock_2conv(nn.Module):
    """
    The basic block for 3x3x3 convolutions in the analysis path,
    with 2 convolutions, no batch_norm.
    
    Based on the 1st block of VGG16
    
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> desired number of output channels
    -- forward()
    :param input -> input Tensor to be convolved
    :return -> Tensor
    """

    def __init__(self, in_channels, out_channels) -> None:
        super(Conv3DBlock_2conv, self).__init__()
        self.conv1 = nn.Conv3d(in_channels= in_channels, out_channels=out_channels, kernel_size=(3,3,3), padding='same')
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(in_channels= out_channels, out_channels=out_channels, kernel_size=(3,3,3), padding='same')

        self.pooling = nn.MaxPool3d(kernel_size=(2,2,2), stride=2)

    
    def forward(self, input):
        #conv1 + relu
        res = self.relu(self.conv1(input))
        #conv2 + relu
        res = self.relu(self.conv2(res))
        #max pooling
        out = self.pooling(res)
  
        return out, res




class Conv3DBlock_3conv(nn.Module):
    """
    The basic block for 3x3x3 convolutions in the analysis path,
    with 3 convolutions, no batch_norm.
    
    Based on the 3rd block of VGG16
    
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> desired number of output channels
    -- forward()
    :param input -> input Tensor to be convolved
    :return -> Tensor
    """

    def __init__(self, in_channels, out_channels) -> None:
        super(Conv3DBlock_3conv, self).__init__()
        self.conv1 = nn.Conv3d(in_channels= in_channels, out_channels=out_channels, kernel_size=(3,3,3), padding='same')
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(in_channels= out_channels, out_channels=out_channels, kernel_size=(3,3,3), padding='same')
        self.pooling = nn.MaxPool3d(kernel_size=(2,2,2), stride=2)

    
    def forward(self, input):
        #conv1 + relu
        res = self.relu(self.conv1(input))
        #conv2 + relu
        res = self.relu(self.conv2(res))
        #conv3 + relu
        res = self.relu(self.conv2(res))
        #max pooling
        out = self.pooling(res)

        return out, res

    
    
    
class Conv3DBlock_center(nn.Module):
    """
    The basic block for 3x3x3 convolutions in the analysis path,
    with 1 convolutions, with batch_norm.
    
    
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> desired number of output channels
    -- forward()
    :param input -> input Tensor to be convolved
    :return -> Tensor
    """
    def __init__(self, in_channels, out_channels) -> None:
        super(Conv3DBlock_center, self).__init__()
        self.conv = nn.Conv3d(in_channels= in_channels, out_channels=out_channels, kernel_size=(3,3,3), padding='same', bias=False)
        self.bn = nn.BatchNorm3d(num_features=out_channels, eps=0.001, momentum=0.99) #copying Keras default args
        self.relu = nn.ReLU()

        
    def forward(self, input):
        #conv1 + bn + relu
        out = self.relu(self.bn(self.conv(input)))
        
        return out
    
    
    
    
class UpConv3DBlock(nn.Module):
    """
    The basic block for upsampling using 3x3x3 convolutions in the synthesis path.
    
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> number of residual connections' channels to be concatenated
    -- forward()
    :param input -> input Tensor
    :param residual -> residual connection to be concatenated with input
    :return -> Tensor
    """

    def __init__(self, in_channels, res_channels=0) -> None:
        super(UpConv3DBlock, self).__init__()
        self.upconv = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv3d(in_channels=in_channels+res_channels, out_channels=in_channels//2, kernel_size=(3,3,3), padding='same', bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(num_features=in_channels//2, eps=0.001, momentum=0.99)
        self.conv2 = nn.Conv3d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=(3,3,3), padding='same', bias=False)
        
        
    def forward(self, input, residual=None):
        #upsample
        out = self.upconv(input)
        #concat with encoder residual
        if residual!=None: out = torch.cat((out, residual), 1)
        #conv1 + bn + relu
        out = self.relu(self.bn(self.conv1(out)))
        #conv2 + bn + relu
        out = self.relu(self.bn(self.conv2(out)))
        return out



class FinalBlock(nn.Module):
    """
    The final block in the synthesis path.
    
    -- __init__()
    :param in_channels -> number of input channels
    :param n_classes -> number of final classes
    -- forward()
    :param input -> input Tensor
    :param residual -> residual connection to be concatenated with input
    :return -> Tensor
    """

    def __init__(self, in_channels, num_classes, use_softmax=False) -> None:
        super(FinalBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=num_classes, kernel_size=(3,3,3), padding='same', bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.use_softmax = use_softmax
        
        
    def forward(self, input):
        #final conv
        out = self.conv(input)
        if self.use_softmax:
            #softmax - class probabilities
            # This doesn't need to be used before torch.CrossEntropyLoss
            out = self.softmax(out)
        return out
        
        
class UNet3D_VGG16(nn.Module):
    """
    The 3D UNet model with VGG16 backbone
    -- __init__()
    :param in_channels -> number of input channels
    :param num_classes -> specifies the number of output channels or masks for different classes
    :param level_channels -> the number of channels at each level (count top-down)
    :param device -> the device on which to run the model
    -- forward()
    :param input -> input Tensor
    :return -> Tensor
    """
    
    def __init__(self, in_channels, num_classes, level_channels=[64, 128, 256, 512, 512], use_softmax_end=False) -> None:
        super(UNet3D_VGG16, self).__init__()
        level_1_chnls, level_2_chnls, level_3_chnls, level_4_chnls, level_5_chnls = level_channels[0], level_channels[1], level_channels[2],level_channels[3],level_channels[4]
 
        self.encoder_block1 = Conv3DBlock_2conv(in_channels=in_channels, out_channels=level_1_chnls)
        self.encoder_block2 = Conv3DBlock_2conv(in_channels=level_1_chnls, out_channels=level_2_chnls)
        self.encoder_block3 = Conv3DBlock_3conv(in_channels=level_2_chnls, out_channels=level_3_chnls)
        self.encoder_block4 = Conv3DBlock_3conv(in_channels=level_3_chnls, out_channels=level_4_chnls)
        self.encoder_block5 = Conv3DBlock_3conv(in_channels=level_4_chnls, out_channels=level_5_chnls)
        
        self.center_block1 = Conv3DBlock_center(in_channels=level_5_chnls, out_channels=level_5_chnls)
        self.center_block2 = Conv3DBlock_center(in_channels=level_5_chnls, out_channels=level_5_chnls)
        
        last_encoder_channels = level_5_chnls
        self.decoder_block5 = UpConv3DBlock(in_channels=last_encoder_channels, res_channels=level_5_chnls)
        self.decoder_block4 = UpConv3DBlock(in_channels=last_encoder_channels//2, res_channels=level_4_chnls)
        self.decoder_block3 = UpConv3DBlock(in_channels=last_encoder_channels//4, res_channels=level_3_chnls)
        self.decoder_block2 = UpConv3DBlock(in_channels=last_encoder_channels//8, res_channels=level_2_chnls)
        self.decoder_block1 = UpConv3DBlock(in_channels=last_encoder_channels//16, res_channels=0)

        self.final_block = FinalBlock(in_channels=last_encoder_channels//32, num_classes=num_classes, use_softmax=use_softmax_end)


    
    def forward(self, input):
        #Analysis path forward feed
        out, _ = self.encoder_block1(input)
        out, residual_level2 = self.encoder_block2(out)
        out, residual_level3 = self.encoder_block3(out)
        out, residual_level4 = self.encoder_block4(out)
        out, residual_level5 = self.encoder_block5(out)

        out = self.center_block1(out)
        out = self.center_block2(out)

        #Synthesis path forward feed
        out = self.decoder_block5(out, residual_level5)
        out = self.decoder_block4(out, residual_level4)
        out = self.decoder_block3(out, residual_level3)
        out = self.decoder_block2(out, residual_level2)
        out = self.decoder_block1(out)

        out = self.final_block(out)
        return out



if __name__ == '__main__':
    #Configurations according to the SAIAD dataset
    model = UNet3D_VGG16(in_channels=1, num_classes=5).cuda()
    start_time = time.time()
    summary(model=model, input_size=(1, 96,96,96), batch_size=-
            1, device="cuda")
    print("--- %s seconds ---" % (time.time() - start_time))