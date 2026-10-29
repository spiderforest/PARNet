
from timm.models.swin_transformer import SwinTransformer


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from .networks import get_norm_layer, init_net
from .deepcrack_networks import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer import SwinTransformer

class DynamicAlignConv(nn.Module):
    def __init__(self, in_channels, out_channels, mask_ratio=0.5):
        super(DynamicAlignConv, self).__init__()
        self.mask_ratio = mask_ratio

        # 主卷积
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        
        # 学习型遮罩生成器
        self.mask_conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, stride=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        # 全局上下文整合
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // 2, out_channels, bias=False),  # 映射到 `out_channels`
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 计算动态遮罩
        mask = self.sigmoid(self.mask_conv(x))  # 输出形状: (B, 1, H, W)
        
        # 对主特征进行加权
        masked_x = x * mask  # 仅保留重要区域
        
        # 主卷积操作
        local_feature = self.conv(masked_x)  # 输出形状: (B, out_channels, H, W)
        
        # 全局上下文特征
        global_feature = self.global_pool(x)  # 输出形状: (B, in_channels, 1, 1)
        global_feature = global_feature.view(global_feature.size(0), -1)  # 展平为 (B, in_channels)
        global_feature = self.fc(global_feature).view(global_feature.size(0), -1, 1, 1)  # 恢复为 (B, out_channels, 1, 1)
        
        # 将全局特征加权到输出特征
        output = local_feature * global_feature
        return output

class ResidualFusion(nn.Module):
    def __init__(self, in_channels_high, in_channels_low, out_channels):
        super(ResidualFusion, self).__init__()
        # 通道对齐
        self.conv_high = nn.Conv2d(in_channels_high, out_channels, kernel_size=1, bias=False)
        self.conv_low = nn.Conv2d(in_channels_low, out_channels, kernel_size=1, bias=False)
        
        # 残差卷积
        self.fuse_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, high_feat, low_feat):
        # 通道对齐
        high_feat_aligned = self.conv_high(high_feat)
        low_feat_aligned = self.conv_low(low_feat)

        # 特征融合
        fused_feat = high_feat_aligned + low_feat_aligned

        # 残差连接
        return self.fuse_conv(fused_feat) + low_feat_aligned

    
class SwinDeepCrackNet(nn.Module):
    def __init__(self, in_nc, num_classes=2, ngf=64, norm="batch", pretrained=True):
        super(SwinDeepCrackNet, self).__init__()

        # DeepCrackNet Encoder
        self.deepcrack = DeepCrackNet(in_nc, num_classes, ngf, norm)

        # Transformer Encoder
        self.transformer = SwinTransformer(
            img_size=256,
            patch_size=4,
            in_chans=in_nc,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            pretrained=pretrained,
        )

        # Store features from hooks
        self.features = []
        self._register_hooks()
        
        # Crack Boundary Branch
#         self.boundary_branch = BoundaryBranch(ngf * 8, ngf * 8)
      
        # Alignment convolutions ori
#         self.align_conv2 = nn.Conv2d(48, ngf * 2, kernel_size=1, stride=1, bias=False)
#         self.align_conv3 = nn.Conv2d(96, ngf * 4, kernel_size=1, stride=1, bias=False)
#         self.align_conv4 = nn.Conv2d(192, ngf * 8, kernel_size=1, stride=1, bias=False)
#         self.align_conv5 = nn.Conv2d(192, ngf * 8, kernel_size=1, stride=1, bias=False)
        self.align_conv2 = DynamicAlignConv(96, ngf * 2)
        self.align_conv3 = DynamicAlignConv(192, ngf * 4)
        self.align_conv4 = DynamicAlignConv(384, ngf * 8)
        self.align_conv5 = DynamicAlignConv(768, ngf * 8)
        
        
        # Fusion Convolutions
#         self.fuse_conv2 = nn.Conv2d(ngf * 2 + ngf * 2, ngf * 2, kernel_size=1, stride=1, bias=False)
#         self.fuse_conv3 = nn.Conv2d(ngf * 4 + ngf * 4, ngf * 4, kernel_size=1, stride=1, bias=False)
#         self.fuse_conv4 = nn.Conv2d(ngf * 8 + ngf * 8, ngf * 8, kernel_size=1, stride=1, bias=False)
#         self.fuse_conv5 = nn.Conv2d(ngf * 8 + ngf * 8, ngf * 8, kernel_size=1, stride=1, bias=False)
#         Residual Fusion
        self.residual_fuse2 = ResidualFusion(ngf * 2, ngf * 2, ngf * 2)
        self.residual_fuse3 = ResidualFusion(ngf * 4, ngf * 4, ngf * 4)
        self.residual_fuse4 = ResidualFusion(ngf * 8, ngf * 8, ngf * 8)
        self.residual_fuse5 = ResidualFusion(ngf * 8, ngf * 8, ngf * 8)     
#         self.residual_fuse2 = ResidualFusion(ngf * 2, ngf * 2, ngf * 2)
#         self.residual_fuse3 = ResidualFusion(ngf * 4, ngf * 4, ngf * 4)
#         self.residual_fuse4 = ResidualFusion(ngf * 8, ngf * 8, ngf * 8)
#         self.residual_fuse5 = ResidualFusion(ngf * 8, ngf * 8, ngf * 8)
    
#         hidden_dim = 512  # 根据需要设置隐藏层维度
#         self.residual_fuse2 = ResidualFusion(ngf * 2, ngf * 2, ngf * 2, hidden_dim)
#         self.residual_fuse3 = ResidualFusion(ngf * 4, ngf * 4, ngf * 4, hidden_dim)
#         self.residual_fuse4 = ResidualFusion(ngf * 8, ngf * 8, ngf * 8, hidden_dim)
#         self.residual_fuse5 = ResidualFusion(ngf * 8, ngf * 8, ngf * 8, hidden_dim)        
        # Output Convolution
        self.final_conv = nn.Conv2d(num_classes * 5, num_classes, kernel_size=1, stride=1, bias=False)

    def _register_hooks(self):
        """Register hooks to capture intermediate outputs."""
        def hook(module, input, output):
            self.features.append(output)

        # Register hooks for each stage of SwinTransformer
        for stage in self.transformer.layers:
            stage.register_forward_hook(hook)

    def forward(self, x):
        h, w = x.size(2), x.size(3)

        # Clear previously stored features
        self.features = []

        # DeepCrack Features
        conv1 = self.deepcrack.conv1(x)
        conv2 = self.deepcrack.conv2(self.deepcrack.maxpool(conv1))
        conv3 = self.deepcrack.conv3(self.deepcrack.maxpool(conv2))
        conv4 = self.deepcrack.conv4(self.deepcrack.maxpool(conv3))
        conv5 = self.deepcrack.conv5(self.deepcrack.maxpool(conv4))

#         print(f"DeepCrack conv2 shape: {conv2.shape}")
#         print(f"DeepCrack conv3 shape: {conv3.shape}")
#         print(f"DeepCrack conv4 shape: {conv4.shape}")
#         print(f"DeepCrack conv5 shape: {conv5.shape}")

        # Forward pass through Transformer
        _ = self.transformer(x)

        # Ensure 4 stages of features are captured
        if len(self.features) < 4:
            raise ValueError(f"Expected 4 features from Transformer, got {len(self.features)}")

        # Reshape and Align Transformer Features to match DeepCrack outputs
        tf_stage2 = self.features[0].permute(0, 3, 1, 2)
        tf_stage3 = self.features[1].permute(0, 3, 1, 2)
        tf_stage4 = self.features[2].permute(0, 3, 1, 2)
        tf_stage5 = self.features[3].permute(0, 3, 1, 2)

#         print(f"Transformer tf_stage2 shape: {tf_stage2.shape}")
#         print(f"Transformer tf_stage3 shape: {tf_stage3.shape}")
#         print(f"Transformer tf_stage4 shape: {tf_stage4.shape}")
#         print(f"Transformer tf_stage5 shape: {tf_stage5.shape}")

        # Align Transformer Features to DeepCrack Features
        tf_stage2_aligned = self.align_conv2(tf_stage2)
        tf_stage3_aligned = self.align_conv3(tf_stage3)
        tf_stage4_aligned = self.align_conv4(tf_stage4)
        tf_stage5_aligned = self.align_conv5(tf_stage5)
#         print(f"tf_stage2_aligned shape: {tf_stage2.shape}")
#         print(f"tf_stage2_aligned shape: {tf_stage3.shape}")
#         print(f"tf_stage2_aligned shape: {tf_stage4.shape}")
#         print(f"tf_stage2_aligned shape: {tf_stage5.shape}")
        # Ensure spatial sizes match
        tf_stage2_aligned = F.interpolate(tf_stage2_aligned, size=(conv2.size(2), conv2.size(3)), mode='bilinear', align_corners=True)
        tf_stage3_aligned = F.interpolate(tf_stage3_aligned, size=(conv3.size(2), conv3.size(3)), mode='bilinear', align_corners=True)
        tf_stage4_aligned = F.interpolate(tf_stage4_aligned, size=(conv4.size(2), conv4.size(3)), mode='bilinear', align_corners=True)
        tf_stage5_aligned = F.interpolate(tf_stage5_aligned, size=(conv5.size(2), conv5.size(3)), mode='bilinear', align_corners=True)
#         print(f"tf_stage2_aligned shape: {tf_stage2.shape}")
#         print(f"tf_stage2_aligned shape: {tf_stage3.shape}")
#         print(f"tf_stage2_aligned shape: {tf_stage4.shape}")
#         print(f"tf_stage2_aligned shape: {tf_stage5.shape}")

#         # Boundary Branch for each convolutional output
#         boundary_features2 = self.boundary_branch(conv2)
#         boundary_features3 = self.boundary_branch(conv3)
#         boundary_features4 = self.boundary_branch(conv4)
#         boundary_features5 = self.boundary_branch(conv5)
#         # Debug: Print boundary feature sizes
#         print(f"boundary_features2 size: {boundary_features2.shape}")
#         print(f"boundary_features3 size: {boundary_features3.shape}")
#         print(f"boundary_features4 size: {boundary_features4.shape}")
#         print(f"boundary_features5 size: {boundary_features5.shape}")
#         # Combine Boundary features with aligned transformer and conv features
#         combined2 = torch.cat([boundary_features2, tf_stage2_aligned, conv2], dim=1)
#         combined3 = torch.cat([boundary_features3, tf_stage3_aligned, conv3], dim=1)
#         combined4 = torch.cat([boundary_features4, tf_stage4_aligned, conv4], dim=1)
#         combined5 = torch.cat([boundary_features5, tf_stage5_aligned, conv5], dim=1)
#         # Debugging: Print shapes of DeepCrack Features
#         print(f"combined2 shape: {combined2.shape}")
#         print(f"combined3 shape: {combined3.shape}")
#         print(f"combined4 shape: {combined4.shape}")
#         print(f"combined5 shape: {combined5.shape}")
#         # Apply residual fusion
#         fused2 = self.residual_fuse2(combined2, conv2)
#         fused3 = self.residual_fuse3(combined3, conv3)
#         fused4 = self.residual_fuse4(combined4, conv4)
#         fused5 = self.residual_fuse5(combined5, conv5)

        # Debug: Check boundary feature shape
#         print(f"Boundary features shape: {boundary_features.shape}")          
#         Feature Fusion
#         fused2 = self.fuse_conv2(torch.cat([conv2, tf_stage2_aligned], dim=1))
#         fused3 = self.fuse_conv3(torch.cat([conv3, tf_stage3_aligned], dim=1))
#         fused4 = self.fuse_conv4(torch.cat([conv4, tf_stage4_aligned], dim=1))
#         fused5 = self.fuse_conv5(torch.cat([conv5, tf_stage5_aligned], dim=1))
#         print(f"tf_stage2_aligned shape: {tf_stage2_aligned.shape}")
#         print(f"tf_stage3_aligned shape: {tf_stage3_aligned.shape}")
#         print(f"tf_stage4_aligned shape: {tf_stage4_aligned.shape}")
#         print(f"tf_stage5_aligned shape: {tf_stage5_aligned.shape}")
        fused2 = self.residual_fuse2(tf_stage2_aligned, conv2)
        fused3 = self.residual_fuse3(tf_stage3_aligned, conv3)
        fused4 = self.residual_fuse4(tf_stage4_aligned, conv4)
        fused5 = self.residual_fuse5(tf_stage5_aligned, conv5)
        


        # Side Outputs
        side_output1 = self.deepcrack.side_conv1(conv1)
        side_output2 = self.deepcrack.side_conv2(fused2)
        side_output3 = self.deepcrack.side_conv3(fused3)
        side_output4 = self.deepcrack.side_conv4(fused4)
        side_output5 = self.deepcrack.side_conv5(fused5)


        # Upsample Side Outputs
        side_output2 = F.interpolate(side_output2, size=(h, w), mode='bilinear', align_corners=True)
        side_output3 = F.interpolate(side_output3, size=(h, w), mode='bilinear', align_corners=True)
        side_output4 = F.interpolate(side_output4, size=(h, w), mode='bilinear', align_corners=True)
        side_output5 = F.interpolate(side_output5, size=(h, w), mode='bilinear', align_corners=True)

        # Fuse Outputs
        fused_output = self.final_conv(torch.cat([side_output1, side_output2, side_output3, side_output4, side_output5], dim=1))

        return side_output1, side_output2, side_output3, side_output4, side_output5, fused_output



    
    
    
    
    
    
def define_swintrans(in_nc, 
#                      out_ch,
                     num_classes, 
                     ngf,
                     norm='batch',
                     init_type='xavier', 
                     
                     init_gain=0.02, 
                     
                     gpu_ids=[]):
#     net = UNetWithSwinTransformer(in_nc, out_ch, ngf, norm)
#     net = SwinDeepCrackNet(in_nc, out_ch)
#     return init_net(net, init_type, init_gain, gpu_ids)
    depths = [2, 2, 6, 2]
#     net = SwinDeepCrackNet(in_nc=3, num_classes=1)
    net = SwinDeepCrackNet(in_nc, num_classes, ngf)
    return init_net(net, init_type, init_gain, gpu_ids)
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, size_average=True):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.size_average = size_average
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
#         self.criterion = nn.BCELoss(size_average=True)

    def forward(self, inputs, targets):
        BCE_loss = self.criterion(inputs, targets.float())
#         F_loss = BCE_loss
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        

        if self.size_average:
            return F_loss.mean()
        else:
            return F_loss.sum()
        
#really bad performance with dice bce-dice loss
        
import torch
import torch.nn as nn
from torch.autograd import Variable as V

import cv2
import numpy as np
class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
#         self.bce_loss = nn.BCEWithLogitsLoss()
        self.bce_loss = nn.BCELoss(size_average=True)
        
    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        #score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss
        
    def __call__(self, y_true, y_pred):
        a =  self.bce_loss(y_pred.float(), y_true)
#         a = self.bce_loss(y_pred.float(), y_true)
        b =  self.soft_dice_loss(y_true, y_pred)
        return b       
