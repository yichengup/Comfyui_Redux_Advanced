import torch
import comfy.ops
from comfy.ldm.flux.redux import ReduxImageEncoder
import math
import torch.nn.functional as F

# 获取ops引用
ops = comfy.ops.manual_cast

class StyleAdvancedApply:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "style_model": ("STYLE_MODEL",),
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "reference_image": ("IMAGE",),
                
                # 处理模式选择
                "processing_mode": (["balanced", "style_focus", "content_focus", "custom"], {
                    "default": "balanced",
                    "tooltip": "预设处理模式：平衡、风格优先、内容优先、自定义"
                }),
                
                # 基础影响力控制
                "prompt_influence": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "控制提示词的影响强度"
                }),
                "image_influence": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "控制参考图像的影响强度"
                }),
                
                # 特征混合模式
                "feature_blend_mode": (["adaptive", "add", "multiply", "maximum"], {
                    "default": "adaptive",
                    "tooltip": "特征混合方式：自适应、加法、乘法、最大值"
                }),
                
                # 风格网格控制
                "style_grid_size": ("INT", {
                    "default": 9,
                    "min": 1,
                    "max": 14,
                    "step": 1,
                    "tooltip": "控制风格细节级别(1=27×27最细致, 14=1×1最粗略)"
                }),
            },
            
            "optional": {
                # 蒙版控制组
                "mask": ("MASK",),
                "mask_blur": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 64,
                    "step": 1,
                    "tooltip": "蒙版边缘模糊半径"
                }),
                "mask_expansion": ("INT", {
                    "default": 0,
                    "min": -64,
                    "max": 64,
                    "step": 1,
                    "tooltip": "蒙版扩张/收缩像素"
                }),
                
                # 高级调优选项
                "feature_weights": ("STRING", {
                    "default": "1.2,1.0,1.1,1.3,1.0",
                    "tooltip": "风格,颜色,内容,结构,纹理的权重(用逗号分隔)"
                }),
                "noise_level": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "添加随机噪声以增加风格变化"
                })
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_style"
    CATEGORY = "conditioning/style_model"

    def __init__(self):
        self.text_projector = ops.Linear(4096, 4096)
        self.feature_dim = 4096
        self.num_segments = 5  # 特征分段数量
        
    def compute_similarity(self, text_feat, image_feat):
        """计算多维度相似度"""
        # 余弦相似度
        cos_sim = torch.cosine_similarity(text_feat, image_feat, dim=-1)
        
        # L2距离相似度
        l2_dist = torch.norm(text_feat - image_feat, p=2, dim=-1)
        l2_sim = 1 / (1 + l2_dist)
        
        # 注意力相似度
        attention = torch.matmul(text_feat, image_feat.transpose(-2, -1))
        attention = attention / torch.sqrt(torch.tensor(text_feat.shape[-1], dtype=torch.float32))
        attn_sim = torch.softmax(attention, dim=-1).mean(dim=-1)
        
        # 组合相似度
        combined_sim = (0.4 * cos_sim + 0.3 * l2_sim + 0.3 * attn_sim)
        return combined_sim
        
    def process_features(self, image_features, text_features, mode, weights):
        """处理和混合特征"""
        # 维度检查和调整
        image_features = self.check_dimensions(image_features, "image_features")
        text_features = self.check_dimensions(text_features, "text_features")
        
        # 确保batch维度匹配
        if image_features.shape[0] != text_features.shape[0]:
            if image_features.shape[0] == 1:
                image_features = image_features.expand(text_features.shape[0], -1, -1)
            elif text_features.shape[0] == 1:
                text_features = text_features.expand(image_features.shape[0], -1, -1)
            else:
                raise ValueError(f"Batch size mismatch: image={image_features.shape[0]}, text={text_features.shape[0]}")
        
        # 分割特征
        splits = self.feature_dim // self.num_segments
        feature_types = ['style', 'color', 'content', 'structure', 'texture']
        
        # 解析权重
        try:
            feature_weights = [float(w) for w in weights.split(",")]
            if len(feature_weights) != self.num_segments:
                print(f"Warning: Expected {self.num_segments} weights, got {len(feature_weights)}. Using defaults.")
                feature_weights = [1.0] * self.num_segments
        except:
            feature_weights = [1.0] * self.num_segments
            
        # 根据模式调整权重
        if mode == "style_focus":
            feature_weights = [w * 1.5 if i != 2 else w * 0.5 
                             for i, w in enumerate(feature_weights)]
        elif mode == "content_focus":
            feature_weights = [w * 0.5 if i != 2 else w * 1.5 
                             for i, w in enumerate(feature_weights)]
        
        # 处理每个特征段
        processed_features = []
        for i, feat_type in enumerate(feature_types):
            start_idx = i * splits
            end_idx = start_idx + splits if i < len(feature_types) - 1 else self.feature_dim
            
            # 获取当��特征段
            img_feat = image_features[..., start_idx:end_idx]
            txt_feat = text_features[..., start_idx:end_idx]
            
            # 计算相似度和混合
            similarity = self.compute_similarity(txt_feat, img_feat)
            weight = feature_weights[i]
            
            # 确保相似度维度正确
            if similarity.dim() == 1:
                similarity = similarity.unsqueeze(-1)
            
            # 混合特征
            processed = txt_feat * similarity.unsqueeze(-1) * weight + \
                       img_feat * (1 - similarity.unsqueeze(-1)) * weight
            
            processed_features.append(processed)
        
        # 合并所有特征
        final_features = torch.cat(processed_features, dim=-1)
        
        # 最终维度检查
        final_features = self.check_dimensions(final_features, "final_features")
        
        return final_features
        
    def blend_features(self, img_feat, text_feat, mode="adaptive", weight=1.0):
        """特征混合"""
        if mode == "adaptive":
            similarity = self.compute_similarity(text_feat, img_feat)
            blend_weight = torch.sigmoid(similarity * weight)
            return img_feat * (1 - blend_weight.unsqueeze(-1)) + \
                   text_feat * blend_weight.unsqueeze(-1)
        elif mode == "add":
            return img_feat + text_feat * weight
        elif mode == "multiply":
            return img_feat * (1 + text_feat * weight)
        elif mode == "maximum":
            return torch.maximum(img_feat, text_feat * weight)
        return img_feat
        
    def prepare_image(self, image, mask=None, mode="center_crop", padding=32):
        """预处理参考图像"""
        B, H, W, C = image.shape
        
        if mode == "center_crop":
            # 计算裁剪位置（居中裁剪）
            crop_size = min(H, W)
            x = max(0, (W - crop_size) // 2)
            y = max(0, (H - crop_size) // 2)
            
            # 执行裁剪
            end_x = x + crop_size
            end_y = y + crop_size
            image = image[:, y:end_y, x:end_x, :]
            
            # 调整大小
            image = torch.nn.functional.interpolate(
                image.transpose(-1, 1),
                size=(self.desired_size, self.desired_size),
                mode="bicubic",
                antialias=True,
                align_corners=True
            ).transpose(1, -1)
            
        elif mode == "keep_aspect":
            # 保持宽高比调整大小
            ratio = self.desired_size / max(H, W)
            new_h = int(H * ratio)
            new_w = int(W * ratio)
            
            image = torch.nn.functional.interpolate(
                image.transpose(-1, 1),
                size=(new_h, new_w),
                mode="bicubic",
                antialias=True,
                align_corners=True
            ).transpose(1, -1)
            
        elif mode in ["mask_crop", "mask_region"] and mask is not None:
            # 处理蒙版相关的裁剪
            mask_binary = (mask > 0.5).float()
            y_indices, x_indices = torch.where(mask_binary > 0)
            
            if len(y_indices) > 0 and len(x_indices) > 0:
                # 计算边界框
                top = max(0, y_indices.min().item() - padding)
                bottom = min(H, y_indices.max().item() + padding)
                left = max(0, x_indices.min().item() - padding)
                right = min(W, x_indices.max().item() + padding)
                
                # 裁剪图像
                image = image[:, top:bottom, left:right, :]
                
                # 调整大小
                image = torch.nn.functional.interpolate(
                    image.transpose(-1, 1),
                    size=(self.desired_size, self.desired_size),
                    mode="bicubic",
                    antialias=True,
                    align_corners=True
                ).transpose(1, -1)
        
        return image
        
    def gaussian_blur(self, tensor, kernel_size):
        """实现高斯模糊
        Args:
            tensor: 输入张量 [B, C, H, W]
            kernel_size: 模糊核大小
        """
        if kernel_size <= 0:
            return tensor
            
        # 确保kernel_size是奇数
        kernel_size = kernel_size * 2 + 1 if kernel_size > 0 else 1
        
        # 创建高斯核
        sigma = kernel_size / 6.0  # 标准差
        channels = tensor.shape[1] if len(tensor.shape) == 4 else 1
        
        # 生成一维高斯核
        x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32, device=tensor.device)
        gaussian_1d = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        
        # 扩展为二维核
        kernel = gaussian_1d.view(1, 1, -1, 1) * gaussian_1d.view(1, 1, 1, -1)
        kernel = kernel.expand(channels, 1, kernel_size, kernel_size)
        
        # 应用padding
        padding = kernel_size // 2
        
        # 确保输入tensor是4D
        input_dim = len(tensor.shape)
        if input_dim == 3:
            tensor = tensor.unsqueeze(1)
        
        # 应用模糊
        blurred = F.conv2d(
            tensor,
            kernel,
            padding=padding,
            groups=channels
        )
        
        # 恢复原始维度
        if input_dim == 3:
            blurred = blurred.squeeze(1)
            
        return blurred

    def process_mask_in_pixel_space(self, mask, blur_radius, expansion, reference_image):
        """在像素空间处理蒙版，使用参考图像进行空间定位
        Args:
            mask: 输入蒙版
            blur_radius: 模糊半径
            expansion: 扩张/收缩像素
            reference_image: 参考图像，用于空间定位
        """
        if mask is None:
            return None
            
        # 获取参考图像的尺寸
        image_size = reference_image.shape[1:3]  # [B, H, W, C] -> [H, W]
            
        # 确保蒙版是正确的尺寸
        if mask.shape[-2:] != image_size:
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(1) if mask.dim() == 3 else mask,
                size=image_size,
                mode='bilinear',
                align_corners=False
            )
        
        # 扩张/收缩处理
        if expansion != 0:
            kernel_size = abs(expansion) * 2 + 1
            padding = kernel_size // 2
            kernel = torch.ones(1, 1, kernel_size, kernel_size).to(mask.device)
            
            if expansion > 0:  # 扩张
                mask = torch.nn.functional.max_pool2d(mask, kernel_size, stride=1, padding=padding)
            else:  # 收缩
                mask = 1 - torch.nn.functional.max_pool2d(1 - mask, kernel_size, stride=1, padding=padding)
        
        # 高斯模糊
        if blur_radius > 0:
            mask = self.gaussian_blur(mask, blur_radius)
            
        # 归一化到[0,1]范围
        mask = torch.clamp(mask, 0, 1)
        
        return mask

    def apply_style(self, conditioning, style_model, clip_vision_output, reference_image,
                   processing_mode="balanced",
                   prompt_influence=1.0, image_influence=1.0,
                   style_grid_size=9, feature_blend_mode="adaptive",
                   mask=None, mask_blur=4, mask_expansion=0,
                   noise_level=0.0, feature_weights="1.2,1.0,1.1,1.3,1.0"):
        """应用风格的主要方法"""
        try:
            # 获取并检查图像特征（从CLIP Vision获取）
            image_features = style_model.get_cond(clip_vision_output)
            image_features = self.check_dimensions(
                image_features.flatten(start_dim=0, end_dim=1),
                "image_features"
            )
            
            # 获取并检查文本特征
            text_features = conditioning[0][0]
            text_features = self.check_dimensions(
                text_features.mean(dim=1),
                "text_features"
            )
            
            # 处理特征
            final_features = self.process_features(
                image_features,
                text_features,
                processing_mode,
                feature_weights
            )
            
            # 应用全局影响力
            final_features = (
                final_features * image_influence + 
                text_features * prompt_influence
            ) / (image_influence + prompt_influence)
            
            # 在像素空间处理蒙版，使用参考图像进行空间定位
            if mask is not None:
                processed_mask = self.process_mask_in_pixel_space(
                    mask,
                    mask_blur,
                    mask_expansion,
                    reference_image
                )
                
                # 将处理后的蒙版应用到特征上
                if processed_mask is not None:
                    # 计算特征的空间尺寸
                    feature_size = int(math.sqrt(final_features.shape[1]))
                    
                    # 调整蒙版尺寸以匹配特征空间
                    processed_mask = torch.nn.functional.interpolate(
                        processed_mask.unsqueeze(1) if processed_mask.dim() == 3 else processed_mask,
                        size=(feature_size, feature_size),
                        mode='bilinear',
                        align_corners=False
                    )
                    
                    # 调整蒙版维度以匹配特征
                    processed_mask = processed_mask.flatten(1)
                    processed_mask = processed_mask.unsqueeze(-1)
                    
                    # 确保维度匹配
                    if processed_mask.shape[1] != final_features.shape[1]:
                        print(f"Warning: Mask shape {processed_mask.shape} does not match feature shape {final_features.shape}")
                        processed_mask = torch.nn.functional.interpolate(
                            processed_mask.view(processed_mask.shape[0], 1, feature_size, feature_size),
                            size=(int(math.sqrt(final_features.shape[1])), int(math.sqrt(final_features.shape[1]))),
                            mode='bilinear',
                            align_corners=False
                        ).flatten(1).unsqueeze(-1)
                    
                    # 应用蒙版
                    final_features = final_features * processed_mask + \
                                   text_features * (1 - processed_mask)
            
            # 添加噪声
            if noise_level > 0:
                noise = torch.randn_like(final_features) * noise_level
                final_features = final_features + noise
            
            # 构建新的条件
            c = []
            for t in conditioning:
                # 确保维度匹配
                orig_shape = t[0].shape
                adjusted_features = final_features
                
                # 调整batch维度
                if adjusted_features.shape[0] != orig_shape[0]:
                    adjusted_features = adjusted_features.expand(orig_shape[0], -1, -1)
                
                # 连接特征
                try:
                    n = [torch.cat((t[0], adjusted_features), dim=1), t[1].copy()]
                    c.append(n)
                except RuntimeError as e:
                    print(f"维度信息 - 原始条件: {t[0].shape}, 调整后特征: {adjusted_features.shape}")
                    raise RuntimeError(f"特征连接失败: {str(e)}") from e
            
            return (c,)
            
        except Exception as e:
            print(f"Error in apply_style: {str(e)}")
            print(f"维度信息:")
            if 'text_features' in locals(): print(f"- text_features: {text_features.shape}")
            if 'image_features' in locals(): print(f"- image_features: {image_features.shape}")
            if 'final_features' in locals(): print(f"- final_features: {final_features.shape}")
            raise
        
    def check_dimensions(self, features, name="features"):
        """检查并修正特征维度"""
        if features is None:
            raise ValueError(f"{name} cannot be None")
            
        # 确保是tensor
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features)
            
        # 添加缺失的维度
        if features.dim() == 2:
            features = features.unsqueeze(0)  # [N, D] -> [1, N, D]
        elif features.dim() == 1:
            features = features.unsqueeze(0).unsqueeze(0)  # [D] -> [1, 1, D]
        elif features.dim() == 4:  # [B, H, W, C]
            features = features.flatten(1, 2)  # -> [B, H*W, C]
            
        # 检查最后一个维度
        if features.shape[-1] != self.feature_dim:
            print(f"Warning: {name} dimension mismatch. Expected {self.feature_dim}, got {features.shape[-1]}. Adjusting...")
            features = self.resize_feature_dim(features)
            
        return features
        
    def resize_feature_dim(self, features):
        """调整特征维度到目标维度"""
        orig_shape = features.shape
        # 展到2D进行处理
        flat_features = features.reshape(-1, features.shape[-1])
        
        # 使用线性插值调整维度
        resized = torch.nn.functional.interpolate(
            flat_features.unsqueeze(0).unsqueeze(-2),
            size=(1, self.feature_dim),
            mode='linear',
            align_corners=False
        ).squeeze(0).squeeze(-2)
        
        # 恢复原始维度数
        new_shape = list(orig_shape[:-1]) + [self.feature_dim]
        return resized.reshape(new_shape)

