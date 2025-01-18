import torch
import math
import torch.nn.functional as F
import cv2
import numpy as np


class YC_LG_Redux:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "conditioning": ("CONDITIONING", ),
            "style_model": ("STYLE_MODEL", ),
            "clip_vision": ("CLIP_VISION",),
            "image": ("IMAGE",),
            "crop": (["center", "mask_area", "none"], {
                "default": "none",
                "tooltip": "裁剪模式：center-中心裁剪, mask_area-遮罩区域裁剪, none-不裁剪"
            }),
            "sharpen": ("FLOAT", {
                "default": 0.0,
                "min": -5.0,
                "max": 5.0,
                "step": 0.1,
                "tooltip": "锐化强度：负值为模糊，正值为锐化，0为不处理"
            }),
            "patch_res": ("INT", {
                "default": 16,
                "min": 1,
                "max": 64,
                "step": 1,
                "tooltip": "patch分辨率，数值越大分块越细致"
            }),
            "style_strength": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 2.0,
                "step": 0.01,
                "tooltip": "风格强度，越高越偏向参考图片"
            }),
            "prompt_strength": ("FLOAT", {  # 新增参数
                "default": 1.0,
                "min": 0.0,
                "max": 2.0,
                "step": 0.01,
                "tooltip": "文本提示词强度，越高文本特征越强"
            }),
            "blend_mode": (["lerp", "feature_boost", "frequency"], {
                "default": "lerp",
                "tooltip": "风格强度的计算方式：\n" +
                        "lerp - 线性混合 - 高度参考原图\n" +
                        "feature_boost - 特征增强 - 增强真实感\n" +
                        "frequency - 频率增强 - 增强高频细节"
            }),
            "noise_level": ("FLOAT", {  # 新增噪声参数
                "default": 0.0,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "tooltip": "添加随机噪声的强度，可用于修复错误细节"
            }),
        },
        "optional": {  # 新增可选参数
            "mask": ("MASK", ),  # 添加遮罩输入
        }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_stylemodel"
    CATEGORY = "conditioning/style_model"

    def crop_to_mask_area(self, image, mask):
        """裁剪到遮罩区域"""
        # 处理图片维度
        if len(image.shape) == 4:  # B H W C
            B, H, W, C = image.shape
            image = image.squeeze(0)  # 移除batch维度
        else:  # H W C
            H, W, C = image.shape
        
        # 处理遮罩维度
        if len(mask.shape) == 3:  # B H W
            mask = mask.squeeze(0)  # 移除batch维度
        
        # 找到遮罩中非零值的坐标
        nonzero_coords = torch.nonzero(mask)
        if len(nonzero_coords) == 0:  # 如果遮罩全为零
            return image, mask
        
        # 获取边界框
        top = nonzero_coords[:, 0].min().item()
        bottom = nonzero_coords[:, 0].max().item()
        left = nonzero_coords[:, 1].min().item()
        right = nonzero_coords[:, 1].max().item()
        
        # 确保裁剪区域是正方形
        width = right - left
        height = bottom - top
        size = max(width, height)
        
        # 计算中心点
        center_y = (top + bottom) // 2
        center_x = (left + right) // 2
        
        # 计算正方形裁剪区域
        half_size = size // 2
        new_top = max(0, center_y - half_size)
        new_bottom = min(H, center_y + half_size)
        new_left = max(0, center_x - half_size)
        new_right = min(W, center_x + half_size)
        
        # 裁剪图片和遮罩
        cropped_image = image[new_top:new_bottom, new_left:new_right]
        cropped_mask = mask[new_top:new_bottom, new_left:new_right]
        
        # 恢复batch维度
        cropped_image = cropped_image.unsqueeze(0)
        cropped_mask = cropped_mask.unsqueeze(0)
        
        return cropped_image, cropped_mask
    
    def apply_image_preprocess(self, image, strength):
        """统一的图像预处理函数"""
        # 保存原始维度信息
        original_shape = image.shape
        original_device = image.device
        
        # 转换为numpy数组
        if torch.is_tensor(image):
            if len(image.shape) == 4:  # B H W C
                image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
            else:  # H W C
                image_np = (image.cpu().numpy() * 255).astype(np.uint8)
        
        if strength < 0:  # 模糊处理
            abs_strength = abs(strength)
            kernel_size = int(3 + abs_strength * 12) // 2 * 2 + 1  # 最大到15
            sigma = 0.3 + abs_strength * 2.7  # 最大到3
            processed = cv2.GaussianBlur(image_np, (kernel_size, kernel_size), sigma)
        elif strength > 0:  # 锐化处理
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]]) * strength + np.array([[0,0,0],
                                                               [0,1,0],
                                                               [0,0,0]]) * (1 - strength)
            processed = cv2.filter2D(image_np, -1, kernel)
            processed = np.clip(processed, 0, 255)
        else:  # 不处理
            processed = image_np
        
        # 转回tensor并恢复原始维度
        processed_tensor = torch.from_numpy(processed.astype(np.float32) / 255.0).to(original_device)
        if len(original_shape) == 4:
            processed_tensor = processed_tensor.unsqueeze(0)
        
        return processed_tensor

    def apply_style_strength(self, cond, txt, strength, mode="lerp"):
        """使用不同模式应用风格强度"""
        if mode == "lerp":
            # 线性插值模式保持不变
            if txt.shape[1] != cond.shape[1]:
                txt_mean = txt.mean(dim=1, keepdim=True)
                txt_expanded = txt_mean.expand(-1, cond.shape[1], -1)
                return torch.lerp(txt_expanded, cond, strength)
            return torch.lerp(txt, cond, strength)
        
        elif mode == "feature_boost":
            # feature_boost 模式保持不变
            mean = torch.mean(cond, dim=-1, keepdim=True)
            std = torch.std(cond, dim=-1, keepdim=True)
            normalized = (cond - mean) / (std + 1e-6)
            boost = torch.tanh(normalized * (strength * 2.0))
            return cond * (1 + boost * 2.0)
    
        elif mode == "frequency":
            try:
                # 获取形状
                B, N, C = cond.shape
                
                # 1. 准备数据
                x = cond.float()  # 确保使用浮点数
                
                # 2. 在特征维度上进行FFT
                fft = torch.fft.rfft(x, dim=-1)
                
                # 3. 分离幅度和相位
                magnitudes = torch.abs(fft)
                phases = torch.angle(fft)
                
                # 4. 创建频率增强滤波器
                freq_dim = fft.shape[-1]
                
                # 创建一个更复杂的频率响应曲线
                freq_range = torch.linspace(0, 1, freq_dim, device=cond.device)
                
                # 设计高频增强滤波器
                alpha = 2.0 * strength  # 控制增强强度
                beta = 0.5  # 控制频率响应的形状
                
                # 构建频率响应
                filter_response = 1.0 + alpha * torch.pow(freq_range, beta)
                filter_response = filter_response.view(1, 1, -1)
                
                # 5. 应用滤波器
                enhanced_magnitudes = magnitudes * filter_response
                
                # 6. 重建信号
                enhanced_fft = enhanced_magnitudes * torch.exp(1j * phases)
                enhanced = torch.fft.irfft(enhanced_fft, n=C, dim=-1)
                
                # 7. 归一化处理
                mean = enhanced.mean(dim=-1, keepdim=True)
                std = enhanced.std(dim=-1, keepdim=True)
                enhanced_norm = (enhanced - mean) / (std + 1e-6)
                
                # 8. 混合原始信号和增强信号
                mix_ratio = torch.sigmoid(torch.tensor(strength * 2 - 1))
                result = torch.lerp(cond, enhanced_norm.to(cond.dtype), mix_ratio)
                
                # 9. 添加残差连接
                residual = (result - cond) * strength
                final = cond + residual
                
                return final
                
            except Exception as e:
                print(f"频率处理出错: {e}")
                print(f"输入张量形状: {cond.shape}")
                return cond
                
        return cond

    def apply_stylemodel(self, conditioning, style_model, clip_vision, image, 
                        patch_res=16, style_strength=1.0, prompt_strength=1.0, 
                        noise_level=0.0, crop="none", sharpen=0.0,
                        blend_mode="lerp", mask=None):
        
        # 预处理输入图像
        processed_image = image.clone()
        if sharpen != 0:
            processed_image = self.apply_image_preprocess(processed_image, sharpen)
        # 处理裁剪
        if crop == "mask_area" and mask is not None:
            processed_image, mask = self.crop_to_mask_area(processed_image, mask)
            clip_vision_output = clip_vision.encode_image(processed_image, crop=False)
        else:
            # 对于center和none模式，直接使用CLIP的crop参数
            crop_image = True if crop == "center" else False
            clip_vision_output = clip_vision.encode_image(processed_image, crop=crop_image)
        
        # 获取原始条件向量
        cond = style_model.get_cond(clip_vision_output)
        
        # 重塑为空间结构 [batch, height, width, channels]
        B = cond.shape[0]
        H = W = int(math.sqrt(cond.shape[1]))  # 假设是方形
        C = cond.shape[2]
        cond = cond.reshape(B, H, W, C)
        
        # 根据patch_res进行重新划分
        new_H = H * patch_res // 16  # 16是CLIP默认的patch size
        new_W = W * patch_res // 16
        
        # 使用插值调整特征图大小
        cond = torch.nn.functional.interpolate(
            cond.permute(0, 3, 1, 2),  # [B, C, H, W]
            size=(new_H, new_W),
            mode='bilinear',
            align_corners=False
        )
        
        # 重新展平
        cond = cond.permute(0, 2, 3, 1)  # [B, H, W, C]
        cond = cond.reshape(B, -1, C)  # [B, H*W, C]
        cond = cond.flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
        
        c_out = []
        for t in conditioning:
            txt, keys = t
            keys = keys.copy()
            
            # 增强文本特征 - 使用更强的缩放方式
            if prompt_strength != 1.0:
                # 使用非线性缩放来增强文本特征
                txt_enhanced = txt * (prompt_strength ** 3)
                # 添加额外的文本特征副本来增强其影响力
                txt_repeated = txt_enhanced.repeat(1, 2, 1)  # 重复文本特征
                txt = txt_repeated
            
            # 处理风格特征
            if style_strength != 1.0:
                processed_cond = self.apply_style_strength(
                    cond, txt, style_strength, blend_mode
                )
            else:
                processed_cond = cond

            # 处理遮罩
            if mask is not None:
                feature_size = int(math.sqrt(processed_cond.shape[1]))
                processed_mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(1) if mask.dim() == 3 else mask,
                    size=(feature_size, feature_size),
                    mode='bilinear',
                    align_corners=False
                ).flatten(1).unsqueeze(-1)
                
                # 确保文本特征维度匹配
                if txt.shape[1] != processed_cond.shape[1]:
                    txt_mean = txt.mean(dim=1, keepdim=True)
                    txt_expanded = txt_mean.expand(-1, processed_cond.shape[1], -1)
                else:
                    txt_expanded = txt
                
                # 在遮罩区域使用处理后的特征，非遮罩区域使用原始文本特征
                processed_cond = processed_cond * processed_mask + \
                               txt_expanded * (1 - processed_mask)

            # 添加噪声
            if noise_level > 0:
                noise = torch.randn_like(processed_cond) * noise_level
                processed_cond = processed_cond + noise
                
            c_out.append([torch.cat((txt, processed_cond), dim=1), keys])
        
        return (c_out,)
    
    

