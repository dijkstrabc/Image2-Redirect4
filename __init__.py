import torch
import numpy as np
from PIL import Image
import io
import urllib.request
import urllib.error
import json
import base64

class GPTImageCreateNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "sk-12345678"}),
                "endpoint": ("STRING", {"default": "http://3.34.136.247:8080/v1/images/generations"}),
                "model": ("STRING", {"default": "gpt-image-2"}),
                
                # 尺寸控制：根据你反馈的 Flexible sizes 特性，保留自定义调节
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 8}),
                
                # 官方文档提到的高级参数
                "quality": (["standard", "hd"], {"default": "standard"}),
                "style": (["vivid", "natural"], {"default": "vivid"}),
                
                # 提示词连线
                "prompt": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "Custom API/GPT"

    def generate(self, api_key, endpoint, model, width, height, quality, style, prompt):
        if not api_key:
            raise ValueError("API Key 不能为空")
        
        # 组装符合官方文档规范的 Payload
        # 注意：size 必须是 '宽x高' 的格式
        payload = {
            "model": model,
            "prompt": str(prompt),
            "n": 1,
            "size": f"{width}x{height}",
            "quality": quality,
            "style": style,
            "response_format": "b64_json"
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "ComfyUI-GPT-Image-v2"
        }

        req = urllib.request.Request(endpoint, headers=headers, data=json.dumps(payload).encode('utf-8'))
        
        try:
            # 官方文档提示生图可能较慢，设置 120 秒超时
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode('utf-8')
            raise RuntimeError(f"API请求报错 {e.code}: {err_msg}")
        except Exception as e:
            raise RuntimeError(f"网络请求失败: {str(e)}")

        # 错误处理逻辑（遵循官方 Error Object 结构）
        if "error" in result:
            err_msg = result["error"].get("message", "未知错误")
            raise RuntimeError(f"接口业务报错: {err_msg}")

        # 解析并加载图像
        try:
            image_data = result["data"][0]
            
            # 优先使用 Base64 解析，速度最快且最稳定
            if "b64_json" in image_data:
                img_bytes = base64.b64decode(image_data["b64_json"])
            elif "url" in image_data:
                img_url = image_data["url"]
                req_img = urllib.request.Request(img_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req_img, timeout=60) as response:
                    img_bytes = response.read()
            else:
                raise ValueError("未在响应中找到图像数据")
            
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            
            # 转换为 ComfyUI 标准的 Tensor 格式 (Batch, H, W, C)
            img_np = np.array(pil_img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).unsqueeze(0)
            
            return (img_tensor,)
            
        except Exception as e:
            raise RuntimeError(f"图像数据转换失败: {str(e)}")