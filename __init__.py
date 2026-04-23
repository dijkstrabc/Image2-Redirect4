import torch
import numpy as np
from PIL import Image
import io
import urllib.request
import urllib.error
import json
import base64

# ==========================================
# 节点逻辑实现类
# ==========================================
class GPTImageCreateNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 1. 基础接口配置
                "api_key": ("STRING", {"default": "sk-12345678"}),
                "endpoint": ("STRING", {"default": "http://3.34.136.247:8080/v1/images/generations"}),
                "model": ("STRING", {"default": "gpt-image-2"}),
                
                # 2. 尺寸控制：遵循文档 Flexible sizes 特性，支持 8 像素步长调节
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 8}),
                
                # 3. 官方文档参数：质量与风格
                "quality": (["standard", "hd"], {"default": "standard"}),
                "style": (["vivid", "natural"], {"default": "vivid"}),
                
                # 4. 连线输入：强制文本输入点放在最后，彻底防止 ComfyUI 参数位移 Bug
                "prompt": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "Custom API/GPT"

    def generate(self, api_key, endpoint, model, width, height, quality, style, prompt):
        if not api_key:
            raise ValueError("错误：API Key 不能为空")
        
        if not prompt or str(prompt).strip() == "":
            raise ValueError("错误：输入的 Prompt 不能为空")

        # 根据 OpenAI 官方文档构造 Payload
        # size 格式必须为 '1024x1024' (小写x连接)
        payload = {
            "model": model,
            "prompt": str(prompt),
            "n": 1,
            "size": f"{width}x{height}",
            "quality": quality,
            "style": style,
            "response_format": "b64_json" # 优先使用 base64 传输，稳定性最高
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "ComfyUI-GPT-Image-v2"
        }

        # 转换数据并准备请求
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(endpoint, headers=headers, data=data)
        
        try:
            # 图像生成可能耗时，设置 120 秒超时
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode('utf-8')
            raise RuntimeError(f"API请求报错 HTTP {e.code}: {err_body}")
        except Exception as e:
            raise RuntimeError(f"网络连接失败: {str(e)}")

        # 处理 API 返回的业务错误
        if "error" in result:
            err_msg = result["error"].get("message", "未知 API 错误")
            raise RuntimeError(f"模型接口报错: {err_msg}")

        # 解析并加载图像数据
        try:
            image_item = result["data"][0]
            
            # 方案 A: 解析 Base64
            if "b64_json" in image_item:
                img_bytes = base64.b64decode(image_item["b64_json"])
            # 方案 B: 如果接口强行返回 URL
            elif "url" in image_item:
                img_url = image_item["url"]
                req_img = urllib.request.Request(img_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req_img, timeout=60) as response:
                    img_bytes = response.read()
            else:
                raise ValueError("API 返回数据中缺少图像内容 (b64_json/url)")
            
            # 将二进制流转为 PIL Image 再转为 ComfyUI Tensor
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            
            # 转换：(H, W, C) -> (0-1 float) -> (1, H, W, C)
            img_np = np.array(pil_img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).unsqueeze(0)
            
            return (img_tensor,)
            
        except Exception as e:
            raise RuntimeError(f"图像数据转换 Tensor 失败: {str(e)}")


# ==========================================
# ComfyUI 节点注册
# ==========================================
NODE_CLASS_MAPPINGS = {
    "GPTImageCreateNode": GPTImageCreateNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPTImageCreateNode": "GPT Image Create (OpenAI API)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
