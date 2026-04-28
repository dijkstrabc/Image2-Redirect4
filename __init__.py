import torch
import numpy as np
from PIL import Image
import io
import urllib.request
import urllib.error
import json
import base64

# ==========================================
# 辅助函数：将 ComfyUI Tensor 转换为 Base64
# ==========================================
def tensor_to_base64(tensor):
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    i = 255. * tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

# ==========================================
# 节点 1：GPT 图像生成 (纯文生图)
# ==========================================
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
                # 修正：将调节步长修改为 16
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
                
                # 提示词连线项
                "prompt": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "GPT Image"

    def generate(self, api_key, endpoint, model, width, height, prompt):
        payload = {
            "model": model,
            "prompt": str(prompt),
            "size": f"{width}x{height}",
            "stream": False,
            "response_format": "b64_json"
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key.strip()}"
        }

        req = urllib.request.Request(endpoint, data=json.dumps(payload).encode('utf-8'), headers=headers, method='POST')
        
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode('utf-8'))
                if "data" in result and len(result["data"]) > 0:
                    b64_data = result["data"][0].get("b64_json")
                    img_bytes = base64.b64decode(b64_data)
                    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    img_np = np.array(pil_img).astype(np.float32) / 255.0
                    return (torch.from_numpy(img_np).unsqueeze(0),)
        except Exception as e:
            raise RuntimeError(f"GPT Image 生成失败: {str(e)}")

# ==========================================
# 节点 2：GPT 图像编辑 (支持 1-3 张参考图)
# ==========================================
class GPTImageEditNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",), # 必选第一张
                "api_key": ("STRING", {"default": "sk-12345678"}),
                "endpoint": ("STRING", {"default": "http://3.34.136.247:8080/v1/images/edits"}),
                "model": ("STRING", {"default": "gpt-image-2"}),
                # 修正：将调节步长修改为 16
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
                
                "prompt": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "image2": ("IMAGE",), # 可选第二张
                "image3": ("IMAGE",), # 可选第三张
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "edit"
    CATEGORY = "GPT Image"

    def edit(self, image1, api_key, endpoint, model, width, height, prompt, image2=None, image3=None):
        # 组装图片数组
        image_list = []
        image_list.append(tensor_to_base64(image1))
        
        if image2 is not None:
            image_list.append(tensor_to_base64(image2))
        
        if image3 is not None:
            image_list.append(tensor_to_base64(image3))

        payload = {
            "model": model,
            "images": image_list, # 发送数组格式
            "prompt": str(prompt),
            "size": f"{width}x{height}",
            "response_format": "b64_json"
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key.strip()}"
        }

        req = urllib.request.Request(endpoint, data=json.dumps(payload).encode('utf-8'), headers=headers, method='POST')
        
        try:
            with urllib.request.urlopen(req, timeout=150) as response:
                result = json.loads(response.read().decode('utf-8'))
                if "data" in result and len(result["data"]) > 0:
                    b64_data = result["data"][0].get("b64_json")
                    img_bytes = base64.b64decode(b64_data)
                    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    img_np = np.array(pil_img).astype(np.float32) / 255.0
                    return (torch.from_numpy(img_np).unsqueeze(0),)
        except Exception as e:
            raise RuntimeError(f"GPT Image 编辑失败: {str(e)}")

# ==========================================
# 插件注册映射
# ==========================================
NODE_CLASS_MAPPINGS = {
    "GPTImageCreateNode": GPTImageCreateNode,
    "GPTImageEditNode": GPTImageEditNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPTImageCreateNode": "GPT Image Generator",
    "GPTImageEditNode": "GPT Image Editor"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
