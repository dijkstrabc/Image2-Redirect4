import torch
import numpy as np
from PIL import Image
import io
import urllib.request
import urllib.error
import json
import base64

# ==========================================
# 核心逻辑类：GPT 流式图像生成
# ==========================================
class GPTStreamImageNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 1. 基础配置
                "api_key": ("STRING", {"default": "sk-12345678"}),
                "endpoint": ("STRING", {"default": "http://3.34.136.247:8080/v1/images/generations"}),
                "model": ("STRING", {"default": "gpt-image-2"}),
                
                # 2. 尺寸控制：支持 8 像素步长调节
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 8}),
                
                # 3. 流式预览：partial_images 是模型在生成过程中返回的预览图数量
                "partial_images": ("INT", {"default": 2, "min": 0, "max": 10}),
                
                # 4. 连线输入：Prompt 放在最后，防止 ComfyUI 内部参数位移 Bug
                "prompt": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_stream"
    CATEGORY = "GPT Stream" # 修改了节点分类名

    def generate_stream(self, api_key, endpoint, model, width, height, partial_images, prompt):
        if not api_key:
            raise ValueError("错误：API Key 不能为空")

        # 构造 Payload，开启 stream 模式
        payload = {
            "model": model,
            "prompt": str(prompt),
            "size": f"{width}x{height}",
            "stream": True,
            "partial_images": partial_images,
            "response_format": "b64_json"
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Accept": "text/event-stream",
            "User-Agent": "ComfyUI-GPT-Stream-Client/2.0"
        }

        req = urllib.request.Request(
            endpoint, 
            data=json.dumps(payload).encode('utf-8'), 
            headers=headers,
            method='POST'
        )
        
        last_captured_b64 = None

        try:
            # 开启长连接流式读取
            with urllib.request.urlopen(req, timeout=150) as response:
                for line in response:
                    line_str = line.decode('utf-8').strip()
                    
                    # 仅解析 data: 开头的消息
                    if not line_str.startswith("data: "):
                        continue
                    
                    raw_data = line_str[6:]
                    if raw_data == "[DONE]":
                        break
                    
                    try:
                        event = json.loads(raw_data)
                        
                        # 逻辑：不断抓取流中出现的最新 b64 数据
                        # 兼容 partial_image 预览图和最终 data[0] 高清图
                        if "b64_json" in event:
                            last_captured_b64 = event["b64_json"]
                        elif "data" in event and isinstance(event["data"], list) and len(event["data"]) > 0:
                            if "b64_json" in event["data"][0]:
                                last_captured_b64 = event["data"][0]["b64_json"]
                                
                    except json.JSONDecodeError:
                        continue

        except urllib.error.HTTPError as e:
            err_body = e.read().decode('utf-8')
            raise RuntimeError(f"API 报错 HTTP {e.code}: {err_body}")
        except Exception as e:
            raise RuntimeError(f"连接异常: {str(e)}")

        if not last_captured_b64:
            raise RuntimeError("生成结束但未获取到图像。请检查模型名或 API Key 是否正确。")

        # 最终图像转换
        try:
            img_bytes = base64.b64decode(last_captured_b64)
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            
            # Tensor 转换
            img_np = np.array(pil_img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).unsqueeze(0)
            
            return (img_tensor,)
        except Exception as e:
            raise RuntimeError(f"图像渲染失败: {str(e)}")

# ==========================================
# 插件注册映射
# ==========================================
NODE_CLASS_MAPPINGS = {
    "GPTStreamImageNode": GPTStreamImageNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPTStreamImageNode": "GPT Stream Image Generator"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
