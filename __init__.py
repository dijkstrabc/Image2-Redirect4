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
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 8}),
                "partial_images": ("INT", {"default": 2, "min": 0, "max": 5}),
                
                # 依然放在最后，防止参数位移 Bug
                "prompt": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "Custom API/GPT"

    def generate(self, api_key, endpoint, model, width, height, partial_images, prompt):
        if not api_key:
            raise ValueError("API Key 不能为空")

        # 严格按照你的 curl 成功的参数构造 Payload
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
            "Accept": "text/event-stream", # 明确告知服务器我们要接收流
            "User-Agent": "ComfyUI-GPT-Client/1.0"
        }

        # 准备 POST 请求
        req = urllib.request.Request(
            endpoint, 
            data=json.dumps(payload).encode('utf-8'), 
            headers=headers,
            method='POST'
        )
        
        last_b64_data = None

        try:
            # 发起请求
            with urllib.request.urlopen(req, timeout=120) as response:
                # 核心：逐行读取流数据 (SSE 协议)
                for line in response:
                    line_str = line.decode('utf-8').strip()
                    
                    # 只有以 data: 开头的行才是有效数据
                    if not line_str.startswith("data: "):
                        continue
                    
                    # 提取数据内容
                    raw_data = line_str[6:]
                    if raw_data == "[DONE]":
                        break
                    
                    try:
                        event = json.loads(raw_data)
                        
                        # 情况 A: 官方示例中的中间预览图格式 (partial_image)
                        if "b64_json" in event:
                            last_b64_data = event["b64_json"]
                        
                        # 情况 B: 标准 OpenAI 最终图格式 (data[0].b64_json)
                        elif "data" in event and isinstance(event["data"], list) and len(event["data"]) > 0:
                            if "b64_json" in event["data"][0]:
                                last_b64_data = event["data"][0]["b64_json"]
                                
                    except json.JSONDecodeError:
                        continue

        except urllib.error.HTTPError as e:
            err_msg = e.read().decode('utf-8')
            raise RuntimeError(f"API 请求失败 {e.code}: {err_msg}")
        except Exception as e:
            raise RuntimeError(f"网络连接异常: {str(e)}")

        if not last_b64_data:
            raise RuntimeError("流式传输结束，但未能在响应中捕获到任何有效的图像数据 (b64_json)")

        # 将最终抓取到的 Base64 数据转为 Tensor
        try:
            img_bytes = base64.b64decode(last_b64_data)
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            
            # 转换为 (Batch, H, W, C) 的 Tensor 格式
            img_np = np.array(pil_img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).unsqueeze(0)
            
            return (img_tensor,)
        except Exception as e:
            raise RuntimeError(f"图像数据处理失败: {str(e)}")

# 节点注册
NODE_CLASS_MAPPINGS = {"GPTImageCreateNode": GPTImageCreateNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GPTImageCreateNode": "GPT Image Create (Official Stream)"}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
