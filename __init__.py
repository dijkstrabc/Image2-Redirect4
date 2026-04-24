import torch
import numpy as np
from PIL import Image
import io
import urllib.request
import urllib.error
import json
import base64

# ==========================================
# 核心逻辑类：GPT Codex Response (v1/responses)
# ==========================================
class GPTCodexResponseNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 1. 基础配置
                "api_key": ("STRING", {"default": "sk-12345678"}),
                # 注意：Codex Response API 的标准端点通常是 /v1/responses
                "endpoint": ("STRING", {"default": "http://3.34.136.247:8080/v1/responses"}),
                "model": ("STRING", {"default": "gpt-image-2"}),
                
                # 2. 尺寸控制：支持 8 像素步长调节
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 8}),
                
                # 3. 连线输入：Prompt 放在最后
                "prompt": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "call_codex_api"
    CATEGORY = "GPT Codex"

    def call_codex_api(self, api_key, endpoint, model, width, height, prompt):
        if not api_key:
            raise ValueError("错误：API Key 不能为空")

        # 构造 Codex Response API 专用 Payload
        # 根据官方 Responses API 规范，input 可以直接是字符串，也可以是消息数组
        payload = {
            "model": model,
            "input": str(prompt), # 简化输入格式
            "size": f"{width}x{height}",
            "stream": True,
            "response_format": "b64_json"
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key.strip()}",
            "Accept": "text/event-stream"
        }

        # 强制不走代理，确保内网/特定IP连通性
        proxy_handler = urllib.request.ProxyHandler({})
        opener = urllib.request.build_opener(proxy_handler)
        
        req = urllib.request.Request(
            endpoint, 
            data=json.dumps(payload).encode('utf-8'), 
            headers=headers,
            method='POST'
        )
        
        last_captured_b64 = None
        print(f"[GPT-Codex] 发送请求到: {endpoint}")

        try:
            with opener.open(req, timeout=150) as response:
                print("[GPT-Codex] 成功连接，解析 Responses 流...")
                for line in response:
                    line_str = line.decode('utf-8').strip()
                    
                    if not line_str.startswith("data: "):
                        continue
                    
                    raw_data = line_str[6:]
                    if raw_data == "[DONE]":
                        break
                    
                    try:
                        event = json.loads(raw_data)
                        
                        # Codex Response API 的数据解析逻辑：
                        # 1. 查找 content_block 事件中的图像
                        if "type" in event:
                            # 最终图通常在 content_block.stop 或类似结构中
                            if event["type"] == "content_block" and "image" in event.get("content_block", {}):
                                img_info = event["content_block"]["image"]
                                if "b64_json" in img_info:
                                    last_captured_b64 = img_info["b64_json"]
                            
                            # 2. 兼容旧的流式 delta 格式
                            elif "delta" in event and "image" in event["delta"]:
                                last_captured_b64 = event["delta"]["image"].get("b64_json")

                        # 3. 如果是简单的 data[0] 结构也支持
                        elif "data" in event and isinstance(event["data"], list):
                            last_captured_b64 = event["data"][0].get("b64_json")
                            
                    except Exception:
                        continue

        except urllib.error.HTTPError as e:
            err_body = e.read().decode('utf-8')
            print(f"[GPT-Codex] API 错误: {err_body}")
            raise RuntimeError(f"API 报错 HTTP {e.code}: {err_body}")
        except Exception as e:
            raise RuntimeError(f"连接异常: {str(e)}")

        if not last_captured_b64:
            raise RuntimeError("API 响应结束，但未捕获到有效的图像数据。请检查端点是否为 /v1/responses。")

        # 最终图像转换并输出到 ComfyUI
        try:
            img_bytes = base64.b64decode(last_captured_b64)
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_np = np.array(pil_img).astype(np.float32) / 255.0
            return (torch.from_numpy(img_np).unsqueeze(0),)
        except Exception as e:
            raise RuntimeError(f"图像渲染失败: {str(e)}")

# ==========================================
# 插件注册
# ==========================================
NODE_CLASS_MAPPINGS = {
    "GPTCodexResponseNode": GPTCodexResponseNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPTCodexResponseNode": "GPT Codex Response Generator"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
