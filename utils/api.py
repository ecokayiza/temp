from openai import OpenAI
import os
import json
from typing import Optional, Generator, Union, Dict, Any

import base64

class ChatClient:
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 base_url: Optional[str] = None, 
                 model: str = "gemini-3-pro-preview"):
        
        # 优先使用传入参数，其次环境变量，最后使用默认值

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or "https://api.chataiapi.com/v1"
        self.model = model
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        self.history = []

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def clear_history(self):
        """清空对话历史"""
        self.history = []

    def chat(self, 
             prompt: str, 
             system_prompt: Optional[str] = None,
             image_path: Optional[str] = None,
             json_format: bool = False,
             stream: bool = False,
             temperature: float = 0.7,
             include_reasoning: bool = False,
             **kwargs) -> Union[str, Dict[str, str], Generator[Union[str, Dict[str, str]], None, None]]:
        """
        多轮对话模式。会自动维护 self.history。
        """
        # 1. 如果历史为空且提供了 system_prompt，则初始化
        if not self.history and system_prompt:
            self.history.append({"role": "system", "content": system_prompt})
            
        # 2. 处理 JSON 提示增强
        current_prompt = prompt
        if json_format and "json" not in prompt.lower():
             current_prompt += " Please respond in valid JSON format."

        # 3. 构造用户消息
        if image_path:
            base64_image = self._encode_image(image_path)
            content = [
                {"type": "text", "text": current_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
            self.history.append({"role": "user", "content": content})
        else:
            self.history.append({"role": "user", "content": current_prompt})

        # 4. 构造请求参数
        response_format = {"type": "json_object"} if json_format else None

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                stream=stream,
                temperature=temperature,
                response_format=response_format,
                **kwargs
            )

            if stream:
                # 定义回调函数，在流结束后保存完整回复到历史
                def save_history(full_content):
                    self.history.append({"role": "assistant", "content": full_content})
                
                return self._stream_response(response, include_reasoning, on_complete=save_history)
            else:
                message = response.choices[0].message
                content = message.content
                
                # 保存回复到历史
                self.history.append({"role": "assistant", "content": content})
                
                if include_reasoning:
                    reasoning = getattr(message, 'reasoning_content', None) or ""
                    return {"content": content, "reasoning": reasoning}
                return content

        except Exception as e:
            print(f"Chat Request Failed: {e}")
            raise

    def generate(self, 
                 prompt: str, 
                 system_prompt: str = "You are a helpful assistant.",
                 image_path: Optional[str] = None,
                 json_format: bool = False,
                 stream: bool = False,
                 temperature: float = 0.7,
                 include_reasoning: bool = False,
                 **kwargs) -> Union[str, Dict[str, str], Generator[Union[str, Dict[str, str]], None, None]]:
        """
        调用 LLM 生成回复。
        
        Args:
            prompt (str): 用户输入的提示词。
            system_prompt (str): 系统提示词，默认为通用助手。
            image_path (str): 图片路径，如果提供则进行多模态调用。
            json_format (bool): 是否强制返回 JSON 格式 (需要模型支持)。
            stream (bool): 是否开启流式输出。
            temperature (float): 随机性参数。
            include_reasoning (bool): 是否返回思考过程 (reasoning_content)。
            **kwargs: 传递给 client.chat.completions.create 的其他参数。
            
        Returns:
            str: (stream=False, include_reasoning=False) 返回内容字符串。
            dict: (stream=False, include_reasoning=True) 返回 {"content": "...", "reasoning": "..."}。
            Generator: (stream=True) 
                - include_reasoning=False: yield content_string
                - include_reasoning=True: yield {"type": "reasoning"|"content", "content": "..."}
        """
        
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        # 如果开启 JSON 模式，构造 response_format 参数
        response_format = {"type": "json_object"} if json_format else None
        
        # 提示增强：JSON 模式下通常需要在 Prompt 中显式提及 JSON
        if json_format and "json" not in prompt.lower() and "json" not in system_prompt.lower():
            prompt += " Please respond in valid JSON format."

        if image_path:
            base64_image = self._encode_image(image_path)
            messages.append({
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            })
        else:
            messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=stream,
                temperature=temperature,
                response_format=response_format,
                **kwargs
            )

            if stream:
                return self._stream_response(response, include_reasoning)
            else:
                message = response.choices[0].message
                content = message.content
                if include_reasoning:
                    reasoning = getattr(message, 'reasoning_content', None) or ""
                    return {"content": content, "reasoning": reasoning}
                return content

        except Exception as e:
            print(f"API Request Failed: {e}")
            raise

    def _stream_response(self, response, include_reasoning: bool, on_complete=None) -> Generator[Union[str, Dict[str, str]], None, None]:
        """处理流式响应的生成器"""
        full_content = ""
        
        for chunk in response:
            # 跳过无效块
            if not chunk.choices:
                continue
                
            delta = chunk.choices[0].delta
            
            if include_reasoning:
                # 尝试获取思考过程
                reasoning = getattr(delta, 'reasoning_content', None)
                if reasoning:
                    yield {"type": "reasoning", "content": reasoning}
                
                if delta.content:
                    full_content += delta.content
                    yield {"type": "content", "content": delta.content}
            else:
                # 保持原有行为，只返回内容字符串
                if delta.content:
                    full_content += delta.content
                    yield delta.content
        
        # 流结束，执行回调
        if on_complete:
            on_complete(full_content)

# ==========================================
# 测试代码 (直接运行此文件可测试)
# ==========================================
if __name__ == "__main__":
    client = ChatClient()
    
    print("\n--- 连通测试 (Chat History) ---")
    try:
        client.clear_history()
        print("User: Who are you?")
        res1 = client.chat("Who are you?")
        print(f"AI: {res1}")            
    except Exception as e:
        print(f"测试失败: {e}")