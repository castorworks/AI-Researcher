"""
Ollama client wrapper to replace litellm functionality
Completely compatible with litellm interface
"""
import json
import asyncio
from typing import List, Dict, Any, Optional, Union
import httpx
from dataclasses import dataclass
import re


@dataclass
class Message:
    """Message class compatible with litellm Message"""
    content: str
    role: str
    tool_calls: Optional[List] = None
    sender: Optional[str] = None
    
    def model_dump_json(self, **kwargs):
        """Convert to JSON string"""
        return json.dumps(self.__dict__, **kwargs)


@dataclass
class Choice:
    """Choice class compatible with litellm Choice"""
    message: Message
    index: int = 0
    finish_reason: str = "stop"


@dataclass
class CompletionResponse:
    """Completion response class compatible with litellm response"""
    choices: List[Choice]
    model: str = ""
    usage: Optional[Dict] = None


class OllamaClient:
    """Ollama client wrapper with full litellm compatibility"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, self_exc_type, self_exc_val, self_exc_tb):
        await self.client.aclose()
    
    def _convert_messages_to_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Convert OpenAI format messages to Ollama prompt"""
        prompt = ""
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
            elif role == "tool":
                prompt += f"Tool: {content}\n\n"
        
        return prompt.strip()
    
    def _convert_tools_to_ollama_format(self, tools: List[Dict]) -> str:
        """Convert tools to Ollama format string for function calling"""
        if not tools:
            return ""
        
        tool_descriptions = []
        for tool in tools:
            if "function" in tool:
                func = tool["function"]
                name = func.get("name", "")
                description = func.get("description", "")
                parameters = func.get("parameters", {})
                
                tool_desc = f"Tool: {name}\nDescription: {description}\n"
                if parameters:
                    tool_desc += f"Parameters: {json.dumps(parameters, indent=2)}\n"
                tool_descriptions.append(tool_desc)
        
        return "\n".join(tool_descriptions)
    
    def _extract_function_calls_from_response(self, response_text: str, tools: List[Dict]) -> List[Dict]:
        """Extract function calls from Ollama response text"""
        function_calls = []
        
        # Simple pattern matching for function calls
        # This is a basic implementation - in production you might want more sophisticated parsing
        
        # Look for patterns like: function_name(arg1="value1", arg2="value2")
        function_pattern = r'(\w+)\s*\(([^)]*)\)'
        matches = re.findall(function_pattern, response_text)
        
        for func_name, args_str in matches:
            # Parse arguments
            args = {}
            if args_str.strip():
                # Simple argument parsing - this could be enhanced
                arg_pairs = args_str.split(',')
                for pair in arg_pairs:
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"\'')
                        args[key] = value
            
            # Find matching tool
            tool_info = None
            for tool in tools:
                if tool.get("function", {}).get("name") == func_name:
                    tool_info = tool
                    break
            
            if tool_info:
                function_calls.append({
                    "id": f"call_{len(function_calls)}",
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": json.dumps(args)
                    }
                })
        
        return function_calls
    
    async def completion(self, **kwargs) -> CompletionResponse:
        """Synchronous completion (for compatibility)"""
        return await self.acompletion(**kwargs)
    
    async def acompletion(self, **kwargs) -> CompletionResponse:
        """Async completion method with full litellm compatibility"""
        model = kwargs.get("model", "llama3.2")
        messages = kwargs.get("messages", [])
        tools = kwargs.get("tools", [])
        stream = kwargs.get("stream", False)
        base_url = kwargs.get("base_url", self.base_url)
        tool_choice = kwargs.get("tool_choice", "auto")
        
        # Convert messages to prompt
        prompt = self._convert_messages_to_prompt(messages)
        
        # Add tools if provided and tool_choice is not "none"
        if tools and tool_choice != "none":
            tools_desc = self._convert_tools_to_ollama_format(tools)
            if tool_choice == "required":
                prompt += f"\n\n[IMPORTANT] You MUST use the tools provided to complete the task.\n\nAvailable tools:\n{tools_desc}\n\nPlease use the appropriate tools to complete the task."
            else:
                prompt += f"\n\nAvailable tools:\n{tools_desc}\n\nYou can use these tools if needed to complete the task."
        
        # Prepare Ollama request
        ollama_data = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "max_tokens": kwargs.get("max_tokens", 4096)
            }
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/api/generate",
                    json=ollama_data,
                    timeout=60.0
                )
                response.raise_for_status()
                
                result = response.json()
                response_text = result.get("response", "")
                
                # Extract function calls if tools are provided
                tool_calls = None
                if tools and tool_choice != "none":
                    tool_calls = self._extract_function_calls_from_response(response_text, tools)
                
                # Create response compatible with litellm
                message = Message(
                    content=response_text,
                    role="assistant",
                    tool_calls=tool_calls
                )
                
                choice = Choice(message=message)
                completion_response = CompletionResponse(
                    choices=[choice],
                    model=model
                )
                
                return completion_response
                
        except Exception as e:
            # Create error response compatible with litellm
            error_message = Message(
                content=f"Error: {str(e)}",
                role="assistant"
            )
            choice = Choice(message=error_message)
            return CompletionResponse(choices=[choice], model=model)


# Global client instance
_ollama_client = None

def get_ollama_client() -> OllamaClient:
    """Get global Ollama client instance"""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client


# Exception classes for compatibility
class ContextWindowExceededError(Exception):
    """Context window exceeded error"""
    pass


class BadRequestError(Exception):
    """Bad request error"""
    pass


class APIError(Exception):
    """API error"""
    pass


# Export completion functions for compatibility
def completion(**kwargs):
    """Synchronous completion function"""
    client = get_ollama_client()
    return asyncio.run(client.completion(**kwargs))


async def acompletion(**kwargs):
    """Async completion function"""
    client = get_ollama_client()
    return await client.acompletion(**kwargs)


# Additional compatibility functions
def supports_function_calling(model: str) -> bool:
    """Check if model supports function calling (for compatibility)"""
    # Ollama models generally support function calling through prompt engineering
    return True


# Set global configuration (for compatibility)
def set_verbose(verbose: bool):
    """Set verbose mode (for compatibility)"""
    pass


def set_num_retries(num_retries: int):
    """Set number of retries (for compatibility)"""
    pass
