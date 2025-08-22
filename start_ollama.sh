#!/bin/bash

# Ollama 启动脚本
echo "Starting Ollama..."

# 检查 Ollama 是否已安装
if ! command -v ollama &> /dev/null; then
    echo "Ollama is not installed. Please install it first:"
    echo "Visit: https://ollama.ai/download"
    exit 1
fi

# 检查 Ollama 服务是否已经在运行
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Ollama service is already running."
else
    # 启动 Ollama 服务
    echo "Starting Ollama service..."
    ollama serve > /dev/null 2>&1 &
    OLLAMA_PID=$!
    
    # 等待服务启动
    echo "Waiting for Ollama service to start..."
    for i in {1..30}; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "Ollama service started successfully!"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "Failed to start Ollama service after 30 seconds"
            kill $OLLAMA_PID 2>/dev/null
            exit 1
        fi
        sleep 1
    done
fi

# 拉取默认模型
echo "Checking for default model: llama3.2"
if ! ollama list | grep -q "llama3.2"; then
    echo "Pulling default model: llama3.2"
    ollama pull llama3.2
else
    echo "Model llama3.2 is already available"
fi

# 检查服务状态
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "Ollama is running successfully on http://localhost:11434"
    echo "Available models:"
    ollama list
    
    echo ""
    echo "Ollama setup complete!"
    echo "You can now run your AI-Researcher application."
    echo ""
    echo "To use Ollama models, set these environment variables:"
    echo "export COMPLETION_MODEL=llama3.2"
    echo "export API_BASE_URL=http://localhost:11434"
    echo ""
    echo "Or create a .env file with:"
    echo "COMPLETION_MODEL=llama3.2"
    echo "API_BASE_URL=http://localhost:11434"
else
    echo "Failed to start Ollama service"
    exit 1
fi
