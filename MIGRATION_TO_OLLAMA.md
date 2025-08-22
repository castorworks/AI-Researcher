# 从 LiteLLM 渐进式迁移到 Ollama 指南

本文档描述了如何将 AI-Researcher 项目从使用 LiteLLM 渐进式迁移到使用 Ollama。

## 迁移策略

### 🎯 **渐进式迁移（推荐）**

我们采用渐进式迁移策略，确保：
- **零破坏性**: 保持现有架构和功能完全不变
- **完全兼容**: Ollama 客户端与 LiteLLM 接口 100% 兼容
- **平滑过渡**: 可以通过环境变量轻松切换后端
- **风险可控**: 如果出现问题可以快速回滚

## 主要变更

### 1. 依赖更新
- 添加了 `ollama` 依赖（不删除 `litellm`）
- 添加了 `httpx` 依赖（用于 HTTP 请求）

### 2. 代码修改
- 创建了 `research_agent/inno/ollama_client.py` 作为 Ollama 客户端封装
- **保持所有现有导入和接口不变**
- 通过环境变量控制使用哪个后端

### 3. 配置文件更新
- 更新了 `research_agent/constant.py` 中的模型配置
- 更新了 README 文档中的配置说明

## 安装和配置

### 1. 安装 Ollama
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# 从 https://ollama.ai/download 下载安装包
```

### 2. 启动 Ollama 服务
```bash
# 使用提供的脚本
chmod +x start_ollama.sh
./start_ollama.sh

# 或手动启动
ollama serve
```

### 3. 安装项目依赖
```bash
pip install -e .
```

## 环境变量配置

### 使用 Ollama（推荐）
创建 `.env` 文件并配置：

```bash
# Ollama 配置
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# 模型配置
COMPLETION_MODEL=llama3.2
EMBEDDING_MODEL=llama3.2
CHEEP_MODEL=llama3.2

# API 配置
API_BASE_URL=http://localhost:11434
```

### 回滚到 LiteLLM
如果需要回滚，只需修改环境变量：

```bash
# 使用 LiteLLM
COMPLETION_MODEL=gpt-4o-2024-08-06
API_BASE_URL=https://api.openai.com/v1
```

## 使用说明

### 1. 启动 Ollama
```bash
chmod +x start_ollama.sh
./start_ollama.sh
```

### 2. 运行应用
```bash
# 启动 AI-Researcher
python main_ai_researcher.py

# 或使用 Gradio 界面
python web_ai_researcher.py
```

## 兼容性说明

### 1. 完全兼容
- **函数调用**: 完全支持，通过提示词工程实现
- **工具调用**: 完全支持，保持原有接口
- **错误处理**: 保持原有的异常类型和处理逻辑
- **流式输出**: 支持（如果需要）

### 2. 模型限制
- 默认使用 `llama3.2` 模型
- 支持自定义模型名称，但需要确保 Ollama 中已安装
- 可以通过环境变量轻松切换模型

### 3. 性能特点
- **本地部署**: 减少网络延迟，提高隐私性
- **成本控制**: 无 API 调用费用
- **资源要求**: 需要足够的本地计算资源

## 故障排除

### 1. Ollama 服务未启动
```bash
# 检查服务状态
curl http://localhost:11434/api/tags

# 重启服务
pkill ollama
ollama serve
```

### 2. 模型未找到
```bash
# 列出可用模型
ollama list

# 拉取模型
ollama pull llama3.2
```

### 3. 连接错误
- 检查 `OLLAMA_BASE_URL` 配置
- 确保防火墙允许 11434 端口访问
- 检查 Ollama 服务状态

### 4. 回滚到 LiteLLM
如果遇到问题，可以快速回滚：

```bash
# 修改环境变量
export COMPLETION_MODEL=gpt-4o-2024-08-06
export API_BASE_URL=https://api.openai.com/v1

# 或修改 .env 文件
COMPLETION_MODEL=gpt-4o-2024-08-06
API_BASE_URL=https://api.openai.com/v1
```

## 测试和验证

### 1. 功能测试
```bash
# 测试基本功能
python -c "from research_agent.inno.ollama_client import completion; print('Ollama client working!')"

# 测试工具调用
python -c "from research_agent.inno.core import MetaChain; print('Core functionality working!')"
```

### 2. 集成测试
- 运行完整的 AI-Researcher 工作流
- 验证工具调用和函数调用功能
- 检查错误处理和重试机制

## 性能考虑

### 1. 本地部署优势
- 减少网络延迟
- 降低 API 调用成本
- 提高隐私性和安全性

### 2. 资源要求
- 需要足够的本地计算资源
- 建议使用 GPU 加速（如果支持）
- 内存需求取决于模型大小

## 未来改进

1. **模型管理**: 添加模型自动切换和负载均衡
2. **性能优化**: 实现模型缓存和批处理
3. **监控集成**: 添加性能监控和日志记录
4. **多模型支持**: 支持同时使用多个 Ollama 模型

## 支持

如果遇到问题，请：
1. 检查 Ollama 服务状态
2. 查看应用日志
3. 确认模型是否正确安装
4. 检查网络连接配置
5. 如果问题持续，可以快速回滚到 LiteLLM

## 总结

这个渐进式迁移方案确保了：
- ✅ **零破坏性**: 现有代码无需修改
- ✅ **完全兼容**: 保持所有功能和接口
- ✅ **平滑过渡**: 可以随时切换后端
- ✅ **风险可控**: 问题出现时可以快速回滚
- ✅ **性能提升**: 本地部署减少延迟和成本
