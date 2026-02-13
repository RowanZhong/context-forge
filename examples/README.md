# Context Forge 场景示例

本目录包含 6 个完整的场景示例，演示 Context Forge 在不同生产场景中的应用。

## 示例列表

### 1. RAG 上下文质量治理 (`scenario_rag_quality.py`)

演示如何提升 RAG 应用的上下文质量：
- 10 个 RAG chunks（含噪音、重复、PII、过期数据）
- MMR 多样性去重
- PII 自动脱敏
- Prompt Injection 检测
- 时效性加权排序

**运行**：
```bash
python examples/scenario_rag_quality.py --mock
```

**章节映射**：→ 6.3.2 Select / → 6.4 上下文清洗与零信任安全

---

### 2. 多轮对话记忆管理 (`scenario_conversation_memory.py`)

演示如何在长对话中管理上下文：
- 20 轮对话历史
- must_keep 关键信息保护
- 滑动窗口策略
- 历史压缩
- 时效性衰减

**运行**：
```bash
python examples/scenario_conversation_memory.py --mock
```

**章节映射**：→ 6.2.4 压缩策略 / → 6.3.3 Compress

---

### 3. 多 Agent 上下文协调 (`scenario_multi_agent.py`)

演示如何在多 Agent 系统中管理上下文：
- 3 个 Agent：Planner → Executor → Reviewer
- Namespace 隔离
- Publish/Subscribe 模式
- Handoff 机制
- Visibility 控制

**运行**：
```bash
python examples/scenario_multi_agent.py --mock
```

**章节映射**：→ 6.3.1.1 Namespace Isolation / → 6.3.4 Isolate

---

### 4. 安全合规清洗 (`scenario_security_compliance.py`)

演示如何进行安全防护：
- 对抗性输入（HTML injection、Prompt injection、PII、Unicode tricks）
- 全套清洗器链
- 三级检测
- PII 脱敏
- 审计日志

**运行**：
```bash
python examples/scenario_security_compliance.py --mock
```

**章节映射**：→ 6.4 上下文清洗与零信任安全 / → 6.4.3 Injection 检测与防御

---

### 5. Prompt 版本管理与回归 (`scenario_versioning.py`)

演示如何进行 Prompt 版本管理：
- System Prompt v1 vs v2
- Snapshot 保存/加载
- 结构化 Diff
- Golden Set 回归测试
- Metrics 对比

**运行**：
```bash
python examples/scenario_versioning.py --mock
```

**章节映射**：→ 6.5.1 Context Snapshot / → 6.5.2 Prompt Diff / → 6.5.3 Golden Set 回归测试

---

### 6. 多模型适配与成本优化 (`scenario_routing_cost.py`)

演示如何进行智能路由和成本优化：
- 3 个查询（简单/中等/复杂）
- 复杂度估计
- 自动路由
- Budget 调整
- 成本对比

**运行**：
```bash
python examples/scenario_routing_cost.py --mock
```

**章节映射**：→ 6.6 上下文路由与动态调度 / → 6.2.2 预算分配策略

---

## 使用方法

### Mock 模式（默认，无需 API Key）

所有示例默认使用 Mock 模式，无需配置 API Key：

```bash
python examples/scenario_rag_quality.py
# 或明确指定
python examples/scenario_rag_quality.py --mock
```

### 真实 LLM 模式（需要 API Key）

如果要使用真实的 LLM API：

1. 设置环境变量：
```bash
export OPENAI_API_KEY=your_key
# 或
export ANTHROPIC_API_KEY=your_key
```

2. 运行示例：
```bash
python examples/scenario_rag_quality.py --no-mock
```

---

## 共享基础设施

所有示例共享 `_shared.py` 模块，提供：
- `MockLLM`：Mock LLM 实现
- Rich 输出美化工具
- 命令行参数解析
- 数据生成器

---

## 输出格式

每个示例使用 Rich 库美化输出，包括：
- 美化的表格（对比数据）
- 树形结构（层级关系）
- 面板（标题和摘要）
- 彩色文本（强调关键信息）

**注意**：Windows 终端中文可能显示乱码，但这不影响功能正确性。可以：
1. 使用 Windows Terminal（支持 UTF-8）
2. 运行 `chcp 65001` 切换代码页
3. 查看代码逻辑（示例已充分注释）

---

## 技术规格

- **Python 版本**：≥ 3.10
- **依赖**：
  - context-forge（核心库）
  - rich（输出美化）
  - asyncio（异步支持）
- **代码风格**：
  - 完整中文注释和 docstring
  - 单文件可运行
  - < 500 行（每个示例）
  - 完整错误处理

---

## 示例特点

1. **渐进式演示**：每个示例分 5-6 个步骤，循序渐进
2. **可视化输出**：使用表格、树形、面板等 Rich 组件
3. **决策透明**：展示审计日志，说明每一步的决策
4. **性能指标**：显示 Token 使用、耗时、成本等关键指标
5. **最佳实践**：代码中标注生产环境建议

---

## 故障排查

### 导入错误

如果遇到 `ModuleNotFoundError: No module named 'examples'`：

确保从项目根目录运行：
```bash
cd /path/to/context-forge
python examples/scenario_xxx.py
```

### 编码错误（Windows）

如果遇到 `UnicodeEncodeError`：

1. 使用 Windows Terminal
2. 或运行：`chcp 65001`

### API Key 未检测到

自动降级到 Mock 模式，会显示提示信息。

---

## 贡献

欢迎提交新的场景示例！格式要求：
- 单文件 < 500 行
- 支持 `--mock` 模式
- 完整中文注释
- Rich 美化输出
- 映射到书籍章节

---

## 许可证

Apache 2.0
