# 贡献指南

感谢您考虑为 Context Forge 做出贡献！

本文档提供了贡献代码、报告问题和提出功能请求的指南。

---

## 目录

- [行为准则](#行为准则)
- [如何贡献](#如何贡献)
- [开发环境搭建](#开发环境搭建)
- [开发流程](#开发流程)
- [代码规范](#代码规范)
- [测试要求](#测试要求)
- [提交指南](#提交指南)
- [Pull Request 流程](#pull-request-流程)
- [文档贡献](#文档贡献)
- [问题反馈](#问题反馈)

---

## 行为准则

### 我们的承诺

为了营造一个开放和欢迎的环境，我们承诺让参与我们的项目和社区的每个人都能获得无骚扰的体验，无论年龄、体型、残疾、种族、性别认同和表达、经验水平、教育程度、社会经济地位、国籍、个人外观、种族、宗教或性认同和取向如何。

### 我们的标准

**积极行为示例**：

- 使用欢迎和包容的语言
- 尊重不同的观点和经验
- 优雅地接受建设性批评
- 关注对社区最有利的事情
- 对其他社区成员表示同理心

**不可接受的行为示例**：

- 使用性化语言或图像，以及不受欢迎的性关注或调情
- 煽动性/贬损性评论，以及人身或政治攻击
- 公开或私下骚扰
- 未经明确许可发布他人的私人信息
- 在专业场合可被合理视为不适当的其他行为

### 执行

违反行为准则的行为可向项目团队报告。所有投诉都将被审查和调查，并将做出被认为必要和适当的回应。

---

## 如何贡献

### 报告 Bug

在提交 Bug 报告之前，请：

1. **搜索已有 Issue**：确保问题尚未被报告
2. **确认可复现**：在最新版本上验证问题
3. **简化问题**：提供最小可复现示例

**Bug 报告应包含**：

- **标题**：简洁描述问题
- **环境信息**：
  - Python 版本
  - Context Forge 版本
  - 操作系统和版本
- **复现步骤**：详细的步骤列表
- **预期行为**：应该发生什么
- **实际行为**：实际发生了什么
- **错误信息**：完整的错误堆栈
- **最小示例**：可直接运行的代码

**示例**：

```markdown
### Bug 描述
在使用 RAG 场景时，当 rag_chunks 超过 100 个时，组装失败并抛出 BudgetExceededError。

### 环境信息
- Python: 3.12.0
- Context Forge: 0.1.0
- OS: Windows 11

### 复现步骤
1. 创建包含 150 个 chunk 的 rag_chunks 列表
2. 调用 forge.build(..., rag_chunks=chunks)
3. 观察错误

### 预期行为
应该自动截断低优先级的 chunk，而不是抛出错误。

### 实际行为
抛出 BudgetExceededError

### 错误信息
\`\`\`
BudgetExceededError: 预算超限：需要 250000 Token，但上下文窗口仅支持 128000 Token
\`\`\`

### 最小示例
\`\`\`python
from context_forge import ContextForge

forge = ContextForge(model="gpt-4o")
chunks = [{"content": f"Document {i}", "score": 0.9} for i in range(150)]
context = await forge.build(rag_chunks=chunks)  # 抛出错误
\`\`\`
```

### 功能请求

**功能请求应包含**：

- **使用场景**：为什么需要这个功能
- **提议方案**：具体如何实现
- **替代方案**：考虑过的其他方法
- **影响范围**：对现有功能的影响

### 提问

对于使用问题，请：

1. 优先查阅 [文档](https://context-forge.github.io)
2. 搜索 [GitHub Discussions](https://github.com/context-forge/context-forge/discussions)
3. 如果仍无答案，在 Discussions 中提问

---

## 开发环境搭建

### 前置要求

- **Python 3.10+**（推荐 3.12）
- **Git**
- **Redis**（可选，用于缓存测试）

### 快速搭建

```bash
# 克隆仓库
git clone https://github.com/context-forge/context-forge.git
cd context-forge

# 运行自动化脚本
# Linux / macOS
bash scripts/setup_dev.sh

# Windows
powershell -ExecutionPolicy Bypass -File scripts\setup_dev.ps1
```

### 手动搭建

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装 uv
pip install uv

# 安装依赖
uv pip install -e ".[dev,docs]"

# 安装 pre-commit 钩子
pre-commit install

# 验证安装
pytest tests/ -v
```

---

## 开发流程

### 1. Fork 和克隆

```bash
# Fork 项目到你的账号（在 GitHub 网页操作）

# 克隆你的 Fork
git clone https://github.com/YOUR_USERNAME/context-forge.git
cd context-forge

# 添加上游仓库
git remote add upstream https://github.com/context-forge/context-forge.git
```

### 2. 创建分支

```bash
# 确保在最新的 main 分支
git checkout main
git pull upstream main

# 创建特性分支
git checkout -b feature/your-feature-name

# 或者 bugfix 分支
git checkout -b bugfix/issue-number-description
```

### 3. 开发

```bash
# 激活虚拟环境
source .venv/bin/activate

# 开发你的功能

# 运行测试
make test

# 检查代码风格
make lint

# 类型检查
make typecheck
```

### 4. 提交

```bash
# 暂存变更
git add .

# 提交（pre-commit 钩子会自动运行）
git commit -m "feat: add amazing feature"

# 如果 pre-commit 失败，修复后重新提交
git add .
git commit -m "feat: add amazing feature"
```

### 5. 推送

```bash
# 推送到你的 Fork
git push origin feature/your-feature-name
```

### 6. 创建 Pull Request

在 GitHub 网页上创建 Pull Request。

---

## 代码规范

### Python 风格

我们遵循 **PEP 8** 和 **Google Python Style Guide**。

**关键规则**：

- **缩进**：4 个空格
- **行长度**：最大 100 字符
- **引号**：字符串统一使用双引号 `"`
- **导入顺序**：标准库 → 第三方库 → 本地模块
- **类型标注**：所有公共 API 必须有类型标注
- **Docstring**：所有公共函数、类、模块必须有 Docstring

**示例**：

```python
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MyClass(BaseModel):
    """类的简短描述。

    详细描述（可选）。

    属性:
        field_name: 字段描述
    """

    field_name: str = Field(..., description="字段描述")

    def my_method(self, param: str) -> dict[str, Any]:
        """方法的简短描述。

        参数:
            param: 参数描述

        返回:
            返回值描述

        异常:
            ValueError: 异常描述
        """
        return {"result": param}
```

### 不可变性（CRITICAL）

**始终创建新对象，绝不修改现有对象**：

```python
# ❌ 错误：原地修改
def update_segment(segment: Segment, new_content: str) -> Segment:
    segment.content = new_content  # 错误！
    return segment

# ✅ 正确：返回新对象
def update_segment(segment: Segment, new_content: str) -> Segment:
    return segment.model_copy(update={"content": new_content})
```

### 错误处理

**三段式错误信息**：What（发生了什么）+ Why（为什么）+ How（怎么修）

```python
# ❌ 错误：信息不完整
raise ValueError("Invalid input")

# ✅ 正确：完整的错误信息
raise ValidationError(
    "Segment 内容为空（What）。"
    "所有 Segment 必须包含非空的 content 字段（Why）。"
    "请检查输入数据，确保 content 字段不为空字符串（How）。"
)
```

### 代码注释

**使用教学标注**：

```python
# [Design Decision] 使用 frozen=True 确保不可变性
@dataclass(frozen=True)
class Segment:
    content: str

# → 6.1.1.2 Segment 数据模型
# 映射到书籍章节

# 🏭 生产提示：此处省略了分布式锁实现，生产环境需补充

# ⚠️ 反模式：不要直接修改 Segment 内容
# segment.content = "new"  # 这会导致运行时错误
```

---

## 测试要求

### 最低覆盖率：85%

所有新代码必须包含测试，覆盖率不得低于 85%。

### 测试类型

1. **单元测试**：测试单个函数/类
2. **集成测试**：测试多个模块协作
3. **端到端测试**：测试完整流程

### 测试文件结构

```
tests/
├── unit/              # 单元测试
│   ├── test_segment.py
│   ├── test_pipeline.py
│   └── ...
├── integration/       # 集成测试
│   ├── test_full_pipeline.py
│   └── ...
└── e2e/              # 端到端测试
    ├── test_rag_scenario.py
    └── ...
```

### 编写测试

```python
import pytest
from context_forge import ContextForge
from context_forge.models import Segment, SegmentType


class TestSegment:
    """Segment 模型的单元测试。"""

    def test_create_segment(self):
        """测试创建基本 Segment。"""
        segment = Segment(
            content="Test content",
            segment_type=SegmentType.USER,
        )

        assert segment.content == "Test content"
        assert segment.segment_type == SegmentType.USER

    def test_segment_immutability(self):
        """测试 Segment 不可变性。"""
        segment = Segment(content="Original", segment_type=SegmentType.USER)

        with pytest.raises(Exception):  # Pydantic frozen 会抛出异常
            segment.content = "Modified"

    @pytest.mark.asyncio
    async def test_segment_in_pipeline(self):
        """测试 Segment 在 Pipeline 中的处理。"""
        forge = ContextForge(model="gpt-4o")
        context = await forge.build(
            messages=[{"role": "user", "content": "Test"}]
        )

        assert len(context.segments) > 0
        assert context.segments[0].segment_type == SegmentType.USER
```

### 运行测试

```bash
# 运行所有测试
make test

# 运行特定文件
pytest tests/unit/test_segment.py -v

# 运行特定测试
pytest tests/unit/test_segment.py::TestSegment::test_create_segment -v

# 查看覆盖率
make test-cov
```

---

## 提交指南

### Conventional Commits

我们使用 [Conventional Commits](https://www.conventionalcommits.org/) 规范。

**格式**：

```
<type>: <description>

<optional body>

<optional footer>
```

**类型（type）**：

- `feat`: 新功能
- `fix`: Bug 修复
- `refactor`: 重构（不改变功能）
- `docs`: 文档变更
- `test`: 测试相关
- `chore`: 构建/工具链变更
- `perf`: 性能优化
- `ci`: CI/CD 变更

**示例**：

```bash
# 新功能
git commit -m "feat: 添加语义缓存支持"

# Bug 修复
git commit -m "fix: 修复预算分配时的整数溢出问题"

# 重构
git commit -m "refactor: 简化 Pipeline 阶段接口"

# 文档
git commit -m "docs: 更新 RAG 场景示例"

# 带详细说明
git commit -m "feat: 添加多 Agent 上下文协调

实现了 Context Handoff 机制，支持：
- Agent 间上下文传递
- Segment 级别的权限控制
- Namespace 隔离

Closes #123"
```

---

## Pull Request 流程

### PR 标题

使用与提交相同的 Conventional Commits 格式：

```
feat: 添加语义缓存支持
fix: 修复预算分配整数溢出
```

### PR 描述

使用以下模板：

```markdown
## 概述
简短描述此 PR 的目的。

## 变更类型
- [ ] Bug 修复
- [ ] 新功能
- [ ] 破坏性变更
- [ ] 文档更新
- [ ] 重构
- [ ] 性能优化
- [ ] 测试

## 变更内容
- 变更 1
- 变更 2
- 变更 3

## 测试
- [ ] 所有现有测试通过
- [ ] 添加了新测试（如适用）
- [ ] 覆盖率 >= 85%

## 检查清单
- [ ] 代码符合风格指南（`make lint` 通过）
- [ ] 类型检查通过（`make typecheck` 通过）
- [ ] 所有测试通过（`make test` 通过）
- [ ] 文档已更新（如适用）
- [ ] CHANGELOG.md 已更新（如适用）

## 相关 Issue
Closes #123
```

### PR 审查

**审查者会检查**：

1. **代码质量**：是否符合规范
2. **测试覆盖率**：是否 >= 85%
3. **文档完整性**：是否更新了文档
4. **向后兼容性**：是否破坏现有 API
5. **性能影响**：是否有性能回退

**审查反馈**：

- 请及时响应审查意见
- 所有讨论解决后，审查者会批准 PR
- 维护者会合并 PR

---

## 文档贡献

### 文档类型

1. **API 文档**：位于代码 Docstring 中
2. **用户指南**：位于 `docs/` 目录
3. **示例代码**：位于 `examples/` 目录
4. **README.md**：项目主页

### 文档规范

- **语言**：全部中文
- **格式**：Markdown
- **代码示例**：必须可运行
- **图表**：使用 Mermaid

### 构建文档

```bash
# 安装文档依赖
pip install -e ".[docs]"

# 构建文档
make docs

# 预览文档
make docs-serve
# 访问 http://localhost:8000
```

---

## 问题反馈

### 在哪里提问

- **使用问题**：[GitHub Discussions](https://github.com/context-forge/context-forge/discussions)
- **Bug 报告**：[GitHub Issues](https://github.com/context-forge/context-forge/issues)
- **功能请求**：[GitHub Issues](https://github.com/context-forge/context-forge/issues)

### 响应时间

- **Bug 报告**：通常 1-3 天内响应
- **功能请求**：通常 3-7 天内响应
- **Pull Request**：通常 1-5 天内审查

---

## 感谢

感谢您为 Context Forge 做出贡献！

每一个贡献，无论大小，都让这个项目变得更好。

---

**有问题？** 在 [Discussions](https://github.com/context-forge/context-forge/discussions) 中提问。
