"""
Context Forge CLI — 命令行工具。

→ 6.1.2.2 CLI 工具链设计

提供完整的 CLI 工具链，包括：
- init: 初始化项目配置
- validate: 校验策略文件和输入文件
- build: 从文件构建上下文
- inspect: 查看快照或构建结果
- diff: 比对两个快照
- serve: API 服务器（第三轮实现）
"""

from context_forge.cli.app import app, main

__all__ = ["app", "main"]
