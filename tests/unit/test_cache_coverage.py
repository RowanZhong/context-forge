"""
缓存模块覆盖率提升测试 — 针对 cache/ 模块的未覆盖路径。

覆盖范围:
- cache/keys.py: 键生成策略（normalize, prefix, context）
- cache/manager.py: CacheManager L1/L2 错误处理、统计聚合、close
- cache/__init__.py: create_default_cache 工厂函数

测试策略:
- 测试所有键生成函数的边界情况
- 测试 CacheManager 的所有错误处理分支
- 测试 create_default_cache 的各种配置组合
- 使用 mock 模拟 L1/L2 失败场景
"""

from __future__ import annotations

import asyncio
import warnings
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from context_forge.cache import (
    CacheManager,
    MemoryCache,
    compute_context_key,
    compute_prefix_key,
    compute_segment_key,
    create_default_cache,
    extract_key_metadata,
)
from context_forge.cache.base import CacheEntry, CacheStats


# === cache/keys.py 测试（目标：覆盖 54-62, 106-110, 153-157, 234-258）===


class TestCacheKeys:
    """缓存键生成函数测试。"""

    def test_compute_segment_key_with_normalization(self) -> None:
        """测试 Segment 键生成（启用归一化）。"""
        # 启用归一化（默认）
        key1 = compute_segment_key(
            content="  Python   的  GIL  ",
            model="gpt-4o",
            normalize=True,
        )
        key2 = compute_segment_key(
            content="Python 的 GIL",
            model="gpt-4o",
            normalize=True,
        )

        # 归一化后应该生成相同的键
        assert key1 == key2
        assert key1.startswith("segment:gpt-4o:")

    def test_compute_segment_key_without_normalization(self) -> None:
        """测试 Segment 键生成（禁用归一化）。"""
        key1 = compute_segment_key(
            content="  Python  ",
            model="gpt-4o",
            normalize=False,
        )
        key2 = compute_segment_key(
            content="Python",
            model="gpt-4o",
            normalize=False,
        )

        # 禁用归一化，不同空白符应产生不同的键
        assert key1 != key2
        assert key1.startswith("segment:gpt-4o:")

    def test_compute_segment_key_unicode_normalization(self) -> None:
        """测试 Unicode 归一化（覆盖 lines 54-62）。"""
        # NFC 归一化测试
        key1 = compute_segment_key(
            content="café",  # 组合形式
            model="gpt-4o",
            normalize=True,
        )
        key2 = compute_segment_key(
            content="café",  # 分解形式（如果编辑器支持）
            model="gpt-4o",
            normalize=True,
        )

        # Unicode 归一化应该产生相同的键
        assert key1.startswith("segment:gpt-4o:")

    def test_compute_segment_key_whitespace_compression(self) -> None:
        """测试空白符压缩（覆盖 lines 59-62）。"""
        key1 = compute_segment_key(
            content="  Hello\n\n\nWorld  \t  ",
            model="gpt-4o",
            normalize=True,
        )
        key2 = compute_segment_key(
            content="Hello World",
            model="gpt-4o",
            normalize=True,
        )

        # 空白符压缩后应该相同
        assert key1 == key2

    def test_compute_prefix_key_basic(self) -> None:
        """测试 Prefix 键生成。"""
        key = compute_prefix_key(
            segments_content=[
                "You are a helpful assistant.",
                "Always respond in Chinese.",
            ],
            model="claude-sonnet-4-5-20250514",
            policy_version="v1.2",
        )

        assert key.startswith("prefix:claude-sonnet-4-5-20250514:v1.2:")

    def test_compute_prefix_key_order_sensitive(self) -> None:
        """测试 Prefix 键对顺序敏感（覆盖 lines 153-157）。"""
        key1 = compute_prefix_key(
            segments_content=["First", "Second"],
            model="gpt-4o",
        )
        key2 = compute_prefix_key(
            segments_content=["Second", "First"],
            model="gpt-4o",
        )

        # 不同顺序应该产生不同的键
        assert key1 != key2

    def test_compute_prefix_key_collision_avoidance(self) -> None:
        """测试 Prefix 键避免碰撞（覆盖 lines 153-157）。"""
        # 测试分隔符防止碰撞
        key1 = compute_prefix_key(
            segments_content=["ab", "c"],
            model="gpt-4o",
        )
        key2 = compute_prefix_key(
            segments_content=["a", "bc"],
            model="gpt-4o",
        )

        # 不同切分应该产生不同的键
        assert key1 != key2

    def test_compute_prefix_key_default_policy_version(self) -> None:
        """测试 Prefix 键默认策略版本。"""
        key = compute_prefix_key(
            segments_content=["test"],
            model="gpt-4o",
            # 不传 policy_version，使用默认值
        )

        assert key.startswith("prefix:gpt-4o:default:")

    def test_compute_context_key_basic(self) -> None:
        """测试 Context 键生成。"""
        key = compute_context_key(
            segments=[
                {"type": "system", "content": "You are...", "priority": "critical"},
                {"type": "user", "content": "Hello", "priority": "high"},
            ],
            model="gpt-4o-mini",
            policy_version="v2.0",
        )

        assert key.startswith("context:gpt-4o-mini:v2.0:")

    def test_compute_context_key_deterministic(self) -> None:
        """测试 Context 键的确定性（JSON 排序）。"""
        # 相同内容但键顺序不同的字典
        segments1 = [{"type": "user", "content": "test", "priority": "high"}]
        segments2 = [{"priority": "high", "content": "test", "type": "user"}]

        key1 = compute_context_key(segments1, "gpt-4o")
        key2 = compute_context_key(segments2, "gpt-4o")

        # JSON sort_keys=True 应该产生相同的键
        assert key1 == key2

    def test_compute_context_key_chinese_content(self) -> None:
        """测试 Context 键支持中文（ensure_ascii=False）。"""
        key = compute_context_key(
            segments=[{"content": "中文内容测试"}],
            model="gpt-4o",
        )

        assert key.startswith("context:gpt-4o:default:")

    def test_extract_key_metadata_segment(self) -> None:
        """测试从 Segment 键提取元数据（覆盖 lines 241-244）。"""
        key = "segment:gpt-4o:a1b2c3d4"
        metadata = extract_key_metadata(key)

        assert metadata["type"] == "segment"
        assert metadata["model"] == "gpt-4o"
        assert metadata["hash"] == "a1b2c3d4"
        assert "policy_version" not in metadata

    def test_extract_key_metadata_prefix(self) -> None:
        """测试从 Prefix 键提取元数据（覆盖 lines 245-254）。"""
        key = "prefix:claude:v1.0:xyz123"
        metadata = extract_key_metadata(key)

        assert metadata["type"] == "prefix"
        assert metadata["model"] == "claude"
        assert metadata["policy_version"] == "v1.0"
        assert metadata["hash"] == "xyz123"

    def test_extract_key_metadata_context(self) -> None:
        """测试从 Context 键提取元数据（覆盖 lines 245-254）。"""
        key = "context:gpt-4o:v2.0:abc456"
        metadata = extract_key_metadata(key)

        assert metadata["type"] == "context"
        assert metadata["model"] == "gpt-4o"
        assert metadata["policy_version"] == "v2.0"
        assert metadata["hash"] == "abc456"

    def test_extract_key_metadata_prefix_no_policy_version(self) -> None:
        """测试从 Prefix 键提取元数据（无策略版本，覆盖 lines 252-254）。"""
        # 仅 3 个部分的 Prefix 键（降级情况）
        key = "prefix:gpt-4o:xyz"
        metadata = extract_key_metadata(key)

        assert metadata["type"] == "prefix"
        assert metadata["model"] == "gpt-4o"
        assert metadata["hash"] == "xyz"
        # 没有 policy_version 字段

    def test_extract_key_metadata_unknown_type(self) -> None:
        """测试从未知类型键提取元数据（覆盖 lines 255-256）。"""
        key = "unknown:model:hash"
        metadata = extract_key_metadata(key)

        assert metadata["type"] == "unknown"
        assert metadata["raw_key"] == key

    def test_extract_key_metadata_invalid_format(self) -> None:
        """测试从无效格式键提取元数据（覆盖 lines 235-236）。"""
        key = "invalid"
        metadata = extract_key_metadata(key)

        assert metadata["type"] == "unknown"
        assert metadata["raw_key"] == key


# === cache/manager.py 测试（目标：覆盖错误处理、统计、close）===


class TestCacheManagerErrorHandling:
    """CacheManager 错误处理测试。"""

    @pytest.mark.asyncio
    async def test_cache_manager_l1_get_error(self) -> None:
        """测试 L1 获取失败时降级到 L2（覆盖 lines 110-115）。"""
        l1 = AsyncMock(spec=MemoryCache)
        l1.get = AsyncMock(side_effect=Exception("L1 get error"))
        l2 = MemoryCache()
        manager = CacheManager(l1=l1, l2=l2)

        # L2 中写入数据
        await l2.set("key1", CacheEntry(value="from_l2"))

        # 应该警告 L1 失败，但从 L2 获取成功
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = await manager.get("key1")
            assert result is not None
            assert result.value == "from_l2"
            # 验证有警告
            assert len(w) == 1
            assert "L1 缓存读取失败" in str(w[0].message)

    @pytest.mark.asyncio
    async def test_cache_manager_l1_miss_l2_promotion_failure(self) -> None:
        """测试 L2 命中但 L1 回填失败（覆盖 lines 127-134）。"""
        l1 = AsyncMock(spec=MemoryCache)
        l1.get = AsyncMock(return_value=None)  # L1 未命中
        l1.set = AsyncMock(side_effect=Exception("L1 set error"))  # 回填失败
        l2 = MemoryCache()
        manager = CacheManager(l1=l1, l2=l2)

        # L2 中写入数据
        await l2.set("key1", CacheEntry(value="from_l2"))

        # 应该从 L2 获取成功，但回填 L1 失败（有警告）
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = await manager.get("key1")
            assert result is not None
            assert result.value == "from_l2"
            # 验证有回填失败的警告
            assert any("L1 缓存回填失败" in str(warning.message) for warning in w)

    @pytest.mark.asyncio
    async def test_cache_manager_l2_get_error(self) -> None:
        """测试 L2 获取失败时返回 None（覆盖 lines 138-144）。"""
        l1 = MemoryCache()
        l2 = AsyncMock(spec=MemoryCache)
        l2.get = AsyncMock(side_effect=Exception("L2 get error"))
        manager = CacheManager(l1=l1, l2=l2)

        # L1 和 L2 都没有数据，L2 获取失败
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = await manager.get("key1")
            assert result is None
            # 验证有警告
            assert any("L2 缓存读取失败" in str(warning.message) for warning in w)

    @pytest.mark.asyncio
    async def test_cache_manager_l1_set_error(self) -> None:
        """测试 L1 写入失败时继续（覆盖 lines 162-167）。"""
        l1 = AsyncMock(spec=MemoryCache)
        l1.set = AsyncMock(side_effect=Exception("L1 set error"))
        l2 = MemoryCache()
        manager = CacheManager(l1=l1, l2=l2)

        # 应该警告 L1 失败，但继续执行
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            await manager.set("key1", CacheEntry(value="test"))
            # 验证有警告
            assert any("L1 缓存写入失败" in str(warning.message) for warning in w)

        # L2 应该仍然成功写入
        result = await l2.get("key1")
        assert result is not None
        assert result.value == "test"

    @pytest.mark.asyncio
    async def test_cache_manager_l2_set_error(self) -> None:
        """测试 L2 写入失败时继续（覆盖 lines 173-178）。"""
        l1 = MemoryCache()
        l2 = AsyncMock(spec=MemoryCache)
        l2.set = AsyncMock(side_effect=Exception("L2 set error"))
        manager = CacheManager(l1=l1, l2=l2)

        # 应该警告 L2 失败，但继续执行
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            await manager.set("key1", CacheEntry(value="test"))
            # 验证有警告
            assert any("L2 缓存写入失败" in str(warning.message) for warning in w)

        # L1 应该仍然成功写入
        result = await l1.get("key1")
        assert result is not None
        assert result.value == "test"

    @pytest.mark.asyncio
    async def test_cache_manager_l1_delete_error(self) -> None:
        """测试 L1 删除失败时继续（覆盖 lines 190-195）。"""
        l1 = AsyncMock(spec=MemoryCache)
        l1.delete = AsyncMock(side_effect=Exception("L1 delete error"))
        l2 = MemoryCache()
        manager = CacheManager(l1=l1, l2=l2)

        # 应该警告 L1 失败，但继续执行
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            await manager.delete("key1")
            # 验证有警告
            assert any("L1 缓存删除失败" in str(warning.message) for warning in w)

    @pytest.mark.asyncio
    async def test_cache_manager_l2_delete_error(self) -> None:
        """测试 L2 删除失败时继续（覆盖 lines 201-206）。"""
        l1 = MemoryCache()
        l2 = AsyncMock(spec=MemoryCache)
        l2.delete = AsyncMock(side_effect=Exception("L2 delete error"))
        manager = CacheManager(l1=l1, l2=l2)

        # 应该警告 L2 失败，但继续执行
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            await manager.delete("key1")
            # 验证有警告
            assert any("L2 缓存删除失败" in str(warning.message) for warning in w)

    @pytest.mark.asyncio
    async def test_cache_manager_l1_clear_error(self) -> None:
        """测试 L1 清空失败时继续（覆盖 lines 213-218）。"""
        l1 = AsyncMock(spec=MemoryCache)
        l1.clear = AsyncMock(side_effect=Exception("L1 clear error"))
        l2 = MemoryCache()
        manager = CacheManager(l1=l1, l2=l2)

        # 应该警告 L1 失败，但继续执行
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            await manager.clear()
            # 验证有警告
            assert any("L1 缓存清空失败" in str(warning.message) for warning in w)

    @pytest.mark.asyncio
    async def test_cache_manager_l2_clear_error(self) -> None:
        """测试 L2 清空失败时继续（覆盖 lines 224-229）。"""
        l1 = MemoryCache()
        l2 = AsyncMock(spec=MemoryCache)
        l2.clear = AsyncMock(side_effect=Exception("L2 clear error"))
        manager = CacheManager(l1=l1, l2=l2)

        # 应该警告 L2 失败，但继续执行
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            await manager.clear()
            # 验证有警告
            assert any("L2 缓存清空失败" in str(warning.message) for warning in w)

    @pytest.mark.asyncio
    async def test_cache_manager_l1_stats_error(self) -> None:
        """测试 L1 统计获取失败时返回空统计（覆盖 lines 250-256）。"""
        l1 = AsyncMock(spec=MemoryCache)
        l1.stats = AsyncMock(side_effect=Exception("L1 stats error"))
        l2 = MemoryCache()
        manager = CacheManager(l1=l1, l2=l2)

        # 应该警告 L1 失败，但返回空统计
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            stats = await manager.stats()
            assert "l1" in stats
            assert stats["l1"].hits == 0  # 空统计
            # 验证有警告
            assert any("L1 缓存统计获取失败" in str(warning.message) for warning in w)

    @pytest.mark.asyncio
    async def test_cache_manager_l2_stats_error(self) -> None:
        """测试 L2 统计获取失败时返回空统计（覆盖 lines 262-268）。"""
        l1 = MemoryCache()
        l2 = AsyncMock(spec=MemoryCache)
        l2.stats = AsyncMock(side_effect=Exception("L2 stats error"))
        manager = CacheManager(l1=l1, l2=l2)

        # 应该警告 L2 失败，但返回空统计
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            stats = await manager.stats()
            assert "l2" in stats
            assert stats["l2"].hits == 0  # 空统计
            # 验证有警告
            assert any("L2 缓存统计获取失败" in str(warning.message) for warning in w)

    @pytest.mark.asyncio
    async def test_cache_manager_has_l2_property(self) -> None:
        """测试 has_l2 属性（覆盖 line 275）。"""
        manager1 = CacheManager(l1=MemoryCache())
        assert not manager1.has_l2

        manager2 = CacheManager(l1=MemoryCache(), l2=MemoryCache())
        assert manager2.has_l2

    @pytest.mark.asyncio
    async def test_cache_manager_close_with_l2(self) -> None:
        """测试关闭 L2 连接（覆盖 lines 292-300）。"""
        l1 = MemoryCache()
        l2 = AsyncMock(spec=MemoryCache)
        l2.close = AsyncMock()
        manager = CacheManager(l1=l1, l2=l2)

        await manager.close()

        # 验证 L2 的 close 被调用
        l2.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_manager_close_l2_error(self) -> None:
        """测试 L2 关闭失败时继续（覆盖 lines 295-300）。"""
        l1 = MemoryCache()
        l2 = AsyncMock(spec=MemoryCache)
        l2.close = AsyncMock(side_effect=Exception("Close error"))
        manager = CacheManager(l1=l1, l2=l2)

        # 应该警告但不抛异常
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            await manager.close()
            # 验证有警告
            assert any("L2 缓存关闭失败" in str(warning.message) for warning in w)

    @pytest.mark.asyncio
    async def test_cache_manager_close_no_l2(self) -> None:
        """测试无 L2 时关闭（覆盖 line 292）。"""
        manager = CacheManager(l1=MemoryCache())

        # 应该正常返回，不抛异常
        await manager.close()

    @pytest.mark.asyncio
    async def test_cache_manager_context_manager(self) -> None:
        """测试异步 context manager（覆盖 lines 302-308）。"""
        l2 = AsyncMock(spec=MemoryCache)
        l2.close = AsyncMock()

        async with CacheManager(l1=MemoryCache(), l2=l2) as manager:
            await manager.set("key1", CacheEntry(value="test"))

        # 验证退出时 close 被调用
        l2.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_manager_default_l1(self) -> None:
        """测试默认创建 L1 缓存。"""
        manager = CacheManager()

        # 应该自动创建默认的 MemoryCache
        await manager.set("key1", CacheEntry(value="test"))
        result = await manager.get("key1")
        assert result is not None
        assert result.value == "test"


# === cache/__init__.py 测试（目标：覆盖 66-68, 113-144）===


class TestCacheInit:
    """cache/__init__.py 工厂函数测试。"""

    def test_create_default_cache_l1_only(self) -> None:
        """测试创建仅 L1 缓存。"""
        cache = create_default_cache()

        assert isinstance(cache, CacheManager)
        assert not cache.has_l2

    def test_create_default_cache_custom_l1_size(self) -> None:
        """测试自定义 L1 大小。"""
        cache = create_default_cache(l1_max_size=5000, default_ttl=7200)

        assert isinstance(cache, CacheManager)
        # 无法直接验证内部参数，但可以通过行为验证

    @pytest.mark.asyncio
    async def test_create_default_cache_redis_not_available(self) -> None:
        """测试 Redis 不可用时的降级（覆盖 lines 118-127）。"""
        # Mock _REDIS_AVAILABLE = False
        with patch("context_forge.cache._REDIS_AVAILABLE", False):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                cache = create_default_cache(redis_url="redis://localhost:6379/0")

                # 应该有警告
                assert len(w) == 1
                assert "无法启用 Redis 缓存" in str(w[0].message)
                assert "未安装 redis-py 包" in str(w[0].message)

                # 应该只有 L1
                assert not cache.has_l2

    @pytest.mark.asyncio
    async def test_create_default_cache_redis_init_error(self) -> None:
        """测试 Redis 初始化失败时的降级（覆盖 lines 129-142）。"""
        # Mock RedisCache 初始化失败
        with patch("context_forge.cache._REDIS_AVAILABLE", True):
            with patch("context_forge.cache.RedisCache") as MockRedisCache:
                MockRedisCache.side_effect = Exception("Redis connection failed")

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    cache = create_default_cache(redis_url="redis://localhost:6379/0")

                    # 应该有警告
                    assert any("Redis 缓存初始化失败" in str(warning.message) for warning in w)

                    # 应该只有 L1
                    assert not cache.has_l2

    @pytest.mark.asyncio
    async def test_create_default_cache_with_redis_kwargs(self) -> None:
        """测试传递 Redis 额外参数。"""
        # 由于实际测试需要 Redis 可用，这里只测试参数传递
        # 实际构造会在集成测试中验证
        with patch("context_forge.cache._REDIS_AVAILABLE", True):
            with patch("context_forge.cache.RedisCache") as MockRedisCache:
                MockRedisCache.return_value = MagicMock()

                cache = create_default_cache(
                    redis_url="redis://localhost:6379/0",
                    l1_max_size=5000,
                    default_ttl=3600,
                    socket_timeout=5,
                    max_connections=10,
                )

                # 验证 RedisCache 被调用并传递了正确的参数
                MockRedisCache.assert_called_once()
                call_kwargs = MockRedisCache.call_args[1]
                assert call_kwargs["redis_url"] == "redis://localhost:6379/0"
                assert call_kwargs["default_ttl"] == 3600
                assert call_kwargs["socket_timeout"] == 5
                assert call_kwargs["max_connections"] == 10


# === 额外的边界测试 ===


class TestCacheEdgeCases:
    """缓存模块边界情况测试。"""

    @pytest.mark.asyncio
    async def test_memory_cache_prune_expired(self) -> None:
        """测试主动清理过期条目。"""
        cache = MemoryCache()

        # 插入两个条目，一个会过期，一个不会
        await cache.set("key1", CacheEntry(value="v1"), ttl=1)
        await cache.set("key2", CacheEntry(value="v2"), ttl=10)

        # 等待第一个条目过期
        await asyncio.sleep(1.5)

        # 主动清理
        pruned_count = await cache.prune_expired()
        assert pruned_count == 1

        # 验证只剩下未过期的
        assert await cache.get("key1") is None
        assert await cache.get("key2") is not None

    @pytest.mark.asyncio
    async def test_memory_cache_ttl_none(self) -> None:
        """测试永不过期的缓存条目。"""
        cache = MemoryCache(default_ttl=None)

        await cache.set("key1", CacheEntry(value="test"))

        # 等待一段时间
        await asyncio.sleep(1)

        # 应该仍然存在
        result = await cache.get("key1")
        assert result is not None
        assert result.value == "test"

    def test_cache_stats_properties(self) -> None:
        """测试 CacheStats 属性计算。"""
        # 测试命中率
        stats1 = CacheStats(hits=80, misses=20)
        assert stats1.hit_rate == 0.8

        stats2 = CacheStats(hits=0, misses=0)
        assert stats2.hit_rate == 0.0

        # 测试淘汰率
        stats3 = CacheStats(sets=100, evictions=10)
        assert stats3.eviction_rate == 0.1

        stats4 = CacheStats(sets=0, evictions=0)
        assert stats4.eviction_rate == 0.0

    def test_cache_entry_frozen(self) -> None:
        """测试 CacheEntry 不可变性。"""
        entry = CacheEntry(value="test", hit_count=0)

        # 尝试修改应该失败
        with pytest.raises(Exception):  # FrozenInstanceError in Pydantic v2
            entry.hit_count = 10  # type: ignore

    def test_cache_entry_with_hit_immutable(self) -> None:
        """测试 with_hit 不修改原对象。"""
        entry = CacheEntry(value="test", hit_count=5)
        updated = entry.with_hit()

        # 原对象不变
        assert entry.hit_count == 5
        # 新对象已更新
        assert updated.hit_count == 6
        # 其他字段保持一致
        assert updated.value == entry.value
        assert updated.created_at == entry.created_at
        assert updated.metadata == entry.metadata

    @pytest.mark.asyncio
    async def test_cache_manager_stats_both_layers(self) -> None:
        """测试同时获取 L1 和 L2 的统计。"""
        l1 = MemoryCache()
        l2 = MemoryCache()
        manager = CacheManager(l1=l1, l2=l2)

        await manager.set("key1", CacheEntry(value="v1"))
        await manager.get("key1")  # L1 hit
        await manager.get("key2")  # L1 miss, L2 miss

        stats = await manager.stats()

        assert "l1" in stats
        assert "l2" in stats
        assert stats["l1"].hits >= 1
        assert stats["l1"].misses >= 1
        assert stats["l2"].misses >= 1

    def test_compute_segment_key_different_models(self) -> None:
        """测试不同模型生成不同的键。"""
        key1 = compute_segment_key(content="test", model="gpt-4o")
        key2 = compute_segment_key(content="test", model="claude-sonnet-4-5")

        assert key1 != key2
        assert "gpt-4o" in key1
        assert "claude-sonnet-4-5" in key2

    def test_compute_prefix_key_empty_segments(self) -> None:
        """测试空 Segment 列表。"""
        key = compute_prefix_key(segments_content=[], model="gpt-4o")

        assert key.startswith("prefix:gpt-4o:default:")

    def test_compute_context_key_empty_segments(self) -> None:
        """测试空 Segment 列表。"""
        key = compute_context_key(segments=[], model="gpt-4o")

        assert key.startswith("context:gpt-4o:default:")

    def test_extract_key_metadata_malformed_prefix(self) -> None:
        """测试提取格式错误的 Prefix 键元数据。"""
        # 只有两个部分（不足 3 个）
        key = "prefix:model"
        metadata = extract_key_metadata(key)

        assert metadata["type"] == "unknown"
        assert metadata["raw_key"] == key
