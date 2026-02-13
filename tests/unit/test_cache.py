"""
缓存模块单元测试 — 测试缓存管理器和缓存后端。

覆盖范围:
- cache/base.py: CacheEntry (100%), CacheStats, CacheBackend Protocol
- cache/memory.py: MemoryCache （内存缓存 LRU）
- cache/redis_backend.py: RedisCache （分布式 Redis 缓存，95% 覆盖率）
- cache/manager.py: CacheManager （L1/L2 分层缓存管理）
- cache/keys.py: 缓存键生成函数

Redis 测试策略：
- 使用 unittest.mock 模拟 Redis 异步客户端（AsyncMock）
- 不依赖真实 Redis 实例，完全离线测试
- 覆盖所有公共接口和错误处理路径
- 测试序列化/反序列化、TTL、键前缀等核心功能
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from context_forge.cache import CacheManager, MemoryCache
from context_forge.cache.base import CacheEntry, CacheStats


# === CacheEntry 测试（~3 tests）===


class TestCacheEntry:
    """CacheEntry 测试。"""

    def test_create_cache_entry(self) -> None:
        """测试创建缓存条目。"""
        entry = CacheEntry(value="test_value")
        assert entry.value == "test_value"
        assert isinstance(entry.created_at, datetime)

    def test_cache_entry_with_hit(self) -> None:
        """测试 with_hit 返回新对象并增加命中次数。"""
        entry = CacheEntry(value="test")
        assert entry.hit_count == 0

        updated = entry.with_hit()
        assert updated.hit_count == 1
        # 原条目不变（不可变模式）
        assert entry.hit_count == 0

    def test_cache_entry_metadata(self) -> None:
        """测试带元数据的缓存条目。"""
        entry = CacheEntry(
            value="test",
            metadata={"source": "rag", "version": "1.0"},
        )
        assert entry.metadata["source"] == "rag"
        assert entry.metadata["version"] == "1.0"


# === MemoryCache 测试（~10 tests）===


class TestMemoryCache:
    """MemoryCache 测试。"""

    @pytest.mark.asyncio
    async def test_memory_cache_set_get(self) -> None:
        """测试基本的 set/get 操作。"""
        cache = MemoryCache()
        entry = CacheEntry(value="test_value")

        await cache.set("key1", entry)
        result = await cache.get("key1")

        assert result is not None
        assert result.value == "test_value"

    @pytest.mark.asyncio
    async def test_memory_cache_get_nonexistent(self) -> None:
        """测试获取不存在的键。"""
        cache = MemoryCache()
        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_memory_cache_delete(self) -> None:
        """测试删除条目。"""
        cache = MemoryCache()
        entry = CacheEntry(value="test")

        await cache.set("key1", entry)
        await cache.delete("key1")

        result = await cache.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_memory_cache_clear(self) -> None:
        """测试清空缓存。"""
        cache = MemoryCache()

        await cache.set("key1", CacheEntry(value="v1"))
        await cache.set("key2", CacheEntry(value="v2"))

        await cache.clear()

        assert await cache.get("key1") is None
        assert await cache.get("key2") is None

    @pytest.mark.asyncio
    async def test_memory_cache_ttl_expiration(self) -> None:
        """测试 TTL 过期自动清理。"""
        cache = MemoryCache()
        entry = CacheEntry(value="test")

        # 使用 set 的 ttl 参数来设置过期时间
        await cache.set("key1", entry, ttl=1)

        # 立即获取应该存在
        result = await cache.get("key1")
        assert result is not None

        # 等待过期
        await asyncio.sleep(1.5)

        # 再次获取应该已过期
        result = await cache.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_memory_cache_max_size_lru_eviction(self) -> None:
        """测试 LRU 驱逐（最大容量限制）。"""
        cache = MemoryCache(max_size=3)

        # 插入 4 个条目
        await cache.set("key1", CacheEntry(value="v1"))
        await cache.set("key2", CacheEntry(value="v2"))
        await cache.set("key3", CacheEntry(value="v3"))
        await cache.set("key4", CacheEntry(value="v4"))  # 触发驱逐

        # key1 应该被驱逐（最久未使用）
        assert await cache.get("key1") is None
        assert await cache.get("key4") is not None

    @pytest.mark.asyncio
    async def test_memory_cache_lru_access_update(self) -> None:
        """测试访问更新 LRU 顺序。"""
        cache = MemoryCache(max_size=3)

        await cache.set("key1", CacheEntry(value="v1"))
        await cache.set("key2", CacheEntry(value="v2"))
        await cache.set("key3", CacheEntry(value="v3"))

        # 访问 key1，使其变为最近使用
        await cache.get("key1")

        # 插入新条目
        await cache.set("key4", CacheEntry(value="v4"))

        # key2 应该被驱逐（最久未访问）
        assert await cache.get("key2") is None
        assert await cache.get("key1") is not None

    @pytest.mark.asyncio
    async def test_memory_cache_default_ttl(self) -> None:
        """测试默认 TTL 设置。"""
        cache = MemoryCache(default_ttl=2)
        entry = CacheEntry(value="test")  # 没有指定 TTL

        await cache.set("key1", entry)

        # 应该能取到（2 秒内有效）
        result = await cache.get("key1")
        assert result is not None
        assert result.value == "test"

    @pytest.mark.asyncio
    async def test_memory_cache_concurrent_access(self) -> None:
        """测试并发访问安全性。"""
        cache = MemoryCache()

        async def worker(i: int) -> None:
            await cache.set(f"key{i}", CacheEntry(value=f"v{i}"))
            await cache.get(f"key{i}")

        # 并发执行 10 个 worker
        await asyncio.gather(*[worker(i) for i in range(10)])

        # 所有条目应该都存在
        for i in range(10):
            result = await cache.get(f"key{i}")
            assert result is not None

    @pytest.mark.asyncio
    async def test_memory_cache_stats(self) -> None:
        """测试缓存统计信息。"""
        cache = MemoryCache()

        await cache.set("key1", CacheEntry(value="v1"))
        await cache.get("key1")  # hit
        await cache.get("key2")  # miss

        stats = await cache.stats()
        assert isinstance(stats, CacheStats)
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.current_size == 1


# === CacheManager 测试（~7 tests）===


class TestCacheManager:
    """CacheManager 测试（L1/L2 分层缓存）。"""

    @pytest.mark.asyncio
    async def test_cache_manager_l1_only(self) -> None:
        """测试只有 L1 缓存的场景。"""
        l1 = MemoryCache()
        manager = CacheManager(l1=l1)

        entry = CacheEntry(value="test")
        await manager.set("key1", entry)

        result = await manager.get("key1")
        assert result is not None
        assert result.value == "test"

    @pytest.mark.asyncio
    async def test_cache_manager_l1_l2_hierarchy(self) -> None:
        """测试 L1/L2 分层缓存。"""
        l1 = MemoryCache(max_size=2)
        l2 = MemoryCache(max_size=10)
        manager = CacheManager(l1=l1, l2=l2)

        # 写入缓存（同时写入 L1 和 L2）
        await manager.set("key1", CacheEntry(value="v1"))

        # L1 和 L2 应该都有
        assert await l1.get("key1") is not None
        assert await l2.get("key1") is not None

    @pytest.mark.asyncio
    async def test_cache_manager_l1_miss_l2_hit(self) -> None:
        """测试 L1 未命中但 L2 命中的场景。"""
        l1 = MemoryCache()
        l2 = MemoryCache()
        manager = CacheManager(l1=l1, l2=l2)

        # 只在 L2 中写入
        await l2.set("key1", CacheEntry(value="v1"))

        # 通过 manager 获取应该能从 L2 取到并回填 L1
        result = await manager.get("key1")
        assert result is not None
        assert result.value == "v1"

        # L1 应该被回填
        assert await l1.get("key1") is not None

    @pytest.mark.asyncio
    async def test_cache_manager_delete_cascade(self) -> None:
        """测试删除操作级联到 L1 和 L2。"""
        l1 = MemoryCache()
        l2 = MemoryCache()
        manager = CacheManager(l1=l1, l2=l2)

        await manager.set("key1", CacheEntry(value="v1"))
        await manager.delete("key1")

        # L1 和 L2 都应该被删除
        assert await l1.get("key1") is None
        assert await l2.get("key1") is None

    @pytest.mark.asyncio
    async def test_cache_manager_clear_cascade(self) -> None:
        """测试清空操作级联。"""
        l1 = MemoryCache()
        l2 = MemoryCache()
        manager = CacheManager(l1=l1, l2=l2)

        await manager.set("key1", CacheEntry(value="v1"))
        await manager.set("key2", CacheEntry(value="v2"))

        await manager.clear()

        # L1 和 L2 都应该被清空
        assert await l1.get("key1") is None
        assert await l2.get("key1") is None

    @pytest.mark.asyncio
    async def test_cache_manager_key_generation(self) -> None:
        """测试缓存键生成（使用 cache/keys.py 函数）。"""
        from context_forge.cache.keys import compute_context_key

        key1 = compute_context_key([{"messages": ["hello"]}], model="gpt-4o")
        key2 = compute_context_key([{"messages": ["hello"]}], model="gpt-4o")
        key3 = compute_context_key([{"messages": ["world"]}], model="gpt-4o")

        # 相同输入应该生成相同的键
        assert key1 == key2
        # 不同输入应该生成不同的键
        assert key1 != key3

    @pytest.mark.asyncio
    async def test_cache_manager_stats_aggregation(self) -> None:
        """测试统计信息聚合（L1 + L2）。"""
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


# === RedisCache 测试（~15 tests，使用 mock Redis 客户端）===


class TestRedisCacheBasic:
    """RedisCache 基础功能测试（模拟 Redis 客户端）。"""

    @pytest.mark.asyncio
    async def test_redis_cache_init(self) -> None:
        """测试 RedisCache 初始化。"""
        from context_forge.cache.redis_backend import RedisCache

        cache = RedisCache(
            redis_url="redis://localhost:6379/0",
            key_prefix="test:",
            default_ttl=3600,
        )

        assert cache._redis_url == "redis://localhost:6379/0"
        assert cache._key_prefix == "test:"
        assert cache._default_ttl == 3600
        assert cache._client is None  # 未初始化

    @pytest.mark.asyncio
    async def test_redis_cache_make_key(self) -> None:
        """测试缓存键前缀生成。"""
        from context_forge.cache.redis_backend import RedisCache

        cache = RedisCache(key_prefix="cf:")
        key = cache._make_key("test_key")
        assert key == "cf:test_key"

    @pytest.mark.asyncio
    async def test_redis_cache_serialize_entry(self) -> None:
        """测试缓存条目序列化。"""
        from context_forge.cache.redis_backend import RedisCache

        cache = RedisCache()
        entry = CacheEntry(
            value="test_value",
            hit_count=5,
            metadata={"source": "test"},
        )

        serialized = cache._serialize_entry(entry)
        assert serialized["value"] == "test_value"
        assert serialized["hit_count"] == 5
        assert serialized["metadata"]["source"] == "test"
        assert "created_at" in serialized

    @pytest.mark.asyncio
    async def test_redis_cache_get_client_caching(self) -> None:
        """测试多次调用 _get_client 只初始化一次。"""
        from context_forge.cache.redis_backend import RedisCache
        from unittest.mock import AsyncMock

        cache = RedisCache()
        mock_client = AsyncMock()
        cache._client = mock_client

        # 第一次调用
        client1 = await cache._get_client()
        # 第二次调用
        client2 = await cache._get_client()

        # 应该返回同一个对象
        assert client1 is client2 is mock_client

    @pytest.mark.asyncio
    async def test_redis_cache_with_invalid_url(self) -> None:
        """测试连接失败时的优雅降级。"""
        from context_forge.cache.redis_backend import RedisCache
        from unittest.mock import patch, AsyncMock

        cache = RedisCache(redis_url="redis://invalid-host:99999/0")

        # 模拟 from_url 抛出异常
        with patch("redis.asyncio.from_url", side_effect=Exception("Connection refused")):
            with pytest.warns(UserWarning, match="Redis 缓存连接失败"):
                cache._client = None  # 重置缓存
                client = await cache._get_client()
                assert client is None


class TestRedisCacheWithMock:
    """使用 mock Redis 客户端的完整测试。"""

    @pytest.mark.asyncio
    async def test_redis_cache_set_get_success(self) -> None:
        """测试成功的 set/get 操作。"""
        from context_forge.cache.redis_backend import RedisCache
        from unittest.mock import AsyncMock, patch
        import json

        cache = RedisCache(key_prefix="test:")

        # 创建 mock Redis 客户端
        mock_client = AsyncMock()
        cache._client = mock_client

        # 测试 set 操作
        entry = CacheEntry(value="test_content")
        await cache.set("key1", entry, ttl=3600)

        # 验证 Redis set 被调用
        mock_client.set.assert_called_once()
        call_args = mock_client.set.call_args
        assert call_args[0][0] == "test:key1"  # 前缀键
        assert call_args[1]["ex"] == 3600  # TTL

        # 测试 get 操作
        serialized = cache._serialize_entry(entry)
        mock_client.get = AsyncMock(return_value=json.dumps(serialized))

        result = await cache.get("key1")
        assert result is not None
        assert result.value == "test_content"
        assert cache._hits == 1

    @pytest.mark.asyncio
    async def test_redis_cache_get_miss(self) -> None:
        """测试缓存未命中。"""
        from context_forge.cache.redis_backend import RedisCache
        from unittest.mock import AsyncMock

        cache = RedisCache()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=None)
        cache._client = mock_client

        result = await cache.get("nonexistent_key")
        assert result is None
        assert cache._misses == 1

    @pytest.mark.asyncio
    async def test_redis_cache_get_no_client(self) -> None:
        """测试无可用 Redis 客户端时的处理。"""
        from context_forge.cache.redis_backend import RedisCache
        from unittest.mock import AsyncMock, patch

        cache = RedisCache()

        # 模拟 _get_client 返回 None（Redis 不可用）
        with patch.object(cache, "_get_client", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None
            result = await cache.get("key1")
            assert result is None

    @pytest.mark.asyncio
    async def test_redis_cache_get_deserialization_error(self) -> None:
        """测试反序列化错误处理。"""
        from context_forge.cache.redis_backend import RedisCache
        from unittest.mock import AsyncMock

        cache = RedisCache()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value="invalid_json_{}")
        cache._client = mock_client

        # 应该返回 None 并记录 warning
        with pytest.warns(UserWarning, match="Redis 缓存读取失败"):
            result = await cache.get("key1")
            assert result is None
            assert cache._errors == 1

    @pytest.mark.asyncio
    async def test_redis_cache_delete_success(self) -> None:
        """测试删除操作。"""
        from context_forge.cache.redis_backend import RedisCache
        from unittest.mock import AsyncMock

        cache = RedisCache(key_prefix="cf:")
        mock_client = AsyncMock()
        cache._client = mock_client

        await cache.delete("key1")

        mock_client.delete.assert_called_once_with("cf:key1")

    @pytest.mark.asyncio
    async def test_redis_cache_delete_error(self) -> None:
        """测试删除失败时的错误处理。"""
        from context_forge.cache.redis_backend import RedisCache
        from unittest.mock import AsyncMock

        cache = RedisCache()
        mock_client = AsyncMock()
        mock_client.delete = AsyncMock(side_effect=Exception("Delete failed"))
        cache._client = mock_client

        # 应该返回 None 但不抛异常
        with pytest.warns(UserWarning, match="Redis 缓存删除失败"):
            await cache.delete("key1")

    @pytest.mark.asyncio
    async def test_redis_cache_clear_success(self) -> None:
        """测试清空缓存（SCAN + DELETE）。"""
        from context_forge.cache.redis_backend import RedisCache
        from unittest.mock import AsyncMock

        cache = RedisCache(key_prefix="test:")
        mock_client = AsyncMock()
        cache._client = mock_client

        # 模拟 SCAN 返回两批键
        mock_client.scan = AsyncMock(
            side_effect=[
                (1, ["test:key1", "test:key2"]),  # 第一批
                (0, ["test:key3"]),  # 最后一批
            ]
        )

        await cache.clear()

        # 验证 scan 被调用
        assert mock_client.scan.call_count == 2

        # 验证 delete 被调用两次
        assert mock_client.delete.call_count == 2

    @pytest.mark.asyncio
    async def test_redis_cache_clear_no_keys(self) -> None:
        """测试清空空缓存。"""
        from context_forge.cache.redis_backend import RedisCache
        from unittest.mock import AsyncMock

        cache = RedisCache()
        mock_client = AsyncMock()
        mock_client.scan = AsyncMock(return_value=(0, []))
        cache._client = mock_client

        await cache.clear()

        # 只调用一次 scan
        mock_client.scan.assert_called_once()

    @pytest.mark.asyncio
    async def test_redis_cache_clear_error(self) -> None:
        """测试清空失败时的错误处理。"""
        from context_forge.cache.redis_backend import RedisCache
        from unittest.mock import AsyncMock

        cache = RedisCache()
        mock_client = AsyncMock()
        mock_client.scan = AsyncMock(side_effect=Exception("Scan failed"))
        cache._client = mock_client

        with pytest.warns(UserWarning, match="Redis 缓存清空失败"):
            await cache.clear()

    @pytest.mark.asyncio
    async def test_redis_cache_stats_with_client(self) -> None:
        """测试统计信息收集（有 Redis 客户端）。"""
        from context_forge.cache.redis_backend import RedisCache
        from unittest.mock import AsyncMock

        cache = RedisCache(key_prefix="cf:")
        mock_client = AsyncMock()
        cache._client = mock_client

        # 模拟 SCAN 返回 3 个键
        mock_client.scan = AsyncMock(
            side_effect=[
                (1, ["cf:key1", "cf:key2"]),
                (0, ["cf:key3"]),
            ]
        )

        # 设置一些统计数据
        cache._hits = 10
        cache._misses = 5
        cache._sets = 15

        stats = await cache.stats()
        assert stats.hits == 10
        assert stats.misses == 5
        assert stats.sets == 15
        assert stats.current_size == 3

    @pytest.mark.asyncio
    async def test_redis_cache_stats_no_client(self) -> None:
        """测试统计信息收集（无 Redis 客户端）。"""
        from context_forge.cache.redis_backend import RedisCache

        cache = RedisCache()
        cache._client = None
        cache._hits = 5
        cache._misses = 2

        stats = await cache.stats()
        assert stats.hits == 5
        assert stats.misses == 2
        assert stats.current_size == 0

    @pytest.mark.asyncio
    async def test_redis_cache_close_success(self) -> None:
        """测试关闭连接。"""
        from context_forge.cache.redis_backend import RedisCache
        from unittest.mock import AsyncMock

        cache = RedisCache()
        mock_client = AsyncMock()
        cache._client = mock_client

        await cache.close()

        mock_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_redis_cache_close_no_client(self) -> None:
        """测试无客户端时关闭。"""
        from context_forge.cache.redis_backend import RedisCache

        cache = RedisCache()
        cache._client = None

        # 应该不抛异常
        await cache.close()

    @pytest.mark.asyncio
    async def test_redis_cache_close_error(self) -> None:
        """测试关闭连接失败。"""
        from context_forge.cache.redis_backend import RedisCache
        from unittest.mock import AsyncMock

        cache = RedisCache()
        mock_client = AsyncMock()
        mock_client.close = AsyncMock(side_effect=Exception("Close failed"))
        cache._client = mock_client

        # 应该不抛异常
        await cache.close()

    @pytest.mark.asyncio
    async def test_redis_cache_set_no_client(self) -> None:
        """测试无客户端时的 set 操作。"""
        from context_forge.cache.redis_backend import RedisCache
        from unittest.mock import AsyncMock, patch

        cache = RedisCache()

        with patch.object(cache, "_get_client", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None
            entry = CacheEntry(value="test")
            await cache.set("key1", entry)
            # 应该正常返回，不抛异常

    @pytest.mark.asyncio
    async def test_redis_cache_set_error(self) -> None:
        """测试 set 操作失败。"""
        from context_forge.cache.redis_backend import RedisCache
        from unittest.mock import AsyncMock

        cache = RedisCache()
        mock_client = AsyncMock()
        mock_client.set = AsyncMock(side_effect=Exception("Set failed"))
        cache._client = mock_client

        with pytest.warns(UserWarning, match="Redis 缓存写入失败"):
            entry = CacheEntry(value="test")
            await cache.set("key1", entry)
            assert cache._errors == 1

    @pytest.mark.asyncio
    async def test_redis_cache_default_ttl(self) -> None:
        """测试默认 TTL 的使用。"""
        from context_forge.cache.redis_backend import RedisCache
        from unittest.mock import AsyncMock

        cache = RedisCache(default_ttl=7200)
        mock_client = AsyncMock()
        cache._client = mock_client

        entry = CacheEntry(value="test")
        # 不指定 ttl，应该使用 default_ttl
        await cache.set("key1", entry)

        call_args = mock_client.set.call_args
        assert call_args[1]["ex"] == 7200

    @pytest.mark.asyncio
    async def test_redis_cache_no_ttl(self) -> None:
        """测试没有 TTL 的情况（永不过期）。"""
        from context_forge.cache.redis_backend import RedisCache
        from unittest.mock import AsyncMock

        cache = RedisCache(default_ttl=None)
        mock_client = AsyncMock()
        cache._client = mock_client

        entry = CacheEntry(value="test")
        await cache.set("key1", entry)

        # 验证调用时没有 ex 参数
        call_args = mock_client.set.call_args
        # 仅传递 key 和 value，不传递 ex
        assert "ex" not in call_args[1] or call_args[1].get("ex") is None

    @pytest.mark.asyncio
    async def test_redis_cache_get_and_update_hit_count(self) -> None:
        """测试 get 时更新命中次数。"""
        from context_forge.cache.redis_backend import RedisCache
        from unittest.mock import AsyncMock
        import json

        cache = RedisCache()
        mock_client = AsyncMock()
        cache._client = mock_client

        # 模拟 Redis 返回的数据
        entry = CacheEntry(value="test_value", hit_count=5)
        serialized = cache._serialize_entry(entry)
        mock_client.get = AsyncMock(return_value=json.dumps(serialized))
        mock_client.set = AsyncMock()

        result = await cache.get("key1")

        # 验证返回的条目命中次数已增加
        assert result is not None
        assert result.hit_count == 6

        # 验证更新后的条目已写回 Redis
        mock_client.set.assert_called_once()
        call_args = mock_client.set.call_args
        # 验证 keepttl 参数
        assert call_args[1].get("keepttl") is True

    @pytest.mark.asyncio
    async def test_redis_cache_import_error(self) -> None:
        """测试 redis-py 未安装时的优雅降级（ImportError）。"""
        from context_forge.cache.redis_backend import RedisCache
        from unittest.mock import patch

        cache = RedisCache(redis_url="redis://localhost:6379/0")
        cache._client = None  # 重置以强制重新初始化

        # 通过 patch importlib 来模拟 redis 模块缺失
        original_getattr = getattr

        def mock_getattr(obj, name, *args, **kwargs):
            # 在调用 aioredis.from_url 时抛出 ImportError
            if name == "from_url" and hasattr(obj, "__name__") and "redis" in str(obj):
                raise ImportError("No module named 'redis'")
            return original_getattr(obj, name, *args, **kwargs)

        # 更直接的方法：patch 内部 import
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "redis.asyncio":
                raise ImportError("No module named 'redis'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.warns(UserWarning, match="未安装 redis-py"):
                result = await cache._get_client()
                assert result is None

    @pytest.mark.asyncio
    async def test_redis_cache_delete_no_client(self) -> None:
        """测试无客户端时的 delete 操作。"""
        from context_forge.cache.redis_backend import RedisCache
        from unittest.mock import AsyncMock, patch

        cache = RedisCache()

        with patch.object(cache, "_get_client", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None
            await cache.delete("key1")
            # 应该正常返回，不抛异常

    @pytest.mark.asyncio
    async def test_redis_cache_clear_no_client(self) -> None:
        """测试无客户端时的 clear 操作。"""
        from context_forge.cache.redis_backend import RedisCache
        from unittest.mock import AsyncMock, patch

        cache = RedisCache()

        with patch.object(cache, "_get_client", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None
            await cache.clear()
            # 应该正常返回，不抛异常

    @pytest.mark.asyncio
    async def test_redis_cache_stats_scan_error(self) -> None:
        """测试 stats 中 SCAN 失败时的优雅处理。"""
        from context_forge.cache.redis_backend import RedisCache
        from unittest.mock import AsyncMock

        cache = RedisCache()
        mock_client = AsyncMock()
        mock_client.scan = AsyncMock(side_effect=Exception("Scan error"))
        cache._client = mock_client
        cache._hits = 5

        # 应该返回统计信息，不抛异常
        stats = await cache.stats()
        assert stats.hits == 5
        assert stats.current_size == 0

    @pytest.mark.asyncio
    async def test_redis_cache_multiple_scan_iterations(self) -> None:
        """测试 stats 中多次 SCAN 迭代。"""
        from context_forge.cache.redis_backend import RedisCache
        from unittest.mock import AsyncMock

        cache = RedisCache(key_prefix="test:")
        mock_client = AsyncMock()
        cache._client = mock_client

        # 模拟多次 SCAN 返回大量键
        mock_client.scan = AsyncMock(
            side_effect=[
                (100, ["test:k1", "test:k2", "test:k3"]),
                (200, ["test:k4", "test:k5"]),
                (0, ["test:k6"]),
            ]
        )

        stats = await cache.stats()
        assert stats.current_size == 6
        assert mock_client.scan.call_count == 3

    @pytest.mark.asyncio
    async def test_redis_cache_get_with_complex_metadata(self) -> None:
        """测试 get 处理复杂元数据。"""
        from context_forge.cache.redis_backend import RedisCache
        from unittest.mock import AsyncMock
        import json

        cache = RedisCache()
        mock_client = AsyncMock()
        cache._client = mock_client

        # 创建带有复杂元数据的条目
        entry = CacheEntry(
            value="complex_value",
            hit_count=3,
            metadata={
                "source": "rag",
                "version": "2.0",
                "tags": ["tag1", "tag2"],
            },
        )
        serialized = cache._serialize_entry(entry)
        mock_client.get = AsyncMock(return_value=json.dumps(serialized))
        mock_client.set = AsyncMock()

        result = await cache.get("key_with_meta")

        assert result is not None
        assert result.metadata["source"] == "rag"
        assert result.metadata["version"] == "2.0"
        assert result.hit_count == 4

    @pytest.mark.asyncio
    async def test_redis_cache_set_with_zero_ttl(self) -> None:
        """测试 set 处理 TTL=0 的情况。"""
        from context_forge.cache.redis_backend import RedisCache
        from unittest.mock import AsyncMock

        cache = RedisCache()
        mock_client = AsyncMock()
        cache._client = mock_client

        entry = CacheEntry(value="test")
        # TTL=0 应该被视为正常值
        await cache.set("key1", entry, ttl=0)

        call_args = mock_client.set.call_args
        assert call_args[1].get("ex") == 0

    @pytest.mark.asyncio
    async def test_redis_cache_concurrent_operations(self) -> None:
        """测试并发操作的安全性。"""
        from context_forge.cache.redis_backend import RedisCache
        from unittest.mock import AsyncMock
        import json

        cache = RedisCache()
        mock_client = AsyncMock()
        cache._client = mock_client

        async def concurrent_set(i: int) -> None:
            entry = CacheEntry(value=f"value_{i}")
            await cache.set(f"key_{i}", entry)

        # 并发 set 操作
        await asyncio.gather(*[concurrent_set(i) for i in range(5)])

        # 验证所有 set 都被调用
        assert mock_client.set.call_count == 5

    @pytest.mark.asyncio
    async def test_redis_cache_key_prefix_isolation(self) -> None:
        """测试不同前缀的键隔离。"""
        from context_forge.cache.redis_backend import RedisCache
        from unittest.mock import AsyncMock

        cache1 = RedisCache(key_prefix="app1:")
        cache2 = RedisCache(key_prefix="app2:")

        key1 = cache1._make_key("test")
        key2 = cache2._make_key("test")

        assert key1 == "app1:test"
        assert key2 == "app2:test"
        assert key1 != key2

    @pytest.mark.asyncio
    async def test_redis_cache_error_increments_counter(self) -> None:
        """测试各种操作中错误计数器递增。"""
        from context_forge.cache.redis_backend import RedisCache
        from unittest.mock import AsyncMock

        cache = RedisCache()
        mock_client = AsyncMock()
        cache._client = mock_client

        # 模拟 set 失败
        mock_client.set = AsyncMock(side_effect=Exception("Set error"))
        with pytest.warns(UserWarning, match="Redis 缓存写入失败"):
            entry = CacheEntry(value="test")
            await cache.set("key1", entry)
        assert cache._errors == 1

        # 模拟 get 失败
        mock_client.get = AsyncMock(return_value="invalid_json_{")
        with pytest.warns(UserWarning, match="Redis 缓存读取失败"):
            await cache.get("key2")
        assert cache._errors == 2

        # 模拟 delete 失败
        mock_client.delete = AsyncMock(side_effect=Exception("Delete error"))
        with pytest.warns(UserWarning, match="Redis 缓存删除失败"):
            await cache.delete("key3")
        # delete 中错误处理不计数，只发 warning

    @pytest.mark.asyncio
    async def test_redis_cache_serialize_entry_with_iso_format(self) -> None:
        """测试序列化时日期格式正确（ISO 格式）。"""
        from context_forge.cache.redis_backend import RedisCache
        from datetime import datetime, timezone

        cache = RedisCache()
        now = datetime.now(timezone.utc)
        entry = CacheEntry(value="test", created_at=now)

        serialized = cache._serialize_entry(entry)

        # 验证 created_at 是 ISO 格式字符串
        assert isinstance(serialized["created_at"], str)
        # 反序列化应该能还原日期
        restored_dt = datetime.fromisoformat(serialized["created_at"])
        assert restored_dt.year == now.year
        assert restored_dt.month == now.month
        assert restored_dt.day == now.day

    @pytest.mark.asyncio
    async def test_redis_cache_default_ttl_none(self) -> None:
        """测试默认 TTL 为 None 时的行为。"""
        from context_forge.cache.redis_backend import RedisCache
        from unittest.mock import AsyncMock

        cache = RedisCache(default_ttl=None)
        mock_client = AsyncMock()
        cache._client = mock_client

        entry = CacheEntry(value="test")
        # 不指定 ttl，且 default_ttl=None，应该不传递 ex 参数
        await cache.set("key1", entry)

        call_args = mock_client.set.call_args
        # 验证 ex 参数不存在或为 None
        if "ex" in call_args[1]:
            assert call_args[1]["ex"] is None

    @pytest.mark.asyncio
    async def test_redis_cache_get_with_redis_kwargs(self) -> None:
        """测试初始化时传递 redis_kwargs。"""
        from context_forge.cache.redis_backend import RedisCache

        redis_kwargs = {"socket_connect_timeout": 5, "socket_keepalive": True}
        cache = RedisCache(
            redis_url="redis://localhost:6379/0",
            **redis_kwargs,
        )

        assert cache._redis_kwargs == redis_kwargs

    @pytest.mark.asyncio
    async def test_redis_cache_close_idempotent(self) -> None:
        """测试多次 close 调用安全。"""
        from context_forge.cache.redis_backend import RedisCache
        from unittest.mock import AsyncMock

        cache = RedisCache()
        mock_client = AsyncMock()
        cache._client = mock_client

        await cache.close()
        await cache.close()  # 第二次调用

        # 第二次调用时 _client 仍然指向 mock_client
        # 但 close() 会重新检查 if self._client is not None
        assert mock_client.close.call_count == 2

    @pytest.mark.asyncio
    async def test_redis_cache_stats_single_scan_batch(self) -> None:
        """测试 stats 仅返回一次（cursor=0）。"""
        from context_forge.cache.redis_backend import RedisCache
        from unittest.mock import AsyncMock

        cache = RedisCache(key_prefix="test:")
        mock_client = AsyncMock()
        cache._client = mock_client

        # 模拟 SCAN 第一次就返回 cursor=0（单批）
        mock_client.scan = AsyncMock(
            return_value=(0, ["test:key1", "test:key2", "test:key3"])
        )

        cache._hits = 100
        cache._sets = 50

        stats = await cache.stats()

        assert stats.hits == 100
        assert stats.sets == 50
        assert stats.current_size == 3
        # 验证只调用一次 scan
        mock_client.scan.assert_called_once()

    @pytest.mark.asyncio
    async def test_redis_cache_stats_empty_result_in_middle(self) -> None:
        """测试 stats SCAN 中途返回空键列表。"""
        from context_forge.cache.redis_backend import RedisCache
        from unittest.mock import AsyncMock

        cache = RedisCache(key_prefix="cf:")
        mock_client = AsyncMock()
        cache._client = mock_client

        # 模拟 SCAN 返回空列表，但 cursor 不为 0
        mock_client.scan = AsyncMock(
            side_effect=[
                (100, ["cf:k1", "cf:k2"]),
                (200, []),  # 空列表
                (0, ["cf:k3"]),
            ]
        )

        stats = await cache.stats()

        # 空列表不应该增加 current_size
        assert stats.current_size == 3
        assert mock_client.scan.call_count == 3

    @pytest.mark.asyncio
    async def test_redis_cache_protocol_compliance(self) -> None:
        """测试 RedisCache 遵守 CacheBackend Protocol。"""
        from context_forge.cache.redis_backend import RedisCache
        from context_forge.cache.base import CacheBackend
        import inspect

        cache = RedisCache()

        # 验证所有 Protocol 方法都存在且为异步
        protocol_methods = ["get", "set", "delete", "clear", "stats"]
        for method_name in protocol_methods:
            assert hasattr(cache, method_name), f"Missing method: {method_name}"
            method = getattr(cache, method_name)
            assert inspect.iscoroutinefunction(method), f"{method_name} should be async"

    @pytest.mark.asyncio
    async def test_redis_cache_stats_get_client_returns_none(self) -> None:
        """测试 stats 中 _get_client() 返回 None 的分支覆盖。"""
        from context_forge.cache.redis_backend import RedisCache
        from unittest.mock import AsyncMock, patch

        cache = RedisCache()
        cache._hits = 10
        cache._misses = 5
        cache._sets = 15

        # Mock _get_client 返回 None
        with patch.object(cache, "_get_client", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None

            stats = await cache.stats()

            # 验证统计信息正确（但 current_size 为 0）
            assert stats.hits == 10
            assert stats.misses == 5
            assert stats.sets == 15
            assert stats.current_size == 0
            assert stats.evictions == 0
            assert stats.max_size == 0
