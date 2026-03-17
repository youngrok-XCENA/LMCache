# SPDX-License-Identifier: Apache-2.0

# Standard
from concurrent.futures import Future
from typing import Any, Callable, List, Optional, Sequence, Union
import asyncio
import re
import threading

# Third Party
from maru import MaruConfig, MaruHandler
from maru_lmcache import CxlMemoryAdapter
import torch

# First Party
from lmcache.integration.vllm.utils import get_size_bytes
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface

logger = init_logger(__name__)


class MaruBackend(AllocatorBackendInterface):
    """Maru CXL shared memory storage backend.

    Implements AllocatorBackendInterface with its own CxlMemoryAdapter.
    No LocalCPUBackend needed — data lives directly in CXL mmap memory.

    Put is async (Future): metadata registration via RPC.
    Get is sync: CXL memory direct read (no network I/O).

    Args:
        config: LMCache engine configuration. Must have maru_path set.
        metadata: LMCache engine metadata.
        loop: asyncio event loop for async put tasks.
        dst_device: Target device string (unused for CXL, kept for interface).
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        loop: asyncio.AbstractEventLoop,
        dst_device: str = "cuda",
    ):
        super().__init__(dst_device=dst_device)

        if config.use_layerwise:
            raise NotImplementedError(
                "MaruBackend does not yet support layerwise KV cache."
            )

        # 1. Config
        self.config = config
        self.loop = loop
        self._operation_timeout: float = float(
            (config.extra_config or {}).get("maru_operation_timeout", 10.0)
        )

        self._full_chunk_size_bytes: int = get_size_bytes(
            metadata.get_shapes(), metadata.get_dtypes()
        )
        assert self._full_chunk_size_bytes % metadata.chunk_size == 0
        self._single_token_size: int = (
            self._full_chunk_size_bytes // metadata.chunk_size
        )

        self._mla_worker_id_as0_mode: bool = (
            config.get_extra_config_value(
                "remote_enable_mla_worker_id_as0", metadata.use_mla
            )
            and metadata.use_mla
            and metadata.world_size > 1
            and metadata.worker_id != 0
        )

        # 2. Handler
        self._handler = self._create_handler(config)

        # 3. Allocator
        self.memory_allocator = self.initialize_allocator(config, metadata)

        # 4. State
        self._connected: bool = True
        self.put_lock = threading.Lock()
        self.put_tasks: set[CacheEngineKey] = set()

        # 5. Metrics
        self._rpc_errors: int = 0
        self._stats_monitor = LMCStatsMonitor.GetOrCreate()

    def __str__(self) -> str:
        return self.__class__.__name__

    @property
    def rpc_errors(self) -> int:
        """Number of RPC errors since startup."""
        return self._rpc_errors

    @staticmethod
    def _parse_pool_size(raw: Optional[str]) -> int:
        """Parse human-readable pool size (e.g. '4G', '512M') to bytes."""
        _DEFAULT = 4 * 1024**3
        if raw is None:
            return _DEFAULT
        if isinstance(raw, (int, float)):
            return int(raw)
        s = str(raw).strip().upper()
        match = re.match(r"^(\d+(?:\.\d+)?)\s*([KMGT]?)B?$", s)
        if not match:
            try:
                return int(s)
            except ValueError:
                logger.warning("Cannot parse maru_pool_size=%r, using default", raw)
                return _DEFAULT
        value, unit = float(match.group(1)), match.group(2)
        multipliers = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
        return int(value * multipliers.get(unit, 1))

    # =========================================================================
    # Initialization helpers
    # =========================================================================

    def _create_handler(
        self,
        config: LMCacheEngineConfig,
    ) -> "MaruHandler":
        """Create and connect a MaruHandler.

        Args:
            config: LMCache engine configuration.

        Returns:
            Connected MaruHandler instance.

        Raises:
            RuntimeError: If MaruHandler connection fails.
        """
        assert config.maru_path is not None, "maru_path must be set for MaruBackend"

        # Convert maru:// scheme to tcp:// for ZMQ
        server_url = config.maru_path
        if server_url.startswith("maru://"):
            server_url = "tcp://" + server_url[len("maru://"):]

        extra = config.extra_config or {}
        maru_config = MaruConfig(
            server_url=server_url,
            instance_id=extra.get("maru_instance_id"),
            pool_size=self._parse_pool_size(config.maru_pool_size),
            chunk_size_bytes=self._full_chunk_size_bytes,
            auto_connect=False,
            timeout_ms=extra.get("maru_timeout_ms", 5000),
            use_async_rpc=extra.get("maru_use_async_rpc", True),
            max_inflight=extra.get("maru_max_inflight", 64),
            eager_map=extra.get("maru_eager_map", True),
        )

        handler = MaruHandler(maru_config)
        if not handler.connect():
            raise RuntimeError(f"Failed to connect MaruHandler to {config.maru_path}")
        logger.debug("[Maru] Connected to %s", config.maru_path)
        return handler

    def _ensure_connected(self) -> bool:
        """Ensure the handler is connected, reconnecting if necessary.

        Returns:
            True if connected.
        """
        if self._connected:
            return True
        try:
            logger.info("[Maru] attempting reconnection to %s", self.config.maru_path)
            self._handler = self._create_handler(self.config)
            self._connected = True
            logger.info("[Maru] reconnected successfully")
            return True
        except Exception as e:
            logger.error("[Maru] reconnection failed: %s", e)
            return False

    # =========================================================================
    # AllocatorBackendInterface
    # =========================================================================

    def initialize_allocator(
        self, config: LMCacheEngineConfig, metadata: LMCacheMetadata
    ) -> MemoryAllocatorInterface:
        """Create CxlMemoryAdapter backed by the connected handler.

        Args:
            config: LMCache engine configuration.
            metadata: LMCache engine metadata.

        Returns:
            CxlMemoryAdapter instance.
        """
        shapes = metadata.get_shapes()
        dtypes = metadata.get_dtypes()
        fmt = MemoryFormat.KV_MLA_FMT if metadata.use_mla else MemoryFormat.KV_2LTD
        chunk_size = self._handler.get_chunk_size()

        return CxlMemoryAdapter(
            handler=self._handler,
            shapes=shapes,
            dtypes=dtypes,
            fmt=fmt,
            chunk_size=chunk_size,
        )

    def get_memory_allocator(self) -> MemoryAllocatorInterface:
        """Returns the underlying CxlMemoryAdapter."""
        return self.memory_allocator

    def get_allocator_backend(self) -> "MaruBackend":
        """Returns self as the allocator backend."""
        return self

    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        """Allocate CXL-backed memory via CxlMemoryAdapter.

        Args:
            shapes: Tensor shape(s).
            dtypes: Tensor dtype(s).
            fmt: Memory format.
            eviction: Unused (no eviction policy yet).
            busy_loop: Unused.

        Returns:
            MemoryObj backed by CXL memory, or None on failure.
        """
        obj = self.memory_allocator.allocate(shapes, dtypes, fmt)
        if obj is not None:
            logger.debug(
                "[Maru] allocate rid=%d pid=%d",
                *CxlMemoryAdapter.decode_address(obj.metadata.address),
            )
        else:
            logger.debug("[Maru] allocate failed shapes=%s dtypes=%s", shapes, dtypes)
        return obj

    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[list[MemoryObj]]:
        """Allocate multiple CXL-backed MemoryObjs.

        Args:
            shapes: Tensor shape(s) (same for each allocation).
            dtypes: Tensor dtype(s) (same for each allocation).
            batch_size: Number of allocations.
            fmt: Memory format.
            eviction: Unused.
            busy_loop: Unused.

        Returns:
            List of MemoryObj, or None if any allocation fails.
        """
        return self.memory_allocator.batched_allocate(shapes, dtypes, batch_size, fmt)

    # =========================================================================
    # Put (async)
    # =========================================================================

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """Check whether key is in ongoing put tasks.

        Args:
            key: The cache key.

        Returns:
            True if the key has a pending put task.
        """
        with self.put_lock:
            return key in self.put_tasks

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Future:
        """Submit a put task to register KV metadata with MaruServer.

        Data is already in CXL memory (zero-copy). This only registers
        the key -> location metadata via RPC.

        Args:
            key: The cache key.
            memory_obj: MemoryObj with data already written to CXL.
            on_complete_callback: Optional callback after registration.

        Returns:
            Future that completes when metadata is registered.
        """
        assert memory_obj.tensor is not None

        with self.put_lock:
            self.put_tasks.add(key)

        future = asyncio.run_coroutine_threadsafe(
            self._async_store(key, memory_obj, on_complete_callback),
            self.loop,
        )
        return future

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Union[List[Future], None]:
        """Submit batched put tasks.

        Args:
            keys: The cache keys.
            memory_objs: MemoryObjs with data already in CXL.
            transfer_spec: Unused.
            on_complete_callback: Optional per-key callback.

        Returns:
            List of Futures, one per key.
        """
        futures = []
        for key, memory_obj in zip(keys, memory_objs, strict=True):
            future = self.submit_put_task(
                key, memory_obj, on_complete_callback=on_complete_callback
            )
            futures.append(future)
        return futures

    async def _async_store(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """Register KV metadata with MaruServer (runs in event loop).

        Uses CxlMemoryAdapter.create_store_handle() to extract
        (region_id, page_index) from the MemoryObj's encoded address.

        Args:
            key: The cache key.
            memory_obj: MemoryObj backed by CXL memory.
            on_complete_callback: Optional callback after registration.
        """
        success = False
        try:
            allocator = self.memory_allocator
            assert isinstance(allocator, CxlMemoryAdapter)
            handle = allocator.create_store_handle(memory_obj)
            key_str = key.to_string()

            await asyncio.wait_for(
                asyncio.to_thread(self._handler.store, key_str, handle),
                timeout=self._operation_timeout,
            )
            success = True

            logger.debug(
                "[Maru] store key=%s rid=%d pid=%d",
                key,
                handle.region_id,
                handle.page_index,
            )

        except Exception as e:
            self._rpc_errors += 1
            self._connected = False
            logger.error("[Maru] store failed key=%s: %s", key, e)
        finally:
            with self.put_lock:
                self.put_tasks.discard(key)

            if success and on_complete_callback is not None:
                try:
                    on_complete_callback(key)
                except Exception as e:
                    logger.warning("on_complete_callback failed for key %s: %s", key, e)

    # =========================================================================
    # Get (sync)
    # =========================================================================

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        """Blocking get: read KV cache directly from CXL memory.

        Queries MaruServer for metadata, then returns a MemoryObj
        via CxlMemoryAdapter.get_by_location().

        Args:
            key: The cache key.

        Returns:
            MemoryObj backed by CXL memory, or None if not found.
        """
        if not self._ensure_connected():
            return None

        if self._mla_worker_id_as0_mode:
            key = key.with_new_worker_id(0)

        key_str = key.to_string()
        mem_info = self._handler.retrieve(key_str)
        if mem_info is None:
            logger.debug("[Maru] get_blocking miss key=%s", key)
            return None

        allocator = self.memory_allocator
        assert isinstance(allocator, CxlMemoryAdapter)

        memory_obj = allocator.get_by_location(
            region_id=mem_info.region_id,
            page_index=mem_info.page_index,
            actual_size=len(mem_info.view),
            single_token_size=self._single_token_size,
        )
        if memory_obj is None:
            logger.debug(
                "[Maru] get_blocking pool miss rid=%d pid=%d",
                mem_info.region_id,
                mem_info.page_index,
            )
            return None

        memory_obj.ref_count_up()
        memory_obj.pin()

        logger.debug(
            "[Maru] get_blocking rid=%d pid=%d size=%d",
            mem_info.region_id,
            mem_info.page_index,
            len(mem_info.view),
        )
        return memory_obj

    # =========================================================================
    # Async lookup API (used by StorageManager.async_lookup_and_prefetch)
    # =========================================================================

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        """Check how many prefix keys exist on MaruServer.

        Uses batch_exists for a single RPC call. Prefix-based: returns
        the count of contiguous keys starting from index 0 that exist.
        Stops at first miss.

        Args:
            lookup_id: Unique request identifier.
            keys: Keys to check in prefix order.
            pin: Whether to pin. Not supported; logged as debug.

        Returns:
            Number of prefix-contiguous keys that exist.
        """
        if not keys:
            return 0

        def _contains_prefix() -> int:
            if self._mla_worker_id_as0_mode:
                key_strs = [k.with_new_worker_id(0).to_string() for k in keys]
            else:
                key_strs = [k.to_string() for k in keys]
            results = self._handler.batch_exists(key_strs)
            count = 0
            for exists in results:
                if not exists:
                    break
                count += 1
            return count

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_contains_prefix),
                timeout=self._operation_timeout,
            )
        except asyncio.TimeoutError:
            self._rpc_errors += 1
            logger.warning(
                "[Maru] batched_async_contains timed out for lookup_id=%s",
                lookup_id,
            )
            return 0

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        """Non-blocking batched get via CXL direct read.

        Uses batch_retrieve for a single RPC call. Stops at first miss
        and returns the prefix that was successfully retrieved.

        Args:
            lookup_id: Unique request identifier.
            keys: Keys to retrieve (already confirmed by contains).
            transfer_spec: Unused.

        Returns:
            List of MemoryObjs backed by CXL memory.
        """
        if not keys:
            return []

        def _get_batch() -> list[MemoryObj]:
            if self._mla_worker_id_as0_mode:
                key_strs = [k.with_new_worker_id(0).to_string() for k in keys]
            else:
                key_strs = [k.to_string() for k in keys]

            raw_results = self._handler.batch_retrieve(key_strs)

            allocator = self.memory_allocator
            assert isinstance(allocator, CxlMemoryAdapter)

            results: list[MemoryObj] = []
            for mem_info in raw_results:
                if mem_info is None:
                    break
                memory_obj = allocator.get_by_location(
                    region_id=mem_info.region_id,
                    page_index=mem_info.page_index,
                    actual_size=len(mem_info.view),
                    single_token_size=self._single_token_size,
                )
                if memory_obj is None:
                    break
                memory_obj.ref_count_up()
                memory_obj.pin()
                results.append(memory_obj)
            return results

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_get_batch),
                timeout=self._operation_timeout,
            )
        except asyncio.TimeoutError:
            self._rpc_errors += 1
            logger.warning(
                "[Maru] batched_get_non_blocking timed out for lookup_id=%s",
                lookup_id,
            )
            return []

    # =========================================================================
    # Contains / Pin / Unpin / Remove
    # =========================================================================

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """Check if key exists on MaruServer.

        Args:
            key: The cache key.
            pin: If True, pin the entry. (TODO: delegate to handler)

        Returns:
            True if key exists.
        """
        if not self._ensure_connected():
            return False

        if self._mla_worker_id_as0_mode:
            key = key.with_new_worker_id(0)

        return self._handler.exists(key.to_string())

    def pin(self, key: CacheEngineKey) -> bool:
        """Pin a key to prevent eviction.

        Attempts to delegate to MaruHandler if the method is available,
        otherwise returns False.

        Args:
            key: The cache key.

        Returns:
            True if pinned successfully.
        """
        if hasattr(self._handler, 'pin'):
            try:
                return self._handler.pin(key.to_string())
            except Exception as e:
                logger.debug("[Maru] pin failed for key=%s: %s", key, e)
                return False
        logger.debug("[Maru] pin not supported by handler")
        return False

    def unpin(self, key: CacheEngineKey) -> bool:
        """Unpin a key to allow eviction.

        Attempts to delegate to MaruHandler if the method is available,
        otherwise returns False.

        Args:
            key: The cache key.

        Returns:
            True if unpinned successfully.
        """
        if hasattr(self._handler, 'unpin'):
            try:
                return self._handler.unpin(key.to_string())
            except Exception as e:
                logger.debug("[Maru] unpin failed for key=%s: %s", key, e)
                return False
        logger.debug("[Maru] unpin not supported by handler")
        return False

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        """Remove a key from MaruServer.

        Args:
            key: The cache key.
            force: Whether to force removal.

        Returns:
            True if removed successfully.
        """
        if not self._ensure_connected():
            return False

        key_str = key.to_string()
        result = self._handler.delete(key_str)
        logger.debug("[Maru] remove key=%s success=%s", key, result)
        return result

    # =========================================================================
    # Health check
    # =========================================================================

    async def healthcheck(self) -> bool:
        """Check if MaruServer is reachable.

        Returns:
            True if the server responds to health check.
        """
        try:
            healthy = await asyncio.wait_for(
                asyncio.to_thread(self._handler.healthcheck),
                timeout=self._operation_timeout,
            )
            if not healthy:
                self._stats_monitor.update_remote_ping_error_code(2)
                logger.warning("[Maru] healthcheck failed")
            else:
                self._stats_monitor.update_remote_ping_error_code(0)
            return healthy
        except asyncio.TimeoutError:
            self._rpc_errors += 1
            self._stats_monitor.update_remote_ping_error_code(2)
            logger.warning("[Maru] healthcheck timed out")
            return False
        except Exception as e:
            self._rpc_errors += 1
            self._stats_monitor.update_remote_ping_error_code(2)
            logger.warning("[Maru] healthcheck error: %s", e)
            return False

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def close(self) -> None:
        """Close the backend and underlying MaruHandler."""
        with self.put_lock:
            pending = len(self.put_tasks)
        if pending > 0:
            logger.warning(
                "[Maru] closing with %d in-flight put tasks still pending",
                pending,
            )
        self.memory_allocator.close()
        self._handler.close()
        self._connected = False
        logger.info("MaruBackend closed.")
