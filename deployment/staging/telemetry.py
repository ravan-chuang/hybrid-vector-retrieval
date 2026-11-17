"""Sample process, host, and rerank queue counters in the staging API."""

import asyncio
from contextlib import aclosing, asynccontextmanager, suppress
import ctypes
import json
import os
from pathlib import Path
import resource
import sys
import time

from backend.api.services import dynamic_batch


_PERIOD_SECONDS = 1.0
# Layout and flavors from the Darwin Mach host_info and vm_statistics headers.
_DARWIN_HOST_CPU_LOAD_INFO = 3
_DARWIN_HOST_VM_INFO64 = 4


class _DarwinVM(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_uint32)
        for name in ("free", "active", "inactive", "wired")
    ] + [
        (name, ctypes.c_uint64)
        for name in ("zero_fill", "reactivations", "pageins", "pageouts", "faults",
                     "cow_faults", "lookups", "hits", "purges")
    ] + [
        (name, ctypes.c_uint32) for name in ("purgeable", "speculative")
    ] + [
        (name, ctypes.c_uint64)
        for name in ("decompressions", "compressions", "swapins", "swapouts")
    ] + [
        (name, ctypes.c_uint32)
        for name in ("compressor_pages", "throttled", "external", "internal")
    ] + [
        (name, ctypes.c_uint64)
        for name in ("uncompressed_in_compressor", "swapped_count")
    ]


class _DarwinCounters:
    def __init__(self):
        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        lib.mach_host_self.restype = ctypes.c_uint
        for reader in (lib.host_statistics, lib.host_statistics64):
            reader.argtypes = (ctypes.c_uint, ctypes.c_int, ctypes.c_void_p,
                               ctypes.POINTER(ctypes.c_uint))
            reader.restype = ctypes.c_int
        self.lib = lib
        self.host = lib.mach_host_self()
        self.cpu = (ctypes.c_uint32 * 4)()
        self.vm = _DarwinVM()
        self.page_size = resource.getpagesize()

    def sample(self):
        fields = {}
        cpu_count = ctypes.c_uint(4)
        if self.lib.host_statistics(self.host, _DARWIN_HOST_CPU_LOAD_INFO,
                                    ctypes.byref(self.cpu), ctypes.byref(cpu_count)) == 0:
            fields["host_cpu_ticks"] = list(self.cpu)
        vm_count = ctypes.c_uint(ctypes.sizeof(self.vm) // ctypes.sizeof(ctypes.c_int))
        if self.lib.host_statistics64(self.host, _DARWIN_HOST_VM_INFO64,
                                      ctypes.byref(self.vm), ctypes.byref(vm_count)) == 0:
            fields["host_vm_page_size"] = self.page_size
            fields["host_vm"] = {key: getattr(self.vm, key) for key in (
                "free", "inactive", "wired", "compressor_pages", "pageins",
                "pageouts", "swapins", "swapouts", "compressions", "decompressions",
            )}
        return fields


async def _sample_lines():
    host = _DarwinCounters() if sys.platform == "darwin" else None
    clock = asyncio.get_running_loop().time
    deadline = clock()
    while True:
        now = clock()
        process = resource.getrusage(resource.RUSAGE_SELF)
        row = {
            "time": time.time(),
            "monotonic": time.perf_counter(),
            "pid": os.getpid(),
            "loop_lag_ms": max(0.0, now - deadline) * 1000,
            "api_cpu_seconds": process.ru_utime + process.ru_stime,
            "api_peak_rss_bytes": process.ru_maxrss * (1 if sys.platform == "darwin" else 1024),
            "api_major_faults": process.ru_majflt,
            "api_involuntary_switches": process.ru_nivcsw,
            **dynamic_batch.batch_queue_snapshot(),
        }
        if host is not None:
            row.update(host.sample())
        yield json.dumps(row, separators=(",", ":")) + "\n"
        deadline = max(deadline + _PERIOD_SECONDS, now)
        await asyncio.sleep(max(0.0, deadline - clock()))


async def _write_samples(log, samples):
    async for line in samples:
        log.write(line)


def lifespan_with_telemetry(lifespan, path: Path):
    @asynccontextmanager
    async def lifespan_context(app):
        try:
            log = path.open("a", buffering=1, encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"Cannot open STAGING_RERANK_TELEMETRY output {path}: {exc}") from exc

        with log:
            async with lifespan(app) as state, aclosing(_sample_lines()) as samples:
                try:
                    log.write(await anext(samples))
                except OSError as exc:
                    raise RuntimeError(
                        f"Cannot write STAGING_RERANK_TELEMETRY output {path}: {exc}"
                    ) from exc

                sampler = asyncio.create_task(_write_samples(log, samples))
                try:
                    yield state
                finally:
                    sampler.cancel()
                    with suppress(asyncio.CancelledError):
                        await sampler

    return lifespan_context
