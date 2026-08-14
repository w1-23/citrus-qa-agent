"""
Global GPU mutex — prevents multi-thread DML ONNX Session collisions.

Windows DML does NOT support concurrent inference on a single GPU context.
Multiple threads hitting different ONNX Sessions on the same DML device
will cause silent crashes, memory corruption, or hangs.

All DML ONNX operations must acquire this lock before inference.
"""
import threading

_GPU_LOCK = threading.Lock()



def release_gpu():
    """Release the global GPU lock. Call after ONNX DML inference completes."""
    _GPU_LOCK.release()


class GPULockGuard:
    """Context manager for GPU-protected operations."""

    def __enter__(self):
        _GPU_LOCK.acquire()
        return self

    def __exit__(self, *args):
        _GPU_LOCK.release()
