# easy_ViTPose/utils/video_reader_fast.py
"""
Fast, drop-in video reader:
    • NVDEC via torchvision.io (CUDA)          ← fastest
    • NVDEC via decord                         ← second
    • threaded OpenCV fallback (your original) ← always works

Regardless of backend, next(reader) returns an RGB
NumPy array  H x W x 3  (uint8), just like before.
"""

from __future__ import annotations
import queue, threading, warnings, cv2, numpy as np, torch

# optional deps
try:
    import torchvision
    _has_torchvision = True
except ModuleNotFoundError:
    _has_torchvision = False

try:
    import decord
    _has_decord = True
except ModuleNotFoundError:
    _has_decord = False


rotation_map = {0: None,
                90: cv2.ROTATE_90_CLOCKWISE,
                180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE}


# ───────────────── back-ends ──────────────────────────────────
class _TorchVisionCuda:
    def __init__(self, fname, rot):
        import torchvision.io as tvio
        torchvision.set_video_backend("cuda")
        self.reader = tvio.VideoReader(fname, "video")
        self.rot = rot

    def __iter__(self):
        for pkt in self.reader:
            frm = pkt["data"]                   # C×H×W uint8 (CUDA)
            if self.rot:
                k = self.rot // 90              # 1,2,3
                frm = torch.rot90(frm, k, (1, 2))
            yield frm.permute(1, 2, 0).cpu().numpy()   # HWC → NumPy


class _DecordCuda:
    def __init__(self, fname, rot):
        self.vr = decord.VideoReader(fname, ctx=decord.gpu(0))
        self.rot_code = rot

    def __iter__(self):
        for f in self.vr:
            frm = f.asnumpy()                   # HWC uint8 (CUDA → host)
            if self.rot_code:
                frm = cv2.rotate(frm, rotation_map[self.rot_code])
            yield frm


class _OpenCVThreaded:
    def __init__(self, fname, rot, max_q=32):
        self.rot_code = rotation_map[rot]
        self.q = queue.Queue(max_q)
        threading.Thread(target=self._worker, args=(fname,), daemon=True).start()

    def _worker(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"Cannot open {path}")
        while True:
            ok, frm = cap.read()
            if not ok:
                break
            if self.rot_code:
                frm = cv2.rotate(frm, self.rot_code)
            frm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
            self.q.put(frm)
        self.q.put(None)

    def __iter__(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            yield item


# ───────────────── public class ───────────────────────────────
class VideoReader:
    """
    Drop-in replacement:
        vr = VideoReader("file.mp4", rotate=90)
        for frame in vr:
            ...  # frame is RGB NumPy array H×W×3
    """
    def __init__(self, file_name: str, rotate: int = 0):
        self.backend = "opencv-thread"
        self.reader = None

        if torch.cuda.is_available() and _has_torchvision:
            try:
                self.reader = _TorchVisionCuda(file_name, rotate)
                self.backend = "torchvision-cuda"
            except Exception as e:
                warnings.warn(f"torchvision cuda backend failed: {e}")

        if self.reader is None and torch.cuda.is_available() and _has_decord:
            try:
                self.reader = _DecordCuda(file_name, rotate)
                self.backend = "decord-cuda"
            except Exception as e:
                warnings.warn(f"decord cuda backend failed: {e}")

        if self.reader is None:              # fallback
            self.reader = _OpenCVThreaded(file_name, rotate)
            self.backend = "opencv-thread"

        print(f"VideoReader backend: {self.backend}")

    def __iter__(self):
        return iter(self.reader)
