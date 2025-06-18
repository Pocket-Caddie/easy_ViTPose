import abc
import os
from typing import Optional
import typing

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .configs.ViTPose_common import data_cfg
from .sort import Sort
from .vit_models.model import ViTPose
from .vit_utils.inference import draw_bboxes, pad_image
from .vit_utils.top_down_eval import keypoints_from_heatmaps
from .vit_utils.util import dyn_model_import, infer_dataset_by_path
from .vit_utils.visualization import draw_points_and_skeleton, joints_dict

try:
    import torch_tensorrt
except ModuleNotFoundError:
    pass

try:
    import onnxruntime
except ModuleNotFoundError:
    pass

__all__ = ['VitInference']
np.bool = np.bool_
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


DETC_TO_YOLO_YOLOC = {
    'human': [0],
    'cat': [15],
    'dog': [16],
    'horse': [17],
    'sheep': [18],
    'cow': [19],
    'elephant': [20],
    'bear': [21],
    'zebra': [22],
    'giraffe': [23],
    'animals': [15, 16, 17, 18, 19, 20, 21, 22, 23]
}


class VitInference:
    """
    ViTPose + YOLOv8 inference with optional SORT tracking.
    Only modifications versus the original:
        • dedicated CUDA stream for ViTPose forward
        • one-time warm-up to build the TRT engine
    """

    def __init__(self, model: str,
                 yolo: str,
                 model_name: Optional[str] = None,
                 det_class: Optional[str] = None,
                 dataset: Optional[str] = None,
                 yolo_size: Optional[int] = 320,
                 device: Optional[str] = None,
                 is_video: Optional[bool] = False,
                 single_pose: Optional[bool] = False,
                 yolo_step: Optional[int] = 1):
        assert os.path.isfile(model), f'The model file {model} does not exist'
        assert os.path.isfile(yolo), f'The YOLOv8 model {yolo} does not exist'

        # Device priority: cuda > mps > cpu
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = device

        self.yolo = YOLO(yolo, task='detect')
        self.yolo_size = yolo_size
        self.yolo_step = yolo_step
        self.is_video = is_video
        self.single_pose = single_pose
        self.reset()

        # State saving during inference
        self.save_state = True
        self._img = None
        self._yolo_res = None
        self._tracker_res = None
        self._keypoints = None

        # Decide which model format we have
        use_onnx = model.endswith('.onnx')
        use_trt  = model.endswith('.ts')    # scripted TRT module from torch_tensorrt

        # Extract dataset name
        if dataset is None:
            dataset = infer_dataset_by_path(model)
        assert dataset in ['mpii', 'coco', 'coco_25', 'wholebody',
                           'aic', 'ap10k', 'apt36k', 'custom']

        self.dataset = dataset

        if det_class is None:
            det_class = 'animals' if dataset in ['ap10k', 'apt36k'] else 'human'
        self.yolo_classes = DETC_TO_YOLO_YOLOC[det_class]

        assert model_name in [None, 's', 'b', 'l', 'h'], \
            f'The model name {model_name} is not valid'

        # For checkpoints we still need the config
        if model_name is None:
            assert use_onnx or use_trt, \
                'Specify model_name when using a .pth checkpoint'
        else:
            model_cfg = dyn_model_import(self.dataset, model_name)

        self.target_size = data_cfg['image_size']

        # ── Build the ViTPose runtime ──────────────────────────────
        if use_onnx:
            self._ort_session = onnxruntime.InferenceSession(
                model,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            inf_fn = self._inference_onnx

            # no TRT stream in this branch
            self.trt_stream = None

        else:
            self._vit_pose = ViTPose(model_cfg)
            self._vit_pose.eval()

            if use_trt:
                # scripted module already contains TRT engine
                self._vit_pose = torch.jit.load(model, map_location=device)
            else:
                ckpt = torch.load(model, map_location='cpu', weights_only=True)
                if 'state_dict' in ckpt:
                    ckpt = ckpt['state_dict']
                self._vit_pose.load_state_dict(ckpt, strict=False)
                self._vit_pose.to(torch.device(device))

            # create a dedicated CUDA stream & warm-up
            if self.device == 'cuda':
                self.trt_stream = torch.cuda.Stream()
                with torch.cuda.stream(self.trt_stream):
                    # dummy = torch.zeros(
                    #     (1, 3, *self.target_size),
                    #     dtype=torch.float32,
                    #     device=self.device
                    # )
                    dummy = torch.zeros(
                        (1, 3, self.target_size[1], self.target_size[0]),
                        dtype=torch.float32,
                        device=self.device
                    )
                    _ = self._vit_pose(dummy)      # builds TRT engine
                torch.cuda.current_stream().wait_stream(self.trt_stream)
            else:
                self.trt_stream = None

            inf_fn = self._inference_torch

        self._inference = inf_fn  # type: ignore

    # ───────────── Misc helpers ──────────────────────────────────
    def reset(self):
        min_hits = 3 if self.yolo_step == 1 else 1
        use_tracker = self.is_video and not self.single_pose
        self.tracker = Sort(max_age=self.yolo_step,
                            min_hits=min_hits,
                            iou_threshold=0.3) if use_tracker else None
        self.frame_counter = 0

    @classmethod
    def postprocess(cls, heatmaps, org_w, org_h):
        points, prob = keypoints_from_heatmaps(
            heatmaps=heatmaps,
            center=np.array([[org_w // 2, org_h // 2]]),
            scale=np.array([[org_w, org_h]]),
            unbiased=True, use_udp=True)
        return np.concatenate([points[:, :, ::-1], prob], axis=2)

    @abc.abstractmethod
    def _inference(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # ───────────── Main public entrypoint ────────────────────────
    def inference(self, img: np.ndarray) -> dict[typing.Any, typing.Any]:
        res_pd = np.empty((0, 5))
        results = None
        if (self.tracker is None or
           (self.frame_counter % self.yolo_step == 0 or self.frame_counter < 3)):
            results = self.yolo(
                img[..., ::-1],
                verbose=False,
                imgsz=self.yolo_size,
                device=self.device if self.device != 'cuda' else 0,
                classes=self.yolo_classes
            )[0]
            res_pd = np.array([r[:5].tolist()
                               for r in results.boxes.data.cpu().numpy()
                               if r[4] > 0.35]).reshape((-1, 5))
        self.frame_counter += 1

        frame_keypoints = {}
        scores_bbox = {}
        ids = None
        if self.tracker is not None:
            res_pd = self.tracker.update(res_pd)
            ids = res_pd[:, 5].astype(int).tolist()

        bboxes = res_pd[:, :4].round().astype(int)
        scores = res_pd[:, 4].tolist()
        pad_bbox = 10

        ids = ids if ids is not None else range(len(bboxes))

        for bbox, pid, score in zip(bboxes, ids, scores):
            bbox[[0, 2]] = np.clip(bbox[[0, 2]] + [-pad_bbox, pad_bbox],
                                   0, img.shape[1])
            bbox[[1, 3]] = np.clip(bbox[[1, 3]] + [-pad_bbox, pad_bbox],
                                   0, img.shape[0])

            img_inf = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            img_inf, (left_pad, top_pad) = pad_image(img_inf, 3 / 4)

            keypoints = self._inference(img_inf)[0]
            keypoints[:, :2] += bbox[:2][::-1] - [top_pad, left_pad]
            frame_keypoints[pid] = keypoints
            scores_bbox[pid] = score

        if self.save_state:
            self._img = img
            self._yolo_res = results
            self._tracker_res = (bboxes, ids, scores)
            self._keypoints = frame_keypoints
            self._scores_bbox = scores_bbox

        return frame_keypoints

    # ───────────── Drawing helpers (unchanged) ──────────────────
    def draw(self, show_yolo=True, show_raw_yolo=False, confidence_threshold=0.5):
        img = self._img.copy()
        bboxes, ids, scores = self._tracker_res

        if self._yolo_res is not None and (show_raw_yolo or
                                           (self.tracker is None and show_yolo)):
            img = np.array(self._yolo_res.plot())[..., ::-1]

        if show_yolo and self.tracker is not None:
            img = draw_bboxes(img, bboxes, ids, scores)

        img = np.array(img)[..., ::-1]
        for idx, k in self._keypoints.items():
            img = draw_points_and_skeleton(
                img.copy(), k,
                joints_dict()[self.dataset]['skeleton'],
                person_index=idx,
                points_color_palette='gist_rainbow',
                skeleton_color_palette='gist_rainbow',
                points_palette_samples=10,
                confidence_threshold=confidence_threshold)
        return img[..., ::-1]

    # ───────────── Pre-/post-processing helpers ─────────────────
    def pre_img(self, img):
        org_h, org_w = img.shape[:2]
        img_input = cv2.resize(img, self.target_size,
                               interpolation=cv2.INTER_LINEAR) / 255
        img_input = ((img_input - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
        return img_input, org_h, org_w

    # ───────────── Inference back-ends ──────────────────────────
    @torch.no_grad()
    def _inference_torch(self, img: np.ndarray) -> np.ndarray:
        img_input, org_h, org_w = self.pre_img(img)
        img_input = torch.from_numpy(img_input).to(torch.device(self.device))

        if self.trt_stream is not None:
            with torch.cuda.stream(self.trt_stream):
                heatmaps_t = self._vit_pose(img_input)
            torch.cuda.current_stream().wait_stream(self.trt_stream)
            heatmaps = heatmaps_t.detach().cpu().numpy()
        else:
            heatmaps = self._vit_pose(img_input).detach().cpu().numpy()

        return self.postprocess(heatmaps, org_w, org_h)

    def _inference_onnx(self, img: np.ndarray) -> np.ndarray:
        img_input, org_h, org_w = self.pre_img(img)
        ort_inputs = {self._ort_session.get_inputs()[0].name: img_input}
        heatmaps = self._ort_session.run(None, ort_inputs)[0]
        return self.postprocess(heatmaps, org_w, org_h)
