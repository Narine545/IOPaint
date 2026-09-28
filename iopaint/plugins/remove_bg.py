import os

import cv2
import numpy as np
import torch
from loguru import logger
from torch.hub import get_dir

from iopaint.accel import onnx_providers, torch_device
from iopaint.plugins.alpha_matting import (
    DEFAULT_BACKGROUND_THRESHOLD,
    DEFAULT_ERODE_SIZE,
    DEFAULT_FOREGROUND_THRESHOLD,
    DEFAULT_MAX_SIZE,
    cutout,
    is_pymatting_available,
    refine_alpha,
)
from iopaint.plugins.base_plugin import BasePlugin
from iopaint.schema import (
    AlphaMattingMethod,
    Device,
    RemoveBGModel,
    RunPluginRequest,
)


class RemoveBG(BasePlugin):
    name = "RemoveBG"
    support_gen_mask = True
    support_gen_image = True

    def __init__(
        self,
        model_name,
        device,
        alpha_matting: str = AlphaMattingMethod.none,
        fg_threshold: int = DEFAULT_FOREGROUND_THRESHOLD,
        bg_threshold: int = DEFAULT_BACKGROUND_THRESHOLD,
        erode_size: int = DEFAULT_ERODE_SIZE,
        matting_max_size: int = DEFAULT_MAX_SIZE,
        remove_color_spill: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.alpha_matting = str(getattr(alpha_matting, "value", alpha_matting))
        self.fg_threshold = fg_threshold
        self.bg_threshold = bg_threshold
        self.erode_size = erode_size
        self.matting_max_size = matting_max_size
        self.remove_color_spill = remove_color_spill

        if self.alpha_matting == AlphaMattingMethod.quality and not is_pymatting_available():
            logger.warning(
                "alpha_matting=quality requires pymatting (pip install pymatting), "
                "falling back to the fast guided-filter method."
            )
            self.alpha_matting = AlphaMattingMethod.fast.value

        if model_name.startswith("birefnet"):
            import rembg

            if rembg.__version__ < "2.0.59":
                raise ValueError(
                    "To use birefnet models, please upgrade rembg to >= 2.0.59. pip install -U rembg"
                )

        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, "checkpoints")
        os.environ.setdefault("U2NET_HOME", model_dir)

        self._init_session(model_name)

    # ------------------------------------------------------------------ setup

    def _init_session(self, model_name: str):
        self.device_warning()

        if model_name == RemoveBGModel.briaai_rmbg_1_4:
            from iopaint.plugins.briarmbg import (
                briarmbg_process,
                create_briarmbg_session,
            )

            self.torch_device = torch_device(self.device)
            self.session = create_briarmbg_session().to(self.torch_device)
            self.predict = lambda bgr: briarmbg_process(
                self.torch_device, bgr, self.session, only_mask=True
            )
        elif model_name == RemoveBGModel.briaai_rmbg_2_0:
            from iopaint.plugins.briarmbg2 import (
                briarmbg2_process,
                create_briarmbg2_session,
            )

            self.torch_device = torch_device(self.device)
            self.session = create_briarmbg2_session().to(self.torch_device)
            self.predict = lambda bgr: briarmbg2_process(
                self.torch_device, bgr, self.session, only_mask=True
            )
        else:
            from rembg import new_session, remove

            providers = onnx_providers(self.device)
            logger.info(f"rembg session: {model_name}, providers={providers}")
            try:
                self.session = new_session(model_name=model_name, providers=providers)
            except TypeError:
                # older rembg versions do not accept `providers`
                self.session = new_session(model_name=model_name)
            self.predict = lambda bgr: remove(
                bgr, session=self.session, only_mask=True
            )

    def switch_model(self, new_model_name):
        if self.model_name == new_model_name:
            return

        logger.info(
            f"Switching removebg model from {self.model_name} to {new_model_name}"
        )
        self._init_session(new_model_name)
        self.model_name = new_model_name

    # ------------------------------------------------------------- inference

    def _predict_alpha(self, rgb_np_img: np.ndarray) -> np.ndarray:
        bgr_np_img = cv2.cvtColor(rgb_np_img, cv2.COLOR_RGB2BGR)
        mask = self.predict(bgr_np_img)
        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if mask.dtype != np.uint8:
            mask = np.clip(mask, 0, 255).astype(np.uint8)
        if mask.shape[:2] != rgb_np_img.shape[:2]:
            mask = cv2.resize(
                mask,
                (rgb_np_img.shape[1], rgb_np_img.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        return mask

    def _matting_params(self, req: RunPluginRequest) -> dict:
        def pick(req_value, default):
            return default if req_value is None else req_value

        method = pick(req.alpha_matting, self.alpha_matting)
        method = str(getattr(method, "value", method))
        if method == AlphaMattingMethod.quality and not is_pymatting_available():
            logger.warning("pymatting is not installed, using the fast method instead")
            method = AlphaMattingMethod.fast.value
        return dict(
            method=method,
            foreground_threshold=pick(req.fg_threshold, self.fg_threshold),
            background_threshold=pick(req.bg_threshold, self.bg_threshold),
            erode_size=pick(req.erode_size, self.erode_size),
            max_size=self.matting_max_size,
        )

    @torch.inference_mode()
    def gen_image(self, rgb_np_img, req: RunPluginRequest) -> np.ndarray:
        alpha = self._predict_alpha(rgb_np_img)
        params = self._matting_params(req)
        remove_color_spill = (
            self.remove_color_spill
            if req.remove_color_spill is None
            else req.remove_color_spill
        )

        if params["method"] != AlphaMattingMethod.none:
            alpha = refine_alpha(rgb_np_img, alpha, **params)

        return cutout(rgb_np_img, alpha, remove_color_spill=remove_color_spill)

    @torch.inference_mode()
    def gen_mask(self, rgb_np_img, req: RunPluginRequest) -> np.ndarray:
        # 255 means foreground, 0 means background
        alpha = self._predict_alpha(rgb_np_img)
        params = self._matting_params(req)
        if params["method"] != AlphaMattingMethod.none:
            alpha = refine_alpha(rgb_np_img, alpha, **params)
        return alpha

    # ----------------------------------------------------------------- misc

    def check_dep(self):
        try:
            import rembg  # noqa: F401
        except ImportError:
            import traceback

            error_msg = traceback.format_exc()
            return f"Install rembg failed, Error details:\n{error_msg}"

    def device_warning(self):
        if self.device == Device.cuda and self.model_name not in [
            RemoveBGModel.briaai_rmbg_1_4,
            RemoveBGModel.briaai_rmbg_2_0,
        ]:
            logger.warning(
                "remove_bg_device=cuda only supports briaai models"
                "(briaai/RMBG-1.4 / briaai/RMBG-2.0), other models run through "
                "onnxruntime. Install onnxruntime-gpu to use the GPU."
            )
