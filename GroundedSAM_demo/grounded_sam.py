"""GroundedSAM utilities: prompt-driven detection + segmentation.

This module integrates GroundingDINO for text-prompted object detection and
Segment Anything (SAM/MobileSAM/YOLO-SAM) for mask generation. It provides
simple classes to:

- Download and load GroundingDINO checkpoints from Hugging Face.
- Run detection from a natural language caption to get bounding boxes.
- Use SAM-family predictors to obtain segmentation masks for those boxes.

Typical flow
-----------
1) GroundingDINO detects boxes from a prompt (e.g., "red box").
2) A SAM predictor refines those boxes into pixel-accurate masks.

Expected checkpoints layout under ``checkpoint_dir``
---------------------------------------------------
- GroundingDINO weights are downloaded automatically into ``checkpoint_dir/gdino``.
- SAM-family weights are expected in:
    - MobileSAM: ``checkpoint_dir/mobilesam/mobile_sam.pt``
    - SAM ViT-B/L/H: ``checkpoint_dir/sam/sam_vit_{b|l|h}_*.pth``
    - YOLO-SAM: ``checkpoint_dir/yolosam/<sam_variant>.pt``

All images are assumed to be numpy RGB arrays with shape [H, W, 3].
"""
import os
import os.path as osp
import logging
import warnings
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.ops import box_convert
from huggingface_hub import hf_hub_download

from groundingdino.models import build_model
from groundingdino.datasets import transforms as T
from groundingdino.util.slconfig import SLConfig as gdino_SLConfig
from groundingdino.util.inference import predict as gdino_predict
from groundingdino.util.utils import clean_state_dict as gdino_clean_state_dict

from GroundedSAM_demo.utils import BatchedData

from typing_extensions import Optional, List, Dict, Any, Union, Tuple


class GroundingDINO:
    """
    Text-prompted object detection powered by GroundingDINO.

    Based on: https://github.com/IDEA-Research/GroundingDINO
    """
    def __init__(self,
                 checkpoint_dir: str,
                 config_dir: str,
                 use_vitb: bool = False,
                 box_threshold: float = 0.1,
                 text_threshold: float = 0.1,
                 device: Optional[str] = "cpu") -> None:
        """Create a GroundingDINO detector.

        Args:
            checkpoint_dir: Directory to cache/download model checkpoints.
            config_dir: Directory containing GroundingDINO config files
                (e.g., ``cfg/gdino`` in this repo).
            use_vitb: If True, use Swin-B config/weights; otherwise Swin-T OGC.
            box_threshold: Minimum box confidence threshold.
            text_threshold: Minimum text matching threshold.
            device: Torch device string ("cuda"/"cpu"). If None, auto-detect.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        checkpoint_repo_id = "ShilongLiu/GroundingDINO"

        if use_vitb:
            checkpoint_filename = "groundingdino_swinb_cogcoor.pth"
            config_filename = "GroundingDINO_SwinB_cfg.py"
        else:
            checkpoint_filename = "groundingdino_swint_ogc.pth"
            config_filename = "GroundingDINO_SwinT_OGC.py"

        config_file_path = osp.join(config_dir, config_filename)

        # download checkpoint
        checkpoint_file_path = hf_hub_download(repo_id=checkpoint_repo_id,
                                               filename=checkpoint_filename,
                                               local_dir=osp.join(checkpoint_dir, "gdino"))
        # load model
        self.model = GroundingDINO.load_model(config_file_path, checkpoint_file_path, self.device)

        self.transform = T.Compose([T.RandomResize([800], max_size=1333),
                                    T.ToTensor(),
                                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def move_to_device(self, device):
        self.model.to(device)
        self.device = device
        return self

    @staticmethod
    def load_model(model_config_path: str,
                   checkpoint_file_path: str,
                   device: str) -> torch.nn.Module:
        """Load a GroundingDINO model and weights.

        Args:
            model_config_path: Path to model configuration file.
            checkpoint_file_path: Path to checkpoint file.
            device: Device type (e.g. "cuda" or "cpu").

        Returns:
            torch.nn.Module: Ready-to-infer GroundingDINO model.
        """
        # create model
        args = gdino_SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)

        # load model weights
        checkpoint = torch.load(checkpoint_file_path, map_location=device)
        log = model.load_state_dict(gdino_clean_state_dict(checkpoint["model"]), strict=False)
        logging.info("Model loaded from {} \n => {}".format(checkpoint_file_path, log))

        model.to(device)
        model.eval()

        return model

    def predict(self, image: np.ndarray, caption: str = "objects") -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Run text-prompted detection on an image.

        Args:
            image: RGB image as numpy array [H, W, 3].
            caption: Text prompt for object detection (e.g., "red box").

        Returns:
            Tuple[torch.Tensor, torch.Tensor, List[str]]: A tuple
            (bboxes_xyxy, confidences, phrases) where:
              - bboxes_xyxy: [B, 4] boxes in xyxy (float, image scale)
              - confidences: [B] box confidence scores
              - phrases: list[str] matched phrases per box
        """
        # prepare image
        processed_image, _ = self.transform(Image.fromarray(image.astype(np.uint8)), None)  # 3 x H' x W'

        # prediction
        boxes, confs, phrases = gdino_predict(model=self.model,
                                              image=processed_image,
                                              caption=caption,
                                              box_threshold=self.box_threshold,
                                              text_threshold=self.text_threshold,
                                              device=self.device)  # B x 4 (as xywh), B, B

        # move back to desired device
        boxes = boxes.to(self.device)  # B x 4
        confs = confs.to(self.device)  # B

        # fix bounding box scale
        height, width = image.shape[:2]
        boxes = boxes * torch.as_tensor([width, height, width, height], device=self.device)  # B x 4
        bboxes_xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")  # B x 4 (as xyxy)

        return bboxes_xyxy, confs, phrases  # B x 4 (as xyxy), B, B


class SAMPredictor:
    """
    Segment objects using the Segment Anything family with bounding-box prompts.

    """

    def __init__(self,
                 checkpoint_dir: str,
                 vit_model: str = "vit_t",
                 mask_threshold: float = 0.1,
                 device: Optional[str] = "cpu",
                 chunk_size: int = 8) -> None:
        """Initialize a SAM/MobileSAM predictor.

        Args:
            checkpoint_dir: Root directory containing SAM-family checkpoints.
            vit_model: One of {"vit_t", "vit_b", "vit_l", "vit_h"}. "vit_t"
                uses MobileSAM.
            mask_threshold: Minimum mask confidence to keep (None keeps all).
            device: Torch device string ("cuda"/"cpu"). If None, auto-detect.
            chunk_size: Batch size for box prompting (controls memory/speed).
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        sam_weight_path = {
            "vit_t": "mobilesam/mobile_sam.pt",
            "vit_b": "sam/sam_vit_b_01ec64.pth",
            "vit_h": "sam/sam_vit_h_4b8939.pth",
            "vit_l": "sam/sam_vit_l_0b3195.pth",
        }

        checkpoint_file = sam_weight_path[vit_model]
        checkpoint_path = osp.join(checkpoint_dir, checkpoint_file)

        # load model
        if "mobilesam" in checkpoint_file:
            # mobile sam (faster and smaller than FastSAM)
            from mobile_sam import sam_model_registry, SamPredictor
        else:
            # normal sam
            from segment_anything import sam_model_registry, SamPredictor

        self.sam = sam_model_registry[vit_model](checkpoint=checkpoint_path)
        self.sam.to(self.device)
        self.sam.eval()

        self.predictor = SamPredictor(self.sam)

        self.mask_threshold = mask_threshold
        self.chunk_size = chunk_size

        logging.info("Loading SAM predictor from {}.".format(checkpoint_path))

    def move_to_device(self, device):
        self.sam.to(device)
        self.device = device
        return self

    @torch.no_grad()
    def predict(self, image: np.ndarray, prompt_bboxes: Union[np.ndarray, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict segmentation masks for the input image.

        Args:
            image: RGB image [H, W, 3] as numpy array.
            prompt_bboxes: Bounding boxes in xyxy (numpy or torch tensor).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
              - masks: [B', H, W] boolean/float masks
              - confidences: [B'] mask confidences
              - indices: [B'] indices into the original input boxes kept after
                thresholding
        """
        input_boxes = torch.as_tensor(prompt_bboxes, device=self.predictor.device)  # B x 4 (as xyxy)
        transformed_boxes = self.predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])  # B x 4

        self.predictor.set_image(image)

        batched_transformed_boxes = BatchedData(batch_size=self.chunk_size, data=transformed_boxes)
        del transformed_boxes

        masks = [None] * len(batched_transformed_boxes)
        confs = [None] * len(batched_transformed_boxes)

        for index_batch in range(len(batched_transformed_boxes)):
            masks[index_batch], confs[index_batch], _ = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=batched_transformed_boxes[index_batch],
                multimask_output=False)  # chunk_size x 1 x H x W, chunk_size x 1

        self.predictor.reset_image()

        masks = torch.cat(masks, dim=0)  # B x 1 x H x W
        confs = torch.cat(confs, dim=0)  # B x 1

        # squeeze channel
        masks = masks.squeeze(1)  # B x H x W
        confs = confs[:, 0]  # B

        # filter mask with low confidence score
        if self.mask_threshold is None:
            indices = torch.arange(0, input_boxes.shape[0], device=masks.device)  # B
        else:
            indices = torch.where(confs > self.mask_threshold)[0]  # B'
            masks = masks[indices]  # B' x H x W
            confs = confs[indices]  # B'

        return masks, confs, indices  # (B|B') x H x W, (B|B'), (B|B')


class YOLOSAMPredictor:
    """
    Segment objects using the YOLO-SAM predictor with bounding-box prompts.

    """

    def __init__(self,
                 checkpoint_dir: str,
                 vit_model: str = "sam_b.pt",
                 mask_threshold: float = 0.1,
                 device: Optional[str] = "cpu",
                 chunk_size: int = 8) -> None:
        """Initialize a YOLO-SAM predictor.

        Args:
            checkpoint_dir: Root directory containing YOLO-SAM checkpoints.
            vit_model: YOLO-SAM checkpoint name (e.g., "sam_b.pt"). The
                extension ".pt" is optional.
                Choose one from: ['sam_h.pt', 'sam_l.pt', 'sam_b.pt', 'mobile_sam.pt', 'sam2_t.pt', 'sam2_s.pt', 'sam2_b.pt', 'sam2_l.pt', 'sam2.1_t.pt', 'sam2.1_s.pt', 'sam2.1_b.pt', 'sam2.1_l.pt']
            mask_threshold: Minimum mask confidence to keep (None keeps all).
            device: Torch device string ("cuda"/"cpu"). If None, auto-detect.
            chunk_size: Batch size for box prompting (controls memory/speed).
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        if vit_model[-3:] != ".pt":
            # does not end on '.pt'
            vit_model += ".pt"

        if "sam2" in vit_model:
            from ultralytics.models.sam import SAM2Predictor as SAMPredictor

        else:
            from ultralytics.models.sam import Predictor as SAMPredictor

        checkpoint_path = osp.join(checkpoint_dir, "yolosam", vit_model)
        os.makedirs(osp.dirname(checkpoint_path), exist_ok=True)

        overrides = {
            "conf": 1.0,  # filter later manually
            "task": "segment",
            "mode": "predict",
            "imgsz": 1024,
            "model": checkpoint_path,
            "device": device,
            "verbose": False,
            "save": False,
        }
        self.predictor = SAMPredictor(overrides=overrides)

        self.mask_threshold = mask_threshold
        self.chunk_size = chunk_size

        logging.info("Loaded SAM predictor checkpoint {}.".format(checkpoint_path))

    def move_to_device(self, device):
        if self.predictor.model:
            self.predictor.model.to(device)
        self.predictor.device = device
        self.device = device
        return self

    @torch.no_grad()
    def predict(self, image: np.ndarray, prompt_bboxes: Union[np.ndarray, torch.Tensor]):
        """Predict segmentation masks for the input image.

        Args:
            image: RGB image [H, W, 3] as numpy array.
            prompt_bboxes: Bounding boxes in xyxy (numpy or torch tensor).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
              - masks: [B', H, W] boolean/float masks
              - confidences: [B'] mask confidences
              - indices: [B'] indices into the original input boxes kept after
                thresholding
        """
        if torch.is_tensor(prompt_bboxes):
            prompt_bboxes = prompt_bboxes.detach().cpu().numpy()

        self.predictor.set_image(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # image is needed as BGR

        batched_prompt_bboxes = BatchedData(batch_size=self.chunk_size, data=prompt_bboxes)
        del prompt_bboxes

        masks = [None] * len(batched_prompt_bboxes)
        confs = [None] * len(batched_prompt_bboxes)

        for index_batch in range(len(batched_prompt_bboxes)):
            result = self.predictor(bboxes=batched_prompt_bboxes[index_batch],
                                    points=None,
                                    labels=None,
                                    multimask_output=False)[0]
            masks[index_batch] = result.masks.data  # chunk_size x H x W
            confs[index_batch] = result.boxes.conf  # chunk_size

        masks = torch.cat(masks, dim=0)  # B x H x W
        confs = torch.cat(confs, dim=0)  # B

        # filter mask with low confidence score
        if self.mask_threshold is None:
            indices = torch.arange(0, prompt_bboxes.shape[0], device=masks.device)  # B
        else:
            indices = torch.where(confs > self.mask_threshold)[0]  # B'
            masks = masks[indices]  # B' x H x W
            confs = confs[indices]  # B'

        return masks, confs, indices  # (B|B') x H x W, (B|B'), (B|B')


class GroundedSAM:
    """High-level wrapper combining GroundingDINO and a SAM-family predictor.

    Use ``generate_masks`` to obtain boxes, masks, and confidence scores from a
    single RGB image given a prompt.
    """

    def __init__(self,
                 ground_dino: GroundingDINO,
                 sam_predictor: SAMPredictor,
                 segmentor_width_size: Optional[int] = None,
                 prompt_text: str = "objects") -> None:

        self.ground_dino = ground_dino
        self.sam_predictor = sam_predictor
        self.segmentor_width_size = segmentor_width_size
        self.prompt_text = prompt_text

    @staticmethod
    def load_grounded_sam_model(checkpoints_dir: str,
                                grounded_dino_config_dir: str,
                                grounded_dino_use_vitb: bool = False,
                                box_threshold: float = 0.1,
                                text_threshold: float = 0.1,
                                use_yolo_sam: bool = False,
                                sam_vit_model: str = "vit_t",
                                mask_threshold: float = 0.01,
                                prompt_text: str = "objects",
                                segmentor_width_size: Optional[int] = None,
                                device: Optional[str] = "cpu") -> "GroundedSAM":
        """Factory to build a ready-to-use GroundedSAM instance.

        Args:
            checkpoints_dir: Root directory holding all checkpoints.
            grounded_dino_config_dir: Directory with GroundingDINO config files.
            grounded_dino_use_vitb: Use the Swin-B variant of GroundingDINO.
            box_threshold: GroundingDINO box confidence threshold.
            text_threshold: GroundingDINO text threshold.
            use_yolo_sam: If True, use YOLO-SAM; else SAM/MobileSAM.
            sam_vit_model: For SAM-family. One of {"vit_t", "vit_b", "vit_l",
                "vit_h"} for SAM/MobileSAM, or YOLO-SAM .pt filename.
            mask_threshold: Minimum mask confidence to keep (None keeps all).
            prompt_text: Default text prompt.
            segmentor_width_size: Optional width to resize before segmentation
                for speed (keeps aspect ratio). Set None to disable.
            device: Torch device string.

        Returns:
            GroundedSAM: Configured model wrapper.
        """
        ground_dino = GroundingDINO(checkpoint_dir=checkpoints_dir,
                                    config_dir=grounded_dino_config_dir,
                                    use_vitb=grounded_dino_use_vitb,
                                    box_threshold=box_threshold,
                                    text_threshold=text_threshold,
                                    device=device)
        if use_yolo_sam:
            sam_predictor = YOLOSAMPredictor(checkpoint_dir=checkpoints_dir,
                                             vit_model=sam_vit_model,
                                             mask_threshold=mask_threshold,
                                             device=device)
        else:
            sam_predictor = SAMPredictor(checkpoint_dir=checkpoints_dir,
                                         vit_model=sam_vit_model,
                                         mask_threshold=mask_threshold,
                                         device=device)
        grounded_sam = GroundedSAM(ground_dino=ground_dino,
                                   sam_predictor=sam_predictor,
                                   segmentor_width_size=segmentor_width_size,
                                   prompt_text=prompt_text)
        return grounded_sam

    def move_to_device(self, device):
        self.ground_dino.move_to_device(device)
        self.sam_predictor.move_to_device(device)
        return self

    def preprocess_resize(self, image: np.ndarray):
        """Resize image to ``segmentor_width_size`` while keeping aspect ratio."""
        orig_size = image.shape[:2]
        height_size = int(self.segmentor_width_size * orig_size[0] / orig_size[1])
        resized_image = cv2.resize(image.copy(),
                                   (self.segmentor_width_size, height_size))  # (width, height)
        return resized_image

    def postprocess_resize(self, detections, orig_size):
        """Upscale masks/boxes back to the original image size after inference."""
        detections["masks"] = F.interpolate(detections["masks"].unsqueeze(1).float(),
                                            size=(orig_size[0], orig_size[1]),
                                            mode="bilinear",
                                            align_corners=False)[:, 0, :, :]

        scale = orig_size[1] / self.segmentor_width_size
        detections["boxes"] = detections["boxes"].float() * scale
        detections["boxes"][:, [0, 2]] = torch.clamp(detections["boxes"][:, [0, 2]],
                                                     0, orig_size[1] - 1)
        detections["boxes"][:, [1, 3]] = torch.clamp(detections["boxes"][:, [1, 3]],
                                                     0, orig_size[0] - 1)
        return detections

    @torch.no_grad()
    def generate_masks(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Generate detection boxes and segmentation masks for an image.

        Args:
            image: RGB image [H, W, 3].

        Returns:
            Optional[Dict[str, Any]]: A dictionary with keys:
              - "boxes": [B', 4] xyxy boxes (torch.Tensor)
              - "boxes_scores": [B'] box confidence scores (torch.Tensor)
              - "masks": [B', H, W] masks (torch.Tensor)
              - "masks_scores": [B'] mask confidence scores (torch.Tensor)
              - "indexes": [B'] indices into the original input boxes kept after
            or None if no detections were found.
        """
        if self.segmentor_width_size:
            orig_size = image.shape[:2]  # 2
            image = self.preprocess_resize(image)  # H' x W' x 3

        # generate bounding boxes from prompt text
        bboxes_xyxy, bboxes_confs, _ = self.ground_dino.predict(image, self.prompt_text)  # B x 4 (as xyxy), B

        #print("Found boxes: {}".format(bboxes_xyxy))

        if bboxes_xyxy.shape[0] == 0:
            # nothing found
            return None

        # generate masks
        masks, masks_confs, indices = self.sam_predictor.predict(image=image,
                                                                 prompt_bboxes=bboxes_xyxy)  # B' x H' x W', B', B'
        
        #print("indices {}".format(indices))

        boxes = bboxes_xyxy[indices]  # B' x 4
        bboxes_confs = bboxes_confs[indices]  # B'

        detections = {"boxes": boxes, "boxes_scores": bboxes_confs, "masks": masks, "masks_scores": masks_confs, "indexes": indices}

        if self.segmentor_width_size:
            detections = self.postprocess_resize(detections, orig_size)  # {..., "masks": B' x H x W, ...}

        return detections  # {"boxes": B' x 4 (as xyxy), "masks": B' x H x W, "boxes_scores": B', "masks_scores": B'}
