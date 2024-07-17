import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import torch
from torch import nn
import torch.nn.functional as F

from torchvision.models.detection import MaskRCNN

from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.roi_heads import paste_masks_in_image
from .SamEncoderViT import ImageEncoderViT
from functools import partial

from typing import List, Tuple, Dict
from collections import OrderedDict
"""
# load a model pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

# replace the classifier with a new one, that has
# num_classes which is user-defined
num_classes = 2  # 1 class (person) + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)



backbone = get_backbone()

backbone.out_channels = 1280

# let's make the RPN generate 5 x 3 anchors per spatial
# location, with 5 different sizes and 3 different aspect
# ratios. We have a Tuple[Tuple[int]] because each feature
# map could potentially have different sizes and
# aspect ratios
anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),)
)

# let's define what are the feature maps that we will
# use to perform the region of interest cropping, as well as
# the size of the crop after rescaling.
# if your backbone returns a Tensor, featmap_names is expected to
# be [0]. More generally, the backbone should return an
# ``OrderedDict[Tensor]``, and in ``featmap_names`` you can choose which
# feature maps to use.
roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    featmap_names=['0'],
    output_size=7,
    sampling_ratio=2
)

# put the pieces together inside a Faster-RCNN model
model = FasterRCNN(
    backbone,
    num_classes=2,
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler
)



def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.backbone = SamEncoderViT()
    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model

"""
class Detev(MaskRCNN):
      
    def __init__(self,
                   pixel_mean,
                   pixel_std,
                   backbone=None,
                   backbone_pretrained=None,
                   box_detector=None,
                   mask_detector=None,
                   img_size =1024,
                   num_classes=2):
            

            backbone = ImageEncoderViT(depth=12,
                                    embed_dim=768,
                                    img_size=1024,
                                    mlp_ratio=4,
                                    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                                    num_heads=12,
                                    patch_size=16,
                                    qkv_bias=True,
                                    use_rel_pos=True,
                                    global_attn_indexes=[2, 5, 8, 11],
                                    window_size=14,
                                    out_chans=256,)
            backbone.out_channels = 256
            backbone.init_weights(backbone_pretrained)
            # self.img_size = img_size
            super().__init__(backbone=backbone, num_classes=num_classes)
            # self.backbone = backbone
            self.transform = None
            anchor_sizes = ((64,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            self.rpn.anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
            in_features = self.roi_heads.box_predictor.cls_score.in_features
            # replace the pre-trained head with a new one
            self.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

            #model.backbone = SamEncoderViT()
            # now get the number of input features for the mask classifier
            in_features_mask = self.roi_heads.mask_predictor.conv5_mask.in_channels
            hidden_layer = 256
            # and replace the mask predictor with a new one
            self.roi_heads.mask_predictor = MaskRCNNPredictor(
                in_features_mask,
                hidden_layer,
                num_classes
            )
            self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
            # self.box_head = box_detector
            # self.mask_head = mask_detector
           
      
    def forward(self, images, targets=None):
    # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")
        if targets is not None:
            original_image_sizes =[(t['orig_size'][0].item(), t['orig_size'][1].item()) for t in targets]##: List[Tuple[int, int]] = []

        # for img in images:
        #     val = img.shape[-2:]
        #     torch._assert(
        #         len(val) == 2,
        #         f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
        #     )
        #     original_image_sizes.append((val[0], val[1]))

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        features = self.backbone(images.tensors)  #.tensors
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(images, features, targets)

        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        
        #detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]
        detections = self.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return losses, detections#self.eager_outputs(losses, detections)

            

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
            """Normalize pixel values and pad to a square input."""
            # Normalize colors
            x = (x - self.pixel_mean) / self.pixel_std

            # Pad
            h, w = x.shape[-2:]
            padh = self.img_size - h
            padw = self.img_size - w
            x = F.pad(x, (0, padw, 0, padh))
            return x
    
    def postprocess(
        self,
        result: List[Dict[str, torch.Tensor]],
        image_shapes: List[Tuple[int, int]],
        original_image_sizes: List[Tuple[int, int]],
    ) -> List[Dict[str, torch.Tensor]]:
        if self.training:
            return result
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
            if "masks" in pred:
                masks = pred["masks"]
                masks = paste_masks_in_image(masks, boxes, o_im_s)
                result[i]["masks"] = masks
        return result

def build_model(num_classes = 17,args=None):
    model = Detev(pixel_mean= [123.675, 116.28, 103.53], pixel_std =[58.395, 57.12, 57.375], num_classes=num_classes, backbone_pretrained=args.pretrained_backbone_path)

    return model


def resize_boxes(boxes: torch.Tensor, original_size: List[int], new_size: List[int]) -> torch.Tensor:
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device)
        / torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


