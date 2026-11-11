# _*_ coding:utf-8 _*_
import cv2
import torch
import numpy as np
from .data.config import cfg_mnet, cfg_re50
from .layers.functions.prior_box import PriorBox
from .models.retinaface import RetinaFace
from .utils.box_utils import decode, decode_landm
from skimage import transform as trans
from PIL import Image
from typing import Optional
from torchvision.ops import nms as tv_nms
import math


#-->left profile
src1 = np.array([[55.550, 49.477], [62.116, 49.339],  [38.075, 70.237], [55.017, 92.262], [61.466, 92.979]], dtype=np.float32)
#<--left                
src2 = np.array([[47.241, 48.598], [72.887, 49.540],  [40.555, 71.067], [47.423, 93.643], [71.236, 94.353]], dtype=np.float32)
#---frontal                                                                                               
src3 = np.array([[33.566, 40.559], [78.321, 40.559],  [55.338, 64.967], [38.479, 90.276], [73.591, 90.276]], dtype=np.float32)
#-->right
src4 = np.array([[39.516, 49.540], [65.162, 48.598],  [71.849, 71.067], [41.167, 94.353], [64.980, 93.643]], dtype=np.float32)
#-->right profile
src5 = np.array([[50.225, 49.339], [56.791, 49.477],  [74.266, 70.237], [50.875, 92.979], [65.317, 92.262]], dtype=np.float32)

src = np.array([src1, src2, src3, src4, src5])
src_map = {112: src, 224: src * 2} 

# lmk is prediction; src is template
def estimate_norm(lmk, image_size=112):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')

    src = src_map[image_size]
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
        #         print(error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index

def norm_crop(img, landmark, image_size=112):
  M, pose_index = estimate_norm(landmark, image_size)
  warped = cv2.warpAffine(img,M, (image_size, image_size), borderValue = 0.0)
  return warped

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, device):
    print('Loading pretrained model from {}'.format(pretrained_path))
    
    pretrained_dict = torch.load(pretrained_path, map_location=device)
    
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model
  
 
def load_retinaface_model(
    device: torch.device, 
    model_path: str, 
) -> tuple[torch.nn.Module, dict]:
  
    lower = model_path.lower()
    cfg = cfg_mnet if ("mobilenet0.25" in lower or "mnet" in lower) else cfg_re50
    net = RetinaFace(cfg=cfg, phase="test")
    net = load_model(net, model_path, device)
    net.to(device)
    net.eval()
    return net, cfg

try:
    _inf = torch.inference_mode()
except AttributeError:
    _inf = torch.no_grad()

@_inf
def align_face(
    frame: np.ndarray,
    net: torch.nn.Module,
    cfg: dict,
    device: torch.device,
    confidence_threshold: float = 0.7,
    top_k: int = 1,
    nms_threshold: float = 0.35,
    vis_thres: float = 0.6
) -> Optional[Image.Image]:
  
    """
    Detect and align the most prominent face from a BGR OpenCV frame.
    Returns a 112x112 PIL Image if a face is found, otherwise None.
    """
    original_rgb = frame.copy()
    
    
    # Downscale-only: cap long side to 320, never upscale (saves time, avoids blur)
    long_cap = 320
    h, w = frame.shape[:2]
    long_side = max(h, w)
    resize = 1.0
    if long_side > long_cap:
        resize = long_cap / float(long_side)
        frame = cv2.resize(frame, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    
    # Prepare input for RetinaFace

    h, w = frame.shape[:2]
    

    frame = frame.astype(np.float32)
    frame  -= (104, 117, 123)
    frame  = frame.transpose(2, 0, 1)
    frame  = torch.from_numpy(frame).unsqueeze(0).to(device)
    
    loc, conf, landms = net(frame)

    priorbox = PriorBox(cfg, image_size=(h, w))
    prior_data = priorbox.forward().to(device).data
    
    scale = torch.tensor([w, h, w, h], dtype=loc.dtype, device=device)
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    
    scale1_size = torch.tensor([w, h] * 5, dtype=landms.dtype, device=device)
    
    landms = landms * scale1_size / resize
    landms = landms.cpu().numpy()
    
    # ---- filter by confidence ----
    inds = np.where(scores > confidence_threshold)[0]
    if inds.size == 0:
        return None
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]
    
    # ---- sort and take top-k ----
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]
    
    # ---- NMS (torchvision) ----
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    boxes  = torch.from_numpy(dets[:, :4])
    scores = torch.from_numpy(dets[:, 4])
    keep_idx = tv_nms(boxes, scores, nms_threshold).cpu().numpy()
    dets   = dets[keep_idx, :]
    landms = landms[keep_idx]
    
    if dets.shape[0] == 0 or float(dets[0, 4]) < vis_thres:
        return None

    lmk = landms[0].reshape((5, 2)).astype(np.float32)
    
    aligned = norm_crop(original_rgb, lmk)
        
    return Image.fromarray(aligned)



"""
RECORDING PROCESSING, CROP TO ORIGINAL SIZE
"""


# lmk is prediction; src is template
def estimate_norm_unscaled(lmk, image_size=112):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')

    src = src_map[image_size]
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T

        error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
        #         print(error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i

    # Decompose min_M into rotation R and scale s
    A = min_M[:, :2]  # 2x2
    # For a similarity transform, both columns should have the same norm.
    # We average them for numerical robustness.
    col_norms = np.linalg.norm(A, axis=0)
    s = float(col_norms.mean())
    R = A / s

    original_pixels = image_size / s

    return R, min_index, original_pixels

def norm_crop_unscaled(landmark, image_size=112):
  R, pose_index, face_pixels = estimate_norm_unscaled(landmark, image_size)
  side = int(math.ceil(face_pixels))

  # center the landmark centroid in the new canvas
  lmk_center = landmark.mean(axis=0)
  canvas_center = np.array([side / 2.0, side / 2.0], dtype=np.float32)

  t = canvas_center - (R @ lmk_center)

  M = np.zeros((2, 3), dtype=np.float32)
  M[:, :2] = R
  M[:, 2]  = t

#  warped = cv2.warpAffine(img, M, (side, side), borderValue = 0.0)
#  return warped, M, side
  return M, side


@_inf
def align_face_unscaled(
    frame: np.ndarray,
    net: torch.nn.Module,
    cfg: dict,
    device: torch.device,
    confidence_threshold: float = 0.70,
    top_k: int = 1,
    nms_threshold: float = 0.35,
    vis_thres: float = 0.6
) -> Optional[Image.Image]:
  
    """
    Detect and align the most prominent face from an OpenCV frame. Returns only alignment matrix and crop size.
    """
#    original_rgb = frame.copy()
    
    # Downsample to 8-bit
    frame = cv2.convertScaleAbs(frame, alpha=255.0 / 4095.0)

    # Downscale-only: cap long side to 320, never upscale (saves time, avoids blur)
    long_cap = 320
    H0, W0 = frame.shape[:2]
    long_side = max(H0, W0)
    resize = 1.0
    if long_side > long_cap:
        resize = long_cap / float(long_side)
        frame = cv2.resize(frame, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    
    # Prepare input for RetinaFace

    h, w = frame.shape[:2]
    
    
    frame = frame.astype(np.float32)
    frame  -= (104, 117, 123)
    frame  = frame.transpose(2, 0, 1)
    frame  = torch.from_numpy(frame).unsqueeze(0).to(device)
    
    loc, conf, landms = net(frame)

    priorbox = PriorBox(cfg, image_size=(h, w))
    prior_data = priorbox.forward().to(device).data
    
    scale = torch.tensor([w, h, w, h], dtype=loc.dtype, device=device)
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    
    scale1_size = torch.tensor([w, h] * 5, dtype=landms.dtype, device=device)
    
    landms = landms * scale1_size / resize
    landms = landms.cpu().numpy()
    
    # ---- filter by confidence ----
    inds = np.where(scores > confidence_threshold)[0]
    if inds.size == 0:
        return None
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]
    
    # ---- sort and take top-k ----
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]
    
    # ---- NMS (torchvision) ----
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    boxes  = torch.from_numpy(dets[:, :4])
    scores = torch.from_numpy(dets[:, 4])
    keep_idx = tv_nms(boxes, scores, nms_threshold).cpu().numpy()
    dets   = dets[keep_idx, :]
    landms = landms[keep_idx]
    
    if dets.shape[0] == 0 or float(dets[0, 4]) < vis_thres:
        return None
    
    lmk = landms[0].reshape((5, 2)).astype(np.float32)


    # ---- Recover face size ----
    x1, y1, x2, y2 = dets[0, :4]
    face_h = y2 - y1
    ratio = 1/16


    # ---- Gate 2 (strict): landmarks must be safely inside frame ----
    # lmk: (5,2) in original coords; H0, W0 known

    eye = lmk[[0, 1], :]       # left/right eye centers
    mou = lmk[[3, 4], :]       # left/right mouth corners

    # strict inward margins (pixels)
    left_margin_x   = 0
    right_margin_x  = 0
    top_margin_y    = face_h * ratio
    bottom_margin_y = face_h * ratio

    xmin = left_margin_x
    xmax = (W0 - 1) - right_margin_x
    ymin = top_margin_y
    ymax = (H0 - 1) - bottom_margin_y

    # X: enforce for both eyes and mouth corners (optional but usually sensible)
    if (
        (eye[:, 0] < xmin).any() or (eye[:, 0] > xmax).any() or
        (mou[:, 0] < xmin).any() or (mou[:, 0] > xmax).any()
    ):
        return None

    # Y: eyes must be below top buffer (eyebrows room above)
    if (eye[:, 1] < ymin).any():
        return None

    # Y: mouth corners must be above bottom buffer (lower lip room below)
    if (mou[:, 1] > ymax).any():
        return None

    M, side = norm_crop_unscaled(lmk)
        
    return M, side

