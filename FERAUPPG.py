import os
import sys
import platform
import json
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Dict, List
import zstandard as zstd
import struct
import numpy as np
import cv2
import time
from threading import Event, Thread, Condition
from collections import deque
import io, queue
import asyncio

from MEGraphAU.model.MEFL import MEFARG as MEGraphModel
from MEGraphAU.utils import load_state_dict
from MEGraphAU.OpenGraphAU.model.MEFL import MEFARG as OpenGraphModel
from ddamfn.affectnet7_test import load_ddamfn_model
from OpenFace3.model.MLT import MLT

"""
DEVICE
"""
if torch.cuda.is_available():
  device = torch.device("cuda")
elif torch.backends.mps.is_available() and platform.system() == "Darwin":
  device = torch.device("mps")
else:
  device = torch.device("cpu")

print(f"Using device: {device}")

"""
IMAGE TO TENSOR
"""

transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
])

"""
ME-GraphAU
"""

def load_megraphau(backbone, train, fold, device="cuda"):
  checkpoint_path = MEGraphAU_weights[backbone][train][fold] # SWIN TINY HAS NO DISFA, WILL THROW IF ATTEMPTED
  num_class = 12 if train == 'BP4D' else 8
  model = MEGraphModel(num_classes=num_class, backbone=backbone)
  model = load_state_dict(model, checkpoint_path)
  model.to(device).eval()
  return model

def predict_megraphau(model, tensor): # Requires size 224

  with torch.inference_mode():
    au_probs, _ = model(tensor) # Omitted output is edge logits.

  au_probs = au_probs.detach().cpu().numpy()

  return au_probs

MEGraphAU_backbones = {
  1: "swin_transformer_base",
  2: "swin_transformer_tiny",
  3: "resnet50"
}

MEGraphAU_weights = {
  "swin_transformer_base": {
    "DISFA": {
      "fold1": "MEGraphAU/checkpoints/MEFARG_swin_base_DISFA_fold1.pth",
      "fold2": "MEGraphAU/checkpoints/MEFARG_swin_base_DISFA_fold2.pth",
      "fold3": "MEGraphAU/checkpoints/MEFARG_swin_base_DISFA_fold3.pth"
    },
    "BP4D": {
      "fold1": "MEGraphAU/checkpoints/MEFARG_swin_base_BP4D_fold1.pth",
      "fold2": "MEGraphAU/checkpoints/MEFARG_swin_base_BP4D_fold2.pth",
      "fold3": "MEGraphAU/checkpoints/MEFARG_swin_base_BP4D_fold3.pth"
    }
  },
  "swin_transformer_tiny": {
    "BP4D": {
      "fold1": "MEGraphAU/checkpoints/MEFARG_swin_tiny_BP4D_fold1.pth",
      "fold2": "MEGraphAU/checkpoints/MEFARG_swin_tiny_BP4D_fold2.pth",
      "fold3": "MEGraphAU/checkpoints/MEFARG_swin_tiny_BP4D_fold3.pth"
    }
  },
  "resnet50": {
    "DISFA": {
      "fold1": "MEGraphAU/checkpoints/MEFARG_resnet50_DISFA_fold1.pth",
      "fold2": "MEGraphAU/checkpoints/MEFARG_resnet50_DISFA_fold2.pth",
      "fold3": "MEGraphAU/checkpoints/MEFARG_resnet50_DISFA_fold3.pth"
    },
    "BP4D": {
      "fold1": "MEGraphAU/checkpoints/MEFARG_resnet50_BP4D_fold1.pth",
      "fold2": "MEGraphAU/checkpoints/MEFARG_resnet50_BP4D_fold2.pth",
      "fold3": "MEGraphAU/checkpoints/MEFARG_resnet50_BP4D_fold3.pth"
    }
  }
}

AU_NAMES = {
  "BP4D": ["AU1", "AU2", "AU4", "AU6", "AU7", "AU10", "AU12", "AU14", "AU15", "AU17", "AU23", "AU24"],
  "DISFA": ["AU1", "AU2", "AU4", "AU6", "AU9", "AU12", "AU25", "AU26"],
  "HYBRID": ['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU9', 'AU10', 'AU11', 'AU12', 'AU13', 'AU14', 'AU15', 'AU16', 'AU17', 'AU18', 'AU19', 'AU20', 'AU22',
    'AU23', 'AU24', 'AU25', 'AU26', 'AU27', 'AU32', 'AU38', 'AU39', 'AUL1', 'AUR1', 'AUL2', 'AUR2', 'AUL4', 'AUR4', 'AUL6', 'AUR6', 'AUL10', 'AUR10', 'AUL12', 'AUR12', 'AUL14', 'AUR14']
}


megraphau_bp4d_fld1_model = load_megraphau(MEGraphAU_backbones[1], "BP4D", "fold1", device=device)
print("MEGraph BP4D model loaded")

megraphau_disfa_fld1_model = load_megraphau(MEGraphAU_backbones[1], "DISFA", "fold1", device=device)
print("MEGraph DISFA model loaded")


"""
OpenGraphAU
"""
OpenGraphAU_weights = "MEGraphAU/checkpoints/OpenGraphAU-maevit_second_stage.pth"

def load_opengraphau(weights, device="cuda"):
  model = OpenGraphModel(backbone="mae")
  model = load_state_dict(model, weights)
  model.to(device).eval()
  return model

def predict_opengraphau(model, tensor): # Requires size 224

  with torch.inference_mode():
    au_logits = model(tensor)

  au_probs = au_logits.detach().cpu().numpy()

  return au_probs

"""
opengraphau_model = load_opengraphau(OpenGraphAU_weights, device=device)
print("OpenGraph model loaded")
"""


"""
DDAMFN
"""

ddamfn_weights = {
  "DDAMFN_Aff7": 'ddamfn/checkpoints_ver2.0/affecnet7_epoch19_acc0.671.pth',
  "DDAMFN_Aff8": 'ddamfn/checkpoints_ver2.0/affecnet8_epoch25_acc0.6469.pth',
  "DDAMFN_RAF": 'ddamfn/checkpoints_ver2.0/rafdb_epoch20_acc0.9204_bacc0.8617.pth',
}

def predict_ddamfn(
  model: torch.nn.Module,
  tensor
):
  
  with torch.inference_mode():
    logits, _, _ = model(tensor)
  probs = torch.softmax(logits, dim=1).detach().cpu().numpy()

  return probs

ddamfn_aff7_model = load_ddamfn_model(device=device, model_path=ddamfn_weights["DDAMFN_Aff7"], num_class=7)
print("DDAMFN_Aff7 model loaded")

ddamfn_aff8_model = load_ddamfn_model(device=device, model_path=ddamfn_weights["DDAMFN_Aff8"], num_class=8)
print("DDAMFN_Aff8 model loaded")

ddamfn_raf_model = load_ddamfn_model(device=device, model_path=ddamfn_weights["DDAMFN_RAF"], num_class=7)
print("DDAMFN_RAF model loaded")


"""
OPENFACE
"""

EMOTIONS = {
  "7": ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"],
  "8": ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
}

GAZE = ["yaw", "pitch"]

openface_weights = 'OpenFace3/weights/MLT_backbone.pth'

def load_openface(weights, device="cuda"):
  model = MLT()  # base_model_name, expr_classes, au_numbers default must match training
  state = torch.load(weights, map_location=device)
  model.load_state_dict(state)
  model.to(device)
  model.eval()
  return model

def predict_openface(model, tensor):

  with torch.inference_mode():
    emotion_logits, gaze_output, au_probs = model(tensor)

  emotion_probs = torch.softmax(emotion_logits, dim=-1).detach().cpu().numpy()
  au_probs = au_probs.detach().cpu().numpy()
  gaze = gaze_output.detach().cpu().numpy()
  
  return (
    emotion_probs,
    au_probs,
    gaze
  )

openface_model = load_openface(openface_weights, device=device)
print("OpenFace model loaded")


"""
rPPG
"""


"""
PROCESS VIDEO
"""


RCLONE_ROOT = Path.home() / "mnt" / "gdrive"
RECORDINGS_ROOT = RCLONE_ROOT / "TesisData" / "Recordings"
#RECORDINGS_ROOT = Path.home() / "repair_recordings"
PROCESSED_ROOT = RCLONE_ROOT / "TesisData" / "Processed"
#PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)

"""
#TESTING
RCLONE_ROOT = Path(__file__).resolve().parent
RECORDINGS_ROOT = RCLONE_ROOT / "TesisData" / "Recordings"
PROCESSED_ROOT = RCLONE_ROOT / "TesisData" / "Processed"
PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
"""


MODEL_NAMES = [
  "DDAMFN_Aff7",
  "DDAMFN_Aff8",
  "DDAMFN_RAF",
  "MEGraphAU_BP4D_FLD1",
  "MEGraphAU_DISFA_FLD1",
  "OpenGraphAU",
  "OpenFace",
]

def scan_recordings():

  if not RECORDINGS_ROOT.exists():
    print(f"Recordings folder not found at: {RECORDINGS_ROOT}")
    return []

  video_paths = sorted(RECORDINGS_ROOT.rglob("*.mraw.zst"))

  print(f"Found {len(video_paths)} raw video file(s):")

  return video_paths


def model_switches(meta_path):
  """
  For a processed video, load a .json meta file describing which models
  have already been run. If the file does not exist, all switches = False.
  """

  # Default: all disabled
  switches = {name: False for name in MODEL_NAMES}

  if not meta_path.exists():
    # No file? Nothing has been run.
    return switches

  # File exists -> read JSON
  try:
    with open(meta_path, "r", encoding="utf-8") as jf:
      meta = json.load(jf)

    # Expect list of model names under "models"
    models_run = [k for k, v in meta.items() if v]

    # Enable switches for models in meta file
    for name in MODEL_NAMES:
      if name in models_run:
        switches[name] = True

  except Exception as e:
    print(f"⚠️ Could not read meta file {meta_path}: {e}")
    # Fallback: all False
    return switches

  return switches

def process_video(video_path: Path, batch_size=128):

  meta_path = video_path.with_name(Path(video_path.stem).stem + ".json")
  hasMeta = meta_path.exists()
  if hasMeta:
    with open(meta_path, "r", encoding="utf-8") as jf:
      meta = json.load(jf)
  else:
    meta={}
    print(f"⚠️  Missing metadata for {video_path}.")

  if not hasMeta:
    metaRawPath = video_path.with_name(Path(video_path.stem).stem + "_raw.json")
    hasMetaRaw = metaRawPath.exists()
    if hasMetaRaw:
      with open(metaRawPath, "r", encoding="utf-8") as jf:
        metaRaw = json.load(jf)
    else:
      metaRaw={}
      print(f"⚠️  Missing raw metadata for {video_path}.")

  base = video_path.stem      # removes .zstd
  base = Path(base).stem      # removes .mraw

  results_meta = PROCESSED_ROOT / f"{base}.json"
  processed_path = PROCESSED_ROOT / f"{base}.npz"

  # Load model switches from processed JSON (if exists)
  switches = model_switches(results_meta)

  # Global early-stop: all requested models exist
  if all(switches.values()):
    print(f"{video_path.name}: all models already processed, skipping.")
    return False

  timestamps = []
  routine_timestamps = []
  totalFrames = None

  header_format = meta.get("header_format", "<HHHHHfffffff")
  header_bytes = struct.calcsize(header_format)

  if hasMeta:
    frames = meta.get("frames", None)
    project = meta.get("project", None)
    if frames is not None:
      timestamps = [f["timestamp"] for f in frames]
      totalFrames = len(frames)
      if project == "Edu":
        routine_timestamps = [f.get("timestamp_routine") for f in frames]
    sessionID = meta.get("session", None)
    playerID = meta.get("player", None)
    roundID = meta.get("round", None)
    recording_index = meta.get("recording_index", None)
  elif hasMetaRaw:
    project = metaRaw.get("project", None)
    sessionID = metaRaw.get("session", None)
    playerID = metaRaw.get("player", None)
    roundID = metaRaw.get("round", None)
    recording_index = metaRaw.get("recording_index", None)
  else:  
    print(f"{video_path.name}: No metadata, skipping.")
    return False

  dctx = zstd.ZstdDecompressor()

  toDo = [k for k, v in switches.items() if not v]

  producerQueue = queue.Queue(maxsize=10)
  errorFlag=Event()


  producerThread = Thread(
    target=producer,
    args=(producerQueue, video_path, base, dctx, header_bytes, header_format, timestamps, hasMeta, totalFrames, batch_size, errorFlag),
    daemon=False
  )
  producerThread.start()

  inferenceDict = {
    "ddamfn_aff7": [],       # list of 7 probs
    "ddamfn_aff8": [],       # list of 8 probs
    "ddamfn_raf": [],      # list of 7 probs 
    "megraph_bp4d_fld1": [],   # list of 12 AU probs
    "megraph_disfa_fld1": [],  # list of 8 AU probs
    "opengraphau": [],       # list of 41 AU probs
    "openface_emo": [],      # list of 8 probs
    "openface_au": [],       # list of 8 probs (DISFA)
    "openface_gaze": []     # list of [yaw, pitch]
  }

  consumerThread = Thread(
    target=consumer,
    args=(switches, producerQueue, inferenceDict),
    daemon=False
  )
  consumerThread.start()

  print(f"Running {toDo} inference on {base}.")

  # Wait for producer to finish reading
  producerThread.join()

  # Wait for consumer to finish inference
  consumerThread.join()

  if errorFlag.is_set():
    return False # Don't save inference from corrupted files

  # ---------------------------------------------------------
  # If processed npz exists, load its arrays for merge/append
  # ---------------------------------------------------------
  existing = None
  if processed_path.exists():
    try:
      existing = np.load(processed_path)
      print(f"🔄 Existing processed file found: merging outputs.")
    except Exception as e:
      print(f"⚠️ Could not read existing npz ({processed_path}): {e}")
      existing = None

  # ---------------------------------------
  # Prepare and save arrays to NPZ
  # ---------------------------------------

  arrays = {
    "project": project,
    "session": sessionID,
    "player": playerID,
    "timestamps": np.asarray(timestamps, dtype=np.float32),
  }

  if project == "Nico":
    arrays["round"] = roundID
  elif project == "Edu":
    arrays["recording_index"] = recording_index
    arrays["timestamp_routine"] = np.asarray(routine_timestamps, dtype=np.float32)

  else:
    print("Error: project meta does not match either 'Nico' nor 'Edu'.")

  # ---------- DDAMFN ----------
  if existing is not None and "DDAMFN_Aff7" in existing:
    arrays["DDAMFN_Aff7"] = existing["DDAMFN_Aff7"]
    switches["DDAMFN_Aff7"] = True
  elif inferenceDict["ddamfn_aff7"]:
    arrays["DDAMFN_Aff7"] = np.vstack(inferenceDict["ddamfn_aff7"])
    switches["DDAMFN_Aff7"] = True
    print(f"Ran DDAMFN_Aff7 for {base}")
  else:
    print(f"Error, DDAMFN_Aff7 not existing nor run for {base}")

  if existing is not None and "DDAMFN_Aff8" in existing:
    arrays["DDAMFN_Aff8"] = existing["DDAMFN_Aff8"]
    switches["DDAMFN_Aff8"] = True
  elif inferenceDict["ddamfn_aff8"]:
    arrays["DDAMFN_Aff8"] = np.vstack(inferenceDict["ddamfn_aff8"])
    switches["DDAMFN_Aff8"] = True
    print(f"Ran DDAMFN_Aff8 for {base}")
  else:
    print(f"Error, DDAMFN_Aff8 not existing nor run for {base}")

  if existing is not None and "DDAMFN_RAF" in existing:
    arrays["DDAMFN_RAF"] = existing["DDAMFN_RAF"]
    switches["DDAMFN_RAF"] = True
  elif inferenceDict["ddamfn_raf"]:
    arrays["DDAMFN_RAF"] = np.vstack(inferenceDict["ddamfn_raf"])
    switches["DDAMFN_RAF"] = True
    print(f"Ran DDAMFN_RAF for {base}")
  else:
    print(f"Error, DDAMFN not existing nor run for {base}")

  # ---------- MEGraphAU (BP4D) ----------
  if existing is not None and "MEGraphAU_BP4D_FLD1" in existing:
    arrays["MEGraphAU_BP4D_FLD1"] = existing["MEGraphAU_BP4D_FLD1"]
    switches["MEGraphAU_BP4D_FLD1"] = True
  elif inferenceDict["megraph_bp4d_fld1"]:
    arrays["MEGraphAU_BP4D_FLD1"] = np.vstack(inferenceDict["megraph_bp4d_fld1"])
    switches["MEGraphAU_BP4D_FLD1"] = True
    print(f"Ran MEGraphAU_BP4D_FLD1 for {base}")
  else:
    print(f"Error, MEGraphAU_BP4D_FLD1 not existing nor run for {base}")

  # ---------- MEGraphAU (DISFA) ----------
  if existing is not None and "MEGraphAU_DISFA_FLD1" in existing:
    arrays["MEGraphAU_DISFA_FLD1"] = existing["MEGraphAU_DISFA_FLD1"]
    switches["MEGraphAU_DISFA_FLD1"] = True
  elif inferenceDict["megraph_disfa_fld1"]:
    arrays["MEGraphAU_DISFA_FLD1"] = np.vstack(inferenceDict["megraph_disfa_fld1"])
    switches["MEGraphAU_DISFA_FLD1"] = True
    print(f"Ran MEGraphAU_DISFA_FLD1 for {base}")
  else:
    print(f"Error, MEGraphAU_DISFA_FLD1 not existing nor run for {base}")
  
  # ---------- OpenGraphAU ----------
  if existing is not None and "OpenGraphAU" in existing:
    arrays["OpenGraphAU"] = existing["OpenGraphAU"]
    switches["OpenGraphAU"] = True
  elif inferenceDict["opengraphau"]:
    arrays["OpenGraphAU"] = np.vstack(inferenceDict["opengraphau"])
    switches["OpenGraphAU"] = True
    print(f"Ran OpenGraphAU for {base}")
  else:
    print(f"Error, OpenGraphAU not existing nor run for {base}")

  # ---------- OpenFace ----------
  if existing is not None and "OpenFace_Emotions" in existing:
    arrays["OpenFace_Emotions"] = existing["OpenFace_Emotions"]
  elif inferenceDict["openface_emo"]:
    arrays["OpenFace_Emotions"] = np.vstack(inferenceDict["openface_emo"])
  else:
    print(f"Error, OpenFace_Emotions not existing nor run for {base}")

  if existing is not None and "OpenFace_AU" in existing:
    arrays["OpenFace_AU"] = existing["OpenFace_AU"]
  elif inferenceDict["openface_au"]:
    arrays["OpenFace_AU"] = np.vstack(inferenceDict["openface_au"])
  else:
    print(f"Error, OpenFace_AU not existing nor run for {base}")

  if existing is not None and "OpenFace_Gaze" in existing:
    arrays["OpenFace_Gaze"] = existing["OpenFace_Gaze"]
    switches["OpenFace"] = True
  elif inferenceDict["openface_gaze"]:
    arrays["OpenFace_Gaze"] = np.vstack(inferenceDict["openface_gaze"])
    switches["OpenFace"] = True
    print(f"Ran OpenFace for {base}")
  else:
    print(f"Error, OpenFace_Gaze not existing nor run for {base}")

  # SAVE RESULTS
  np.savez_compressed(processed_path, **arrays)
  print(f"Results saved to {processed_path}")

  # WRITE META FOR MODELS RUN
  with open(results_meta, "w", encoding="utf-8") as jf:
    json.dump(switches, jf, indent=2)
  print(f"Meta saved to {results_meta}")

  return True

def producer(producerQueue, video_path, base, dctx, header_bytes, header_format, timestamps, hasMeta, totalFrames, batch_size, errorFlag):

  with open(video_path, "rb") as f, dctx.stream_reader(f) as reader:

    buf_112 = []
    buf_224 = []

    frame_idx = 0

    while True:
      
      # --- Read header ---

      try:
        header_buf = reader.read(header_bytes)
      except zstd.ZstdError as e:
        print(f"⚠️ ZSTD corruption in {base} at frame {frame_idx}: {e}")
        errorFlag.set()
        producerQueue.put(None)   # sentinel
        break

      if not header_buf:
        if len(buf_112) > 0:
          X112 = torch.stack(buf_112, dim=0)  # shape [B,3,112,112]
          X224 = torch.stack(buf_224, dim=0)  # shape [B,3,224,224]
          producerQueue.put((X112, X224))

          buf_112.clear()
          buf_224.clear()
        # Normal EOF
        producerQueue.put(None)   # sentinel
        break

      if len(header_buf) < header_bytes:
        print(f"⚠️ Truncated header at frame {frame_idx}, stopping.")
        errorFlag.set()
        producerQueue.put(None)   # sentinel
        break

      (
        magic,
        cropWidth,
        cropHeight,
        bytes_per_sample,
        side,
        ts_f32,
        M00, M01, M02,
        M10, M11, M12,
      ) = struct.unpack(header_format, header_buf)

      if magic != 0xFACE:
        print(
          f"⚠️ Bad magic {hex(magic)} at frame {frame_idx}, expected 0xFACE."
        )
        errorFlag.set()
        producerQueue.put(None)   # sentinel
        break

      # --- Read Bayer pixels ---
      pixel_bytes = cropWidth * cropHeight * bytes_per_sample

      try:
        frame_buf = reader.read(pixel_bytes)
      except zstd.ZstdError as e:
        print(f"⚠️ ZSTD corruption in {base} at frame {frame_idx}: {e}")
        errorFlag.set()
        producerQueue.put(None)   # sentinel
        break

      if len(frame_buf) < pixel_bytes:
        print(
          f"⚠️ Truncated frame data at frame {frame_idx} "
          f"(got {len(frame_buf)} / {pixel_bytes}), stopping."
        )
        errorFlag.set()
        producerQueue.put(None)   # sentinel
        break

      # Interpret as uint16 Bayer crop
      bayer_crop = np.frombuffer(frame_buf, dtype=np.uint16).reshape(
        cropHeight, cropWidth
      )

      rgb = cv2.cvtColor(bayer_crop, cv2.COLOR_BayerBG2RGB)

      # Rebuild M_crop using your naming
      M = np.array(
        [
          [M00, M01, M02],
          [M10, M11, M12],
        ],
        dtype=np.float32,
      )

      # ---- 112x112 ----
      scale_112 = 112.0 / float(side)
      M_112 = M.copy()
      M_112 *= scale_112  # escala TODA la matriz, incl. traslación

      aligned_112 = cv2.warpAffine(
        rgb,
        M_112.astype(np.float32),
        (112, 112),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
      )

      aligned_112_8bit = cv2.convertScaleAbs(aligned_112, alpha=255.0 / 4095.0)
      tensor_112 = transform(Image.fromarray(aligned_112_8bit))
      buf_112.append(tensor_112)

      # ---- 224x224 ----
      scale_224 = 224.0 / float(side)
      M_224 = M.copy()
      M_224 *= scale_224  # igual que arriba

      aligned_224 = cv2.warpAffine(
        rgb,
        M_224.astype(np.float32),
        (224, 224),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
      )

      aligned_224_8bit = cv2.convertScaleAbs(aligned_224, alpha=255.0 / 4095.0)
      tensor_224 = transform(Image.fromarray(aligned_224_8bit))
      buf_224.append(tensor_224)

      if not hasMeta:
        timestamps.append(ts_f32)

      if len(buf_112) == batch_size:
        X112 = torch.stack(buf_112, dim=0)  # shape [B,3,112,112]
        X224 = torch.stack(buf_224, dim=0)  # shape [B,3,224,224]
        producerQueue.put((X112, X224))

        buf_112.clear()
        buf_224.clear()

      frame_idx += 1

  if errorFlag.is_set():
    print(f"Error: Exiting {base} on frame {frame_idx + 1} of {totalFrames}")


def consumer(switches, producerQueue, inferenceDict):

  while True:
    item = producerQueue.get()

    if item is None:
      break   # clean shutdown

    tensor_112, tensor_224 = item

    X112 = tensor_112.to(device, non_blocking=True)
    X224 = tensor_224.to(device, non_blocking=True)
    
    if not switches["DDAMFN_Aff7"]:
      ddamfn_aff7_emotions = predict_ddamfn(ddamfn_aff7_model, X112)
      inferenceDict["ddamfn_aff7"].append(np.array(ddamfn_aff7_emotions, dtype=np.float32))
    
    if not switches["DDAMFN_Aff8"]:
      ddamfn_aff8_emotions = predict_ddamfn(ddamfn_aff8_model, X112)
      inferenceDict["ddamfn_aff8"].append(np.array(ddamfn_aff8_emotions, dtype=np.float32))
    
    if not switches["DDAMFN_RAF"]:
      ddamfn_raf_emotions = predict_ddamfn(ddamfn_raf_model, X112)
      inferenceDict["ddamfn_raf"].append(np.array(ddamfn_raf_emotions, dtype=np.float32))
    
    if not switches["MEGraphAU_BP4D_FLD1"]:
      megraphau_bp4d_fld1_aus = predict_megraphau(megraphau_bp4d_fld1_model, X224)
      inferenceDict["megraph_bp4d_fld1"].append(np.array(megraphau_bp4d_fld1_aus, dtype=np.float32))
    
    if not switches["MEGraphAU_DISFA_FLD1"]:
      megraphau_disfa_fld1_aus = predict_megraphau(megraphau_disfa_fld1_model, X224)
      inferenceDict["megraph_disfa_fld1"].append(np.array(megraphau_disfa_fld1_aus, dtype=np.float32))
    """
    if not switches["OpenGraphAU"]:
      opengraphau_aus = predict_opengraphau(opengraphau_model, X224)
      inferenceDict["opengraphau"].append(np.array(opengraphau_aus, dtype=np.float32))
    """
    if not switches["OpenFace"]:
      openface_emotions, openface_aus, openface_gaze = predict_openface(openface_model, X224)
      inferenceDict["openface_emo"].append(np.array(openface_emotions, dtype=np.float32))
      inferenceDict["openface_au"].append(np.array(openface_aus, dtype=np.float32))
      inferenceDict["openface_gaze"].append(np.array(openface_gaze, dtype=np.float32))

# RUN SCRIPT

if __name__ == "__main__":

  # Find all raw videos under RECORDINGS_ROOT
  video_paths = scan_recordings()

  if not PROCESSED_ROOT.exists():
    print("Processed directory not found, exiting.")
    sys.exit(1)
  
  startTime = time.perf_counter()

  processed_count = 0

  for video in video_paths:
    videoStart = time.perf_counter()

    wasProcessed = process_video(video)

    if wasProcessed:
      videoEnd = time.perf_counter()
      processed_count += 1
      print(f"Finished processing {video.stem}")
      print(f"Completed in {videoEnd - videoStart} seconds")
  
  endTime = time.perf_counter()

  print(f"Processed {processed_count} videos.")
  print(f"Total run time: {(endTime - startTime)/60} minutes")