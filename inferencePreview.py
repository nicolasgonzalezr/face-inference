import os
import json
from pathlib import Path
import zstandard as zstd
import struct
import numpy as np
import cv2
import time
from concurrent.futures import ThreadPoolExecutor



"""
INFERENCE PREVIEW
"""

class InferencePreview:
  """
  Draws DDAMFN emotions and MEGraphAU AUs around a face frame.

  Layout:
    - Original frame (aligned_224_8bit) at top-left.
    - Emotion bars at the bottom (DDAMFN probabilities).
    - AU bars at the right (MEGraphAU BP4D probabilities).
  """

  def __init__(
    self,
    frame_width: int,
    frame_height: int,
    emotion_labels: list[str],
    emotion_indices: list[int],
    au_labels: list[str],
    au_indices: list[int],
    bottom_height: int = 80,
    right_width: int = 220,
  ) -> None:
    self.W = frame_width
    self.H = frame_height
    self.HB = bottom_height
    self.WA = right_width

    # Canvas size
    self.CANVAS_W = self.W + self.WA
    self.CANVAS_H = self.H + self.HB

    # Which entries from the model outputs to display
    self.emotion_labels = emotion_labels
    self.emotion_indices = np.array(emotion_indices, dtype=int)
    self.au_labels = au_labels
    self.au_indices = np.array(au_indices, dtype=int)

  def render(
    self,
    frame_bgr: np.ndarray,
    emo_probs: np.ndarray,
    au_probs: np.ndarray,
  ) -> np.ndarray:

    h, w = frame_bgr.shape[:2]
    if h != self.H or w != self.W:
      raise ValueError(f"Frame size mismatch: got {w}x{h}, expected {self.W}x{self.H}")

    # White canvas
    canvas = np.full((self.CANVAS_H, self.CANVAS_W, 3), 255, dtype=np.uint8)

    # Paste face in top-left
    canvas[0:self.H, 0:self.W] = frame_bgr

    # Clip subsets
    emo = np.clip(emo_probs[self.emotion_indices], 0.0, 1.0)
    au  = np.clip(au_probs[self.au_indices],     0.0, 1.0)

    # Bottom = AUs (vertical bars)
    self._draw_aus(canvas, au)

    # Right = Emotions (horizontal bars, centered labels)
    self._draw_emotions(canvas, emo)

    return canvas

  def _draw_emotions(self, canvas: np.ndarray, emo: np.ndarray) -> None:
    """
    Right panel: horizontal bars for emotions with centered labels.
    Bars: #009E60. Text: black.
    """
    E = len(self.emotion_labels)
    if E == 0:
      return

    panel_x0 = self.W
    panel_x1 = self.W + self.WA
    panel_y0 = 0
    panel_y1 = self.H + self.HB
    panel_w  = panel_x1 - panel_x0
    panel_h  = panel_y1 - panel_y0

    row_h = max(14, panel_h // E)
    bar_x0 = panel_x0 + 5
    bar_x1 = panel_x1 - 5
    bar_max_w = bar_x1 - bar_x0

    for i, (label, v) in enumerate(zip(self.emotion_labels, emo)):
      y0 = panel_y0 + i * row_h
      y1 = y0 + row_h

      v_clamped = float(np.clip(v, 0.0, 1.0))
      bar_len   = int(v_clamped * bar_max_w)
      bar_end_x = bar_x0 + bar_len

      # Bar in #009E60 (BGR: 96,158,0)
      cv2.rectangle(
        canvas,
        (bar_x0,   y0 + 2),
        (bar_end_x, y1 - 2),
        (96, 158, 0),
        -1,
      )

      # Centered label (black)
      text = f"{label} {v_clamped:.2f}"
      text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

      text_x = panel_x0 + (panel_w - text_size[0]) // 2
      text_y = y0 + (row_h + text_size[1]) // 2

      cv2.putText(
        canvas,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),  # black
        1,
        cv2.LINE_AA,
      )

  def _draw_aus(self, canvas: np.ndarray, au: np.ndarray) -> None:
    """
    Bottom panel: vertical bars for AUs.
    Background is white (from render()).
    Bars: #FFC000. Text: black, slightly smaller.
    """
    A = len(self.au_labels)
    if A == 0:
      return

    panel_y0 = self.H
    panel_y1 = self.H + self.HB
    panel_h  = panel_y1 - panel_y0

    bar_width = max(1, self.W // A)

    for i, (label, v) in enumerate(zip(self.au_labels, au)):

      x0 = i * bar_width
      x1 = min(self.W - 1, (i + 1) * bar_width - 2)

      v_clamped = float(np.clip(v, 0.0, 1.0))
      bar_h   = int(v_clamped * (panel_h * 0.70))

      y_bottom  = panel_y1 - 8
      y_top   = max(panel_y0 + 4, y_bottom - bar_h)

      # Bar in #FFC000 (BGR: 0,192,255)
      cv2.rectangle(
        canvas,
        (x0 + 3, y_top),
        (x1,   y_bottom),
        (180, 130, 70),
        thickness=-1,
      )

      # --- AU label (small, black), centered in its slot over the bar ---
      text = label
      font_scale = 0.25

      # Measure text with the SAME font_scale we will use to draw
      text_size, _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        1,
      )

      # Center in the slot [x0, x0 + bar_width]
      slot_center_x = x0 + bar_width // 2
      text_x = int(slot_center_x - text_size[0] / 2)

      # Vertical position stays at the top of the bottom panel
      text_y = panel_y0 + text_size[1] + 2

      cv2.putText(
        canvas,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),  # black
        1,
        cv2.LINE_AA,
      )

"""
PROCESS VIDEO
"""

RCLONE_ROOT = Path.home() / "mnt" / "gdrive"
RECORDINGS_ROOT = RCLONE_ROOT / "TesisData" / "Recordings"
PROCESSED_ROOT = RCLONE_ROOT / "TesisData" / "Processed"
#PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)

"""
#TESTING
RCLONE_ROOT = Path(__file__).resolve().parent
RECORDINGS_ROOT = RCLONE_ROOT / "TesisData" / "Recordings"
PROCESSED_ROOT = RCLONE_ROOT / "TesisData" / "Processed"
PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
"""

AU_NAMES = {
  "BP4D": ["AU1", "AU2", "AU4", "AU6", "AU7", "AU10", "AU12", "AU14", "AU15", "AU17", "AU23", "AU24"],
  "DISFA": ["AU1", "AU2", "AU4", "AU6", "AU9", "AU12", "AU25", "AU26"],
  "HYBRID": ['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU9', 'AU10', 'AU11', 'AU12', 'AU13', 'AU14', 'AU15', 'AU16', 'AU17', 'AU18', 'AU19', 'AU20', 'AU22',
    'AU23', 'AU24', 'AU25', 'AU26', 'AU27', 'AU32', 'AU38', 'AU39', 'AUL1', 'AUR1', 'AUL2', 'AUR2', 'AUL4', 'AUR4', 'AUL6', 'AUR6', 'AUL10', 'AUR10', 'AUL12', 'AUR12', 'AUL14', 'AUR14']
}

EMOTIONS = {
  "7": ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"],
  "8": ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
}

GAZE = ["yaw", "pitch"]

MODEL_NAMES = [
  "DDAMFN_Aff7",
  "DDAMFN_Aff8",
  "DDAMFN_RAF",
  "MEGraphAU_BP4D_FLD1",
  "MEGraphAU_BP4D_FLD2",
  "MEGraphAU_BP4D_FLD3",
  "MEGraphAU_DISFA_FLD1",
  "MEGraphAU_DISFA_FLD2",
  "MEGraphAU_DISFA_FLD3",
  "OpenGraphAU",
  "OpenFace_Emotions",
  "OpenFace_AU",
  "OpenFace_Gaze"
]

emotion_mapping = {
  "DDAMFN_Aff7": "7",
  "DDAMFN_Aff8": "8",
  "DDAMFN_RAF": "7",
  "OpenFace_Emotions": "8"
}

au_mapping = {
  "MEGraphAU_BP4D_FLD1": "BP4D",
  "MEGraphAU_BP4D_FLD2": "BP4D",
  "MEGraphAU_BP4D_FLD3": "BP4D",
  "MEGraphAU_DISFA_FLD1": "DISFA",
  "MEGraphAU_DISFA_FLD2": "DISFA",
  "MEGraphAU_DISFA_FLD3": "DISFA",
  "OpenGraphAU": "HYBRID",
  "OpenFace_AU": "DISFA"
}

emotion_request = "DDAMFN_Aff7"
au_request = "OpenGraphAU"

emotion_labels = EMOTIONS[emotion_mapping[emotion_request]]
au_template = AU_NAMES[au_mapping[au_request]]
au_labels = AU_NAMES["DISFA"] # must be a strict subset of the requested model

def scan_recordings():

  if not RECORDINGS_ROOT.exists():
    print(f"Recordings folder not found at: {RECORDINGS_ROOT}")
    return []

  video_paths = sorted(RECORDINGS_ROOT.rglob("*.mraw.zst"))

  print(f"Found {len(video_paths)} raw video file(s):")

  return video_paths

def process_video(video_path: Path):

  videoStart = time.perf_counter()

  meta_path = video_path.with_name(Path(video_path.stem).stem + ".json")
  hasMeta = meta_path.exists()
  if hasMeta:
    with open(meta_path, "r", encoding="utf-8") as jf:
      meta = json.load(jf)
  else:
    meta={}
    print(f"⚠️  Missing metadata for {video_path}.")

  base = video_path.stem      # removes .zstd
  base = Path(base).stem      # removes .mraw

  processed_path = PROCESSED_ROOT / f"{base}.npz"
  preview_mp4_path = PROCESSED_ROOT / f"{base}.mp4"

  hasInference = processed_path.exists()

  if hasInference:
    inference_data = np.load(processed_path)
  else:
    print(f"No inference data for {base}: Video has not been processed. Skipping.")
    return False

  preview_fps = 54.0
  video_writer = None
  preview_size = None
  inference_preview = None

  totalFrames = None

  header_format = meta.get("header_format", "<HHHHHfffffff")
  header_bytes = struct.calcsize(header_format)
  frames = meta.get("frames", None)
  if frames is not None:
    totalFrames = len(frames)

  frame_idx = 0

  dctx = zstd.ZstdDecompressor()

  errorFlag=False

  with open(video_path, "rb") as f, dctx.stream_reader(f) as reader:

    while True:

      try:
        header_buf = reader.read(header_bytes)
      except zstd.ZstdError as e:
        print(f"⚠️ ZSTD corruption in {base} at frame {frame_idx}: {e}")
        errorFlag = True
        break
        
      if not header_buf:
        # Normal EOF
        break

      if len(header_buf) < header_bytes:
        print(f"⚠️ Truncated header at frame {frame_idx}, stopping.")
        errorFlag = True
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
        errorFlag = True
        break

      # --- Read Bayer pixels ---
      pixel_bytes = cropWidth * cropHeight * bytes_per_sample

      try:
        frame_buf = reader.read(pixel_bytes)
      except zstd.ZstdError as e:
        print(f"⚠️ ZSTD corruption in {base} at frame {frame_idx}: {e}")
        errorFlag = True
        break
      
      if len(frame_buf) < pixel_bytes:
        print(
          f"⚠️ Truncated frame data at frame {frame_idx} "
          f"(got {len(frame_buf)} / {pixel_bytes}), stopping."
        )
        errorFlag = True
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

      # Lazily set up overlay and mp4 writer (Quality Assurance)
      if inference_preview is None:
        frame_h, frame_w = aligned_224_8bit.shape[:2]

        emotion_indices = list(range(len(emotion_labels)))

        # Indices de esas AUs dentro del vector completo
        au_indices = [au_template.index(au) for au in au_labels]

        inference_preview = InferencePreview(
          frame_width=frame_w,
          frame_height=frame_h,
          emotion_labels=emotion_labels,
          emotion_indices=emotion_indices,
          au_labels=au_labels,
          au_indices=au_indices,
          bottom_height=100,
          right_width=200,
        )

        preview_size = (inference_preview.CANVAS_W, inference_preview.CANVAS_H)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
          str(preview_mp4_path),
          fourcc,
          preview_fps,
          preview_size,
        )

      # Write overlay frame to mp4 if we are running both DDAMFN and BP4D
      if (
        video_writer is not None
        and inference_preview is not None
        and inference_data.get(emotion_request, None) is not None
        and inference_data.get(au_request, None) is not None
      ):
        canvas_frame = inference_preview.render(
          aligned_224_8bit,
          emo_probs=inference_data[emotion_request][frame_idx],
          au_probs=inference_data[au_request][frame_idx],
        )
        video_writer.write(canvas_frame)
      else:
        errorFlag = True
        print(f"Error: Data for at least one of the requested models is not available for {base}. Skipping.")
        available_models = [model for model in inference_data.keys() if model not in ["project", "session", "player", "round", "timestamps", "recording_index", "timestamp_routine"]]
        print(f"Available model data: {available_models}")
        break

      frame_idx += 1

  if video_writer is not None:
    video_writer.release()

  if errorFlag:
    print(f"Error: Exiting {base} on frame {frame_idx + 1} of {totalFrames}")
    return False

  videoEnd = time.perf_counter()
  print(f"Finished processing {base}")
  print(f"Completed in {videoEnd - videoStart:.2f} seconds")

  return True

# RUN SCRIPT

if __name__ == "__main__":
  
  cv2.setNumThreads(1)

  # Find all raw videos under RECORDINGS_ROOT
  video_paths = scan_recordings()

  startTime = time.perf_counter()

  max_workers = 16

  with ThreadPoolExecutor(max_workers=max_workers) as pool:
    results = list(pool.map(process_video, video_paths))

  processed_count = sum(results)

  endTime = time.perf_counter()

  print(f"Processed {processed_count} videos.")
  print(f"Total run time: {(endTime - startTime)/60:.2f} minutes")