import cv2
from pathlib import Path
import numpy as np
import torch
import json
import os
from multiprocessing import current_process
import ctypes  # for Windows affinity (if needed)
from concurrent.futures import ProcessPoolExecutor, as_completed
import zstandard as zstd
import struct
import time
import math
import shutil
from ddamfn.Retinaface_alignment_ver2_0.face_align_v2_0 import load_retinaface_model, align_face_unscaled

retina_model = None
retina_cfg = None
retina_path = './ddamfn/Retinaface_alignment_ver2_0/weights/mobilenet0.25_Final.pth'

# Root folders (relative to this script)
SCRIPT_DIR = Path(__file__).resolve().parent
RECORDINGS_ROOT = SCRIPT_DIR / "Recordings"
MASTER_INDEX_PATH = SCRIPT_DIR / "ProcessedLog.json"

# Number of parallel worker processes to use for videos
CPU_COUNT = os.cpu_count() or 2
CORE_IDS = [0, 2, 4, 6, 8, 10]  # for example, if those map to P-cores
MAX_WORKERS = len(CORE_IDS)

def set_process_affinity_to_core(core_id: int):
    """
    Pin current process to a single logical CPU (Windows only).
    Silently continues if the call cannot be made.
    """
    try:
        pid = os.getpid()

        PROCESS_SET_INFORMATION = 0x0200
        PROCESS_QUERY_INFORMATION = 0x0400

        # Open process with needed rights
        handle = ctypes.windll.kernel32.OpenProcess( # type: ignore
            PROCESS_SET_INFORMATION | PROCESS_QUERY_INFORMATION,
            False,
            pid
        ) 

        if handle:
            mask = 1 << core_id
            result = ctypes.windll.kernel32.SetProcessAffinityMask(handle, mask) # type: ignore
            ctypes.windll.kernel32.CloseHandle(handle) # type: ignore

            if result == 0:
                print(f"   [worker] Failed to pin to CPU {core_id}")
                
        else:
            print("   [worker] Could not obtain process handle for affinity pinning.")

    except Exception as e:
        print(f"   [worker] Affinity pin failed: {e}")


def init_worker(core_ids):
    """
    Called once inside each worker process.
    Ensures:
    - CPU affinity is applied BEFORE any heavy CPU kernels spawn threads
    - PyTorch thread pools are restricted
    - Workers do not all load the RetinaFace model at the same time
    """

    # ---------------------------------------------------------
    # 1. Resolve worker index
    # ---------------------------------------------------------
    proc = current_process()
    if proc._identity:
        worker_idx = proc._identity[0] - 1
    else:
        worker_idx = 0

    # ---------------------------------------------------------
    # 2. Pin affinity IMMEDIATELY (before ANY torch operations!)
    # ---------------------------------------------------------
    if core_ids:
        core_id = core_ids[worker_idx % len(core_ids)]
        set_process_affinity_to_core(core_id)
        print(f"   [worker {worker_idx}] Affinity pinned to CPU {core_id}")
    else:
        print(f"   [worker {worker_idx}] No affinity provided")

    # ---------------------------------------------------------
    # 3. Restrict PyTorch thread usage (must come BEFORE model load)
    # ---------------------------------------------------------

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception as e:
        print(f"   [worker {worker_idx}] Could not set PyTorch thread limits: {e}")

    # ---------------------------------------------------------
    # 4. Stagger model loading to avoid all workers blocking CPU
    # ---------------------------------------------------------
    time.sleep(0.35 * worker_idx)

    # ---------------------------------------------------------
    # 5. Load RetinaFace model LAST (after affinity + thread limits)
    # ---------------------------------------------------------
    global retina_model, retina_cfg
    retina_model, retina_cfg = load_retinaface_model(
        device=torch.device("cpu"),
        model_path=retina_path
    )

    print(f"   [worker {worker_idx}] RetinaFace loaded")

def find_raw_videos(RECORDINGS_ROOT):

    if not RECORDINGS_ROOT.exists():
        print(f"Recordings folder not found at: {RECORDINGS_ROOT}")
        return []
    
    processed_root = SCRIPT_DIR / "ProcessedRecordings"

    inference_paths = sorted(RECORDINGS_ROOT.rglob("*.msgpk"))
    
    if not inference_paths:
        print(f"No inference files found under: {RECORDINGS_ROOT}")
    else:
        for infer in inference_paths:
            # Mirror the directory structure under Recordings/
            relative_path = infer.relative_to(RECORDINGS_ROOT)
            processed_dir = processed_root / relative_path.parent
            processed_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy2(infer, processed_dir / infer.name)

            header = infer.with_name(f"{infer.stem}.json")
            if header.exists():
                shutil.copy2(header, processed_dir / header.name)

            print(f"Copied {infer.stem} to ProcessedRecordings")

    video_paths = sorted(RECORDINGS_ROOT.rglob("*.mraw"))

    print(f"Found {len(video_paths)} raw video file(s):")
    for path in video_paths:
        print(path)

    return video_paths

def debayer_rotate(frame: np.ndarray) -> np.ndarray:
    """
    Debayer a single Bayer frame (uint16 → uint16 BGR),
    then rotate 90° clockwise because camera is mounted vertically.
    """

    # Rotate 90° clockwise
    bayer_rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) # NOW ITS BAYER BG!!! ASSUMES ORIGINAL IS GR
    # Debayer
    rgb_rotated = cv2.cvtColor(bayer_rotated, cv2.COLOR_BayerBG2RGB) #ASSUMES BAYER BG

    return bayer_rotated, rgb_rotated

def crop_bayer(
    bayer_img: np.ndarray,
    M: np.ndarray,
    side: int,
    img_height: int,
    img_width: int,
    pad: int = 4,
):
    """
    Compute the bounding box in the *rotated* Bayer/RGB image that contains
    all pixels used by warpAffine when generating a side×side aligned face.

    Args:
        bayer_img  : rotated Bayer frame (H, W), uint16
        M          : 2x3 affine matrix mapping *source → canvas* (as returned by norm_crop_unscaled)
        side       : side length of the aligned canvas
        img_height : height of rotated frame (raw_rot / rgb_rot)
        img_width  : width of rotated frame
        pad        : safety padding in pixels

    Returns:
        (bayer_crop, x_min, y_min), where:
            - bayer_crop is the cropped Bayer patch (still raw, uint16)
            - (x_min, y_min) is the origin of the crop in the rotated frame

        or None if a valid crop cannot be computed.
    """

    # --- 1) Build homogeneous 3x3 matrix and invert ---
    # M maps: source (frame) → destination (canvas)
    # For cropping, we want dst → src, so we invert it.
    T = np.eye(3, dtype=np.float32)
    T[:2, :] = M.astype(np.float32)  # top 2 rows = M
    T_inv = np.linalg.inv(T)
    M_inv = T_inv[:2, :]             # 2x3, maps canvas → source

    # --- 2) Corners of the aligned canvas in destination coords ---
    dst_corners = np.array(
        [
            [0,         0        ],        # top-left
            [side - 1,  0        ],        # top-right
            [0,         side - 1 ],        # bottom-left
            [side - 1,  side - 1 ],        # bottom-right
        ],
        dtype=np.float32,
    )

    ones = np.ones((4, 1), dtype=np.float32)
    dst_h = np.hstack([dst_corners, ones])   # shape (4, 3)

    # --- 3) Map canvas corners back into rotated frame coords ---
    src = (M_inv @ dst_h.T).T                # shape (4, 2)
    xs = src[:, 0]
    ys = src[:, 1]

    # --- 4) Continuous bbox in rotated coords, with padding ---
    x_min_f = float(xs.min()) - pad
    x_max_f = float(xs.max()) + pad
    y_min_f = float(ys.min()) - pad
    y_max_f = float(ys.max()) + pad

    # --- 5) Discretize to integer indices (outward rounding) ---
    x_min = max(0, int(math.floor(x_min_f)))
    y_min = max(0, int(math.floor(y_min_f)))
    x_max = min(img_width,  int(math.ceil(x_max_f)))
    y_max = min(img_height, int(math.ceil(y_max_f)))

    if x_max <= x_min or y_max <= y_min:
        return None

    # --- 6) Enforce Bayer alignment: even origin, even width/height ---
    if x_min % 2 != 0:
        x_min = max(0, x_min - 1)
    if y_min % 2 != 0:
        y_min = max(0, y_min - 1)

    width = x_max - x_min
    height = y_max - y_min

    if width % 2 != 0 and x_max < img_width:
        x_max += 1
        width = x_max - x_min
    if height % 2 != 0 and y_max < img_height:
        y_max += 1
        height = y_max - y_min

    if width <= 0 or height <= 0:
        return None

    # --- 7) Final crop ---
    bayer_crop = bayer_img[y_min:y_max, x_min:x_max]

    return bayer_crop, x_min, y_min


def process_raw_video(mraw_path: Path):
    """Process a single .mraw + .json pair into face-crop .mraw + mp4 + new json."""
    meta_path = mraw_path.with_suffix(".json")
    if not meta_path.exists():
        print(f"⚠️  Missing metadata for {mraw_path}, skipping.")
        return False

    with open(meta_path, "r", encoding="utf-8") as jf:
        meta = json.load(jf)

    width = int(meta["width"])
    height = int(meta["height"])
    frame_count = int(meta["frame_count"])
    timestamps = []

    if meta["project"] == "Nico":
        timestamps = meta.get("timestamps", [])
    elif meta["project"] == "Edu":
        timestamps = meta.get("timestamps_session", [])
    else:
        print("Error: project meta does not match either 'Nico' nor 'Edu'. Timestamps not loaded")

    timestamps_routine = meta.get("timestamps_routine", [])

    if len(timestamps) != frame_count:
        print(f"⚠️  Timestamp/frame_count mismatch in {meta_path}, skipping.")
        return False

    bytes_per_frame = width * height * 2  # uint16 Bayer
    # New root for processed outputs
    processed_root = SCRIPT_DIR / "ProcessedRecordings"

    # Mirror the directory structure under Recordings/
    relative_path = mraw_path.relative_to(RECORDINGS_ROOT)
    processed_dir = processed_root / relative_path.parent
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Copy original json to processed folder
    if meta_path.exists():
        shutil.copy2(meta_path, processed_dir / f"{meta_path.stem}_raw.json")

    base = mraw_path.stem

    faces_mraw_path = processed_dir / f"{base}.mraw.zst"
    faces_json_path = processed_dir / f"{base}.json"
    preview_mp4_path = processed_dir / f"{base}.mp4"

    # Initialize Zstd streaming compressor
    cctx = zstd.ZstdCompressor(level=10)
    fout = cctx.stream_writer(open(faces_mraw_path, "wb"))

    print(f"\n▶ Processing {mraw_path.name}")

    # Minimal: fixed FPS for preview (just for visual inspection)
    preview_fps = 54.0

    processed_entries = []
    processed_idx = 0
    lost_frames = 0
    video_writer = None
    preview_size = None

    initTime = time.perf_counter()

    with open(mraw_path, "rb") as fin:
        for original_idx in range(frame_count):
            raw = fin.read(bytes_per_frame)
            if len(raw) < bytes_per_frame:
                print("Error, bytes per frame mismatch")
                break

            # Bayer16 → (H,W) uint16
            rawFrame = np.frombuffer(raw, dtype=np.uint16).reshape(height, width)


            # Debayer + rotate, still 16-bit
            bayer, rgb16 = debayer_rotate(rawFrame)

            # RetinaFace alignment (returns RGB PIL image or None)
            aligned = align_face_unscaled(
                frame=rgb16, # Assumes int16 input
                net=retina_model,
                cfg=retina_cfg,
                device=torch.device("cpu")
            )

            if aligned is None:
                # No face → skip this frame in processed set
                lost_frames += 1
                continue

            M, side = aligned  # UNPACK: M is 2x3 (source → canvas), side is int

            raw_height, raw_width = bayer.shape[:2]

            # New crop_bayer returns (bayer_crop, x_min, y_min)
            crop_result = crop_bayer(bayer, M, side, raw_height, raw_width)
            if crop_result is None:
                lost_frames += 1
                continue

            bayer_crop, x_min, y_min = crop_result

            # Build crop-local affine: maps crop coords → canvas
            # Original: x' = A x_full + t
            # With crop coords x_c = x_full - o, o = (x_min, y_min):
            #   x' = A (x_c + o) + t = A x_c + (A o + t)
            A = M[:, :2].astype(np.float32)      # 2x2
            t_vec = M[:, 2].astype(np.float32)   # 2,
            o_vec = np.array([x_min, y_min], dtype=np.float32)

            t_crop = A @ o_vec + t_vec          # 2,
            M_crop = np.zeros((2, 3), dtype=np.float32)
            M_crop[:, :2] = A
            M_crop[:, 2] = t_crop

            rgb_crop = cv2.cvtColor(bayer_crop, cv2.COLOR_BayerBG2RGB)  # ASSUMES BAYER BG

            aligned = cv2.warpAffine(rgb_crop, M_crop, (side, side))
            
            if original_idx == 0:
                
                rgb_crop_vis = aligned.astype(np.float32)
                rgb_crop_vis -= rgb_crop_vis.min()
                max_val = rgb_crop_vis.max()
                if max_val > 0:
                    rgb_crop_vis /= max_val
                rgb_crop_vis = (rgb_crop_vis * 255.0).astype(np.uint8)

                rgb16_vis = rgb16.astype(np.float32)
                rgb16_vis -= rgb16_vis.min()
                max_val = rgb16_vis.max()
                if max_val > 0:
                    rgb16_vis /= max_val
                rgb16_vis = (rgb16_vis * 255.0).astype(np.uint8)

                # Guardar directamente como PNG de 16 bits, sin tocar intensidades
                cv2.imwrite(str(processed_dir / f"{base}_aligned.png"), rgb_crop_vis)
                cv2.imwrite(str(processed_dir / f"{base}_original.png"), rgb16_vis)

            aligned_8 = cv2.convertScaleAbs(aligned, alpha=255.0/4095.0)

            cropHeight, cropWidth = bayer_crop.shape[:2]

            # Choose preview size once, using the FIRST aligned 8-bit frame:
            if preview_size is None:
                preview_size = (256, 256)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    str(preview_mp4_path),
                    fourcc,
                    preview_fps,
                    preview_size
                )

            # ALWAYS resize before writing to mp4:
            preview_frame = cv2.resize(aligned_8, preview_size)
            
            if video_writer is not None:
                video_writer.write(preview_frame)
            
            # --- Build extended header ---
            # Layout (little-endian):
            #   uint16 magic              (0xFACE)
            #   uint16 width              (cropWidth)
            #   uint16 height             (cropHeight)
            #   uint16 side               (canvas size for alignment)
            #   uint16 bytes_per_sample   (2 for uint16 Bayer)
            #   float32 timestamp
            #   float32 M00, M01, M02     (first row of M_crop)
            #   float32 M10, M11, M12     (second row of M_crop)
            ts_f32   = np.float32(timestamps[original_idx])

            header = struct.pack(
                "<HHHHHfffffff",
                0xFACE,                     # magic
                cropWidth,                  # width
                cropHeight,                 # height
                2,                          # bytes per sample (uint16)
                side,                       # aligned canvas side
                ts_f32,                     # timestamp
                np.float32(M_crop[0, 0]),   # M00
                np.float32(M_crop[0, 1]),   # M01
                np.float32(M_crop[0, 2]),   # M02
                np.float32(M_crop[1, 0]),   # M10
                np.float32(M_crop[1, 1]),   # M11
                np.float32(M_crop[1, 2]),   # M12
            )


            # Store RGB crop as true uint16 (scientific data)
            # Write header + raw frame (compressed automatically)
            fout.write(header)
            fout.write(bayer_crop.tobytes())

            # Record mapping metadata
            frame_entry = {
                "processed_idx": processed_idx,
                "original_idx": original_idx,
                "timestamp": timestamps[original_idx],
                "height": cropHeight,
                "width": cropWidth,
                "side": side,
                "matrix": M_crop.tolist()
            }

            if meta["project"] == "Edu":
                frame_entry["timestamp_routine"] = timestamps_routine[original_idx]

            processed_entries.append(frame_entry)

            processed_idx += 1

        # Flush and close compressed stream
        fout.flush()
        fout.close()

    if video_writer is not None:
        video_writer.release()

    # If no frames survived, clean up empty files
    if processed_idx == 0:
        print("   ⚠️  No faces detected, removing empty outputs.")
        faces_mraw_path.unlink(missing_ok=True)
        preview_mp4_path.unlink(missing_ok=True)
        return False

    # New JSON describing the processed face-crop mraw
    new_meta = {
        "project": meta["project"],
        "session": meta["session"],
        "player": meta["player"],
        "source_file": str(mraw_path),
        "processed_file": str(faces_mraw_path),
        "preview_video": str(preview_mp4_path),
        "raw_width": width,
        "raw_height": height,
        "pixel_format": "BayerBG12",
        "bit_depth_effective": 12,
        "container_bits_per_sample": 16,
        "channels": 1,
        "dtype": "uint16",
        "frame_count_raw": frame_count,
        "frames_lost": lost_frames,
        "frames": processed_entries,
        "header_bytes": 38,
        "header_format": "<HHHHHfffffff"
    }

    if meta["project"] == "Nico":
        new_meta["round"] = meta["round"]
        new_meta["utc_start"] = meta["utc_start"]
        new_meta["utc_end"] = meta["utc_end"]

    elif meta["project"] == "Edu":
        new_meta["date"] = meta["date"]
        new_meta["session_time"] = meta["session_time"]
        new_meta["current_routine"] = meta["current_routine"]
        new_meta["recording_index"] = meta["recording_index"]
        new_meta["current_trial"] = meta["current_trial"]
    else:
        print("Error: project meta does not match either 'Nico' nor 'Edu'.")

    with open(faces_json_path, "w", encoding="utf-8") as jf:
        json.dump(new_meta, jf, indent=2)

    endTime = time.perf_counter()

    print(f"{base} | Processed frames: {processed_idx} | Lost frames: {lost_frames} | Processing time: {endTime - initTime}")
    
    return True   # so caller knows it succeeded


# RUN SCRIPT

if __name__ == "__main__":

    # Load existing master index (or initialize)
    if MASTER_INDEX_PATH.exists():
        try:
            with open(MASTER_INDEX_PATH, "r", encoding="utf-8") as jf:
                master_index = json.load(jf)
        except json.JSONDecodeError:
            master_index = {}
    else:
        master_index = {}

    processed_sources = set(master_index.get("processed_sources", []))

    # Find all raw videos under RECORDINGS_ROOT
    video_paths = find_raw_videos(RECORDINGS_ROOT)

    if not video_paths:
        print(f"No .mraw files found under: {RECORDINGS_ROOT}")
    else:
        # Filter out videos that are already processed (by filename)
        unprocessed_videos = []
        for mraw_path in video_paths:
            master_key = mraw_path.name
            if master_key in processed_sources:
                print(f"\n⏭  Skipping already processed file: {master_key}")
                continue
            unprocessed_videos.append(mraw_path)

        startTime = time.perf_counter()

        if not unprocessed_videos:
            print("\nAll found .mraw files are already processed.")
        else:
            print(f"\nLaunching parallel processing with up to {MAX_WORKERS} workers...")
            print(f"Videos to process: {len(unprocessed_videos)}")

            finished_videos = []

            # Process videos in parallel across multiple processes
            with ProcessPoolExecutor(
                max_workers=MAX_WORKERS,
                initializer=init_worker,
                initargs=(CORE_IDS,),
            ) as executor:            # Map each future to its filename (master_key)
                future_to_key = {
                    executor.submit(process_raw_video, mraw_path): mraw_path.name
                    for mraw_path in unprocessed_videos
                }

                for future in as_completed(future_to_key):
                    master_key = future_to_key[future]
                    try:
                        success = future.result()
                    except Exception as e:
                        success = False
                        print(f"\n❌ Error while processing {master_key}: {e}")

                    if success:
                        processed_sources.add(master_key)
                        finished_videos.append(master_key)
                        master_index["processed_sources"] = sorted(processed_sources)
                        with open(MASTER_INDEX_PATH, "w", encoding="utf-8") as jf:
                            json.dump(master_index, jf, indent=2)
                        print(f"   📝 Logged {master_key} as processed in {MASTER_INDEX_PATH}")
                        print(list(enumerate(sorted(finished_videos))))
                    else:
                        print(f"   ⚠️  {master_key} was not marked as processed.")
    
    endTime = time.perf_counter()
    print(f"Total time:  {endTime - startTime}.")

