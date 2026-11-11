from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import re
import json
from PIL import Image
import numpy as np
import cv2
import torch
import os
import time
from datetime import datetime, timezone
import platform
from pypylon import pylon
from threading import Event, Thread
import io, queue, msgpack
import asyncio
from threading import Condition
from collections import deque


from ddamfn.affectnet7_test import (
    load_ddamfn_model,
    predict_emotion,
    load_ddamfn_model_ov,
    predict_emotion_ov,
)

from ddamfn.Retinaface_alignment_ver2_0.face_align_v2_0 import load_retinaface_model, align_face

app = FastAPI()

if torch.backends.mps.is_available() and platform.system() == "Darwin":
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else: #USE OPENVINO, ASSUMES INTEL CPU
    device = torch.device("cpu")

print(f"✅ Using device: {device}")

ALLOWED_WS = re.compile(
    r"^https?://("
    r"localhost(:\d+)?|127(?:\.\d{1,3}){3}(:\d+)?|"         # Allow local host
    r"10(?:\.\d{1,3}){3}(:\d+)?|"                           # Allow private network LAN
    r"172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2}(:\d+)?|"   # Allow private network LAN (VPN, institutional networks)
    r"192\.168(?:\.\d{1,3}){2}(:\d+)?|"                     # Allow most common home/office LAN
    r"treasurehunters\.local(:\d+)?"                        # Allow lab HTTPS hostname
    r")$"
)

# Paths to weights
ddamfn_path = './ddamfn/checkpoints_ver2.0/affecnet7_epoch19_acc0.671.pth'
retina_path = './ddamfn/Retinaface_alignment_ver2_0/weights/mobilenet0.25_Final.pth'


# Perf knobs (guarded for CPU)

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

# On CPU, avoid oversubscription: OpenCV + MKL + event loop
cv2.setNumThreads(int(os.environ.get("OPENCV_THREADS", "1")))  # tweak to 1–2 if needed
try:
    torch.set_num_threads(int(os.environ.get("TORCH_CPU_THREADS", "1")))  # 2 is a steady default
    torch.set_num_interop_threads(1)
except Exception:
    pass

# Load RetinaFace
retina_model, retina_cfg = load_retinaface_model(device=torch.device("cpu"), model_path=retina_path)

isOpenVino = False
# --------------------
# OPENVINO BRANCH
# --------------------
ddamfn_model = None
ddamfn_ov_model = None
ddamfn_ov_inp = None
ddamfn_ov_out = None

if device.type == "cpu":
    # CPU → use OpenVINO IR
    try:
        ov_threads = int(os.environ.get("OV_NUM_THREADS", "3")) # TUNE OPENVINO THREAD USE
    except ValueError:
        ov_threads = 3

    ddamfn_ov_model, ddamfn_ov_inp, ddamfn_ov_out = load_ddamfn_model_ov(
        num_threads=ov_threads
    )
    if ddamfn_ov_model is None: #FALLBACK
        print("Error: OpenVino model load unsuccessful, falling back to torch CPU")
        ddamfn_model = load_ddamfn_model(
            device=device,
            model_path=ddamfn_path
        )
    else:
        isOpenVino = True
else:
    # GPU / MPS → use PyTorch model
    ddamfn_model = load_ddamfn_model(
        device=device,
        model_path=ddamfn_path
    )

# Optional warm-up to stabilize latency (CPU also benefits a bit)
@torch.no_grad()
def _warmup():
    dummy = np.zeros((320, 320, 3), dtype=np.uint8)  # WARM UP DIMENSIONS MUST MATCH BASLER CAMERA INPUT
    _ = align_face(frame=dummy, net=retina_model, cfg=retina_cfg, device=torch.device("cpu"))
    img = Image.new("RGB", (112, 112))

    if isOpenVino:
        _ = predict_emotion_ov(
            pil_img=img,
            compiled_model=ddamfn_ov_model,
            input_info=ddamfn_ov_inp,
            output_info=ddamfn_ov_out,
        )
    else:
        _ = predict_emotion(pil_img=img, model=ddamfn_model, device=device)
_warmup()


# --- Open camera ---
cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
cam.Open()

# create a converter that turns raw Bayer into BGR8 (OpenCV/DNN-friendly)
#converter = pylon.ImageFormatConverter()
#converter.OutputPixelFormat   = pylon.PixelType_BGR8packed   # <- BGR
#converter.OutputBitAlignment  = pylon.OutputBitAlignment_MsbAligned

def ddamfn(alignedQueue, alignedCond, resultQueue, writerQueue, terminalOffset, stopEvent):
    frame_count = 0
    last_log_time = time.perf_counter()
    accum_infer = 0.0
    
    while not stopEvent.is_set():
        with alignedCond:
            while not alignedQueue:
                alignedCond.wait()        # Blocks until notify(), no CPU use
            aligned = alignedQueue.popleft()

        t2 = time.perf_counter()

        # 2. Run inference
        if isOpenVino: # BRANCH FOR OPENVINO
            # CPU → OpenVINO IR
            result = predict_emotion_ov(
                pil_img=aligned['frame'],
                compiled_model=ddamfn_ov_model,
                input_info=ddamfn_ov_inp,
                output_info=ddamfn_ov_out,
            )
        else:
            # GPU / MPS → original PyTorch path
            with torch.inference_mode():
                result = predict_emotion(
                    pil_img=aligned['frame'],
                    model=ddamfn_model,
                    device=device,
                )

        if result is None:
            continue
        
        try:
            resultQueue.put_nowait(result)
        except queue.Full:
            resultQueue.get_nowait()
            resultQueue.put_nowait(result)
        
        # Add to inference log write queque 
        try:
            _, probs = zip(*result.items()) # Labels omitted for writing

            record = {
                "timestamp": aligned['time'] + terminalOffset, # Stamp frame capture, not inference time
                "probs": list(probs),
                "avatarSmile": aligned['avatarSmile'],
                "EMA": aligned['EMA'],
                "Spont": aligned['Spont']
            }

            writerQueue.put_nowait(record)

        except queue.Full:
            # drop silently to avoid touching latency
            pass

        t3 = time.perf_counter()

        # --- timing stats ---
        frame_count += 1
        accum_infer += (t3 - t2) * 1000
        if t3 - last_log_time >= 5.0:
            avg_infer  = accum_infer / frame_count
            fps = frame_count / (t3 - last_log_time)
            print(f"[ddamfn] {fps:.1f} FPS | infer {avg_infer:.1f} ms")
            # reset accumulators
            frame_count = 0
            accum_infer = 0.0
            last_log_time = t3

async def faceInference(websocket, camera, terminalOffset, writerQueue, stopEvent):
    frame_count = 0
    last_log_time = time.perf_counter()
    accum_decode = accum_align = 0.0
    avatarSmile = None
    t0 = None
    perfStart = time.perf_counter()

    alignedCond = Condition()
    alignedQueue = deque(maxlen=1)
    resultQueue = queue.Queue(maxsize=1)

    ddamfnThread = Thread(
        target=ddamfn,
        args=(alignedQueue, alignedCond, resultQueue, writerQueue, terminalOffset, stopEvent),
        daemon=True
    )
    ddamfnThread.start()
    
    while True:
        # Non-blocking poll for control/keepalive messages (single message => multiple fields)
        try:
            msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.025)
            payload = json.loads(msg) if msg else {}

            avatarSmile = payload.get("Smile")
            EMA         = payload.get("EMA")
            spontSmile  = payload.get("Spont")

        except asyncio.TimeoutError:
            avatarSmile = EMA = spontSmile = None

        t0 = time.perf_counter()

        # 1. Grab frame and rotate

        res = camera.RetrieveResult(100, pylon.TimeoutHandling_Return)
        if not res.GrabSucceeded():
            res.Release()
            continue

        frame = res.Array

        res.Release()

        # 90° clockwise to handle camera in vertical position
        rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
#        rotated_frame = cv2.transpose(frame_bgr)
#        rotated_frame = cv2.flip(rotated_frame, 1)  # horizontal flip

        t1 = time.perf_counter()

        with torch.inference_mode():
            aligned = align_face(
                frame=rotated_frame, 
                net=retina_model,
                cfg=retina_cfg,
                device=torch.device("cpu"),
            )
        
        if aligned is None:
#            await websocket.send_text(json.dumps({"error": "no face"}))
            continue

        alignedDict= {
                "frame": aligned,
                "time": t0 - perfStart,
                "avatarSmile": avatarSmile,
                "EMA": EMA,
                "Spont": spontSmile
        }
        
        with alignedCond:
            alignedQueue.clear()
            alignedQueue.append(alignedDict)
            alignedCond.notify()

        # Flush avatar smile queue so that only latest is read in the next iteration, keep flush before send so as to not accidentally drop the next valid value
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=0)
            except asyncio.TimeoutError:
                break

        try:
            await websocket.send_text(json.dumps(resultQueue.get_nowait()))
        except queue.Empty:
            pass
        
        t2 = time.perf_counter()

        # --- timing stats ---
        frame_count += 1
        accum_decode += (t1 - t0) * 1000
        accum_align  += (t2 - t1) * 1000
        if t2 - last_log_time >= 5.0:
            avg_decode = accum_decode / frame_count
            avg_align  = accum_align / frame_count
            print(f"[retinaface] decode {avg_decode:.1f} ms | align {avg_align:.1f} ms")
            # reset accumulators
            frame_count = 0
            accum_decode = accum_align = 0.0
            last_log_time = t2

def inferenceWriter(writerQueue: "queue.Queue", stopEvent, outPath: str, header: dict):
    """
    Background writer: pulls per-frame inference dicts from writerQueue,
    writes as MessagePack ND-stream, and flushes every ~0.5 s.
    """
    packer = msgpack.Packer(use_single_float=True)
    # Write a small JSON header once, next to the data stream
    headerPath = outPath.replace(".msgpk", ".json")
    with open(headerPath, "w", encoding="utf-8") as hf:
        json.dump(header, hf, indent=2)

    # Append-only stream of packed dicts
    with open(outPath, "wb", buffering=io.DEFAULT_BUFFER_SIZE) as f:
        lastFlush = time.time()
        while True:
            try:
                item = writerQueue.get(timeout=0.05)
                f.write(packer.pack(item))
            except queue.Empty:
                if stopEvent.is_set():   # we were told to stop and no more items arrived
                    break                 # => queue is effectively drained
                # else: keep writing

            # periodic safety flush (~1s)
            if time.time() - lastFlush >= 1:
                try:
                    f.flush()
                except Exception:
                    pass
                lastFlush = time.time()


def faceRecording(camera, sessionID, playerID, roundNumber, outDir, fileName, terminalStart, terminalOffset, stopEvent=None):

    videoPath = os.path.join(outDir, fileName + ".mraw")
    metaPath = os.path.join(outDir, fileName + ".json")

    width  = int(camera.Width.GetValue())
    height = int(camera.Height.GetValue())
    pixfmt = camera.PixelFormat.GetValue()

    frameCount = 0
    timestamps = []

    t0 = time.perf_counter()

    try:
        with open(videoPath, "wb") as f:
            while camera.IsGrabbing() and (stopEvent is None or not stopEvent.is_set()):
                res = camera.RetrieveResult(100, pylon.TimeoutHandling_Return)
                if not res:
                    continue
                if res.GrabSucceeded():
                    f.write(res.Array.tobytes())
                    frameCount += 1
                    timestamps.append(time.perf_counter() - t0 + terminalOffset)  # seconds since t0
                res.Release()
                
    finally:
        meta = {
            "project": "Nico",
            "session": sessionID,
            "player": playerID,
            "round": int(roundNumber),
            "utc_start": terminalStart.isoformat(),
            "utc_end": datetime.now(timezone.utc).isoformat(),
            "file": videoPath,
            "width": width,
            "height": height,
            "pixel_format": pixfmt,            # e.g., "BayerGR12"
            "bit_depth_effective": 12,         # your RECORD mode choice
            "container_bits_per_sample": 16,   # stored as uint16
            "frame_count": frameCount,
            "timestamps": timestamps
        }
        
        with open(metaPath, "w", encoding="utf-8") as jf:
            json.dump(meta, jf, indent=2)

        print("File saved to: ", videoPath)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    # Reject unallowed WS requests
    origin = websocket.headers.get("origin", "")
    if not ALLOWED_WS.match(origin):
        await websocket.close(code=4403)
        return
    await websocket.accept()
    msg = await websocket.receive_text()
    data = json.loads(msg)

    # Unpack the fields
    sessionID = data["session"]
    playerID = data["player"]
    roundNumber = data["roundNum"]
    roundStart = datetime.fromisoformat(data["roundStart"])
    matchType = data["matchType"]
    print("✅ Camera WebSocket connected")

    stopEvent = Event()
    recorderThread = None
    inferenceWriterThread = None
    writerQueue = None

    # Set writing directory, either for video (recording) or inference logging (avatar)
    outDir = os.path.join("Recordings", "Nico")
    os.makedirs(outDir, exist_ok=True) # Create necessary folders if not already present
    terminalStart = datetime.now(timezone.utc)
    terminalOffset = (terminalStart - roundStart).total_seconds()
    startString = terminalStart.strftime("%Y%m%dT%H%M%SZ")
    baseName = f"Nico_S{sessionID}_{playerID}_R{int(roundNumber):02d}_{startString}" # Base file name

    try:
        if matchType == "avatar":
            cam.UserSetSelector.SetValue("UserSet3") # Load camera settings
            cam.UserSetLoad.Execute() # Apply camera settings
            cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) # Set recording mode
            print("Camera loaded, starting face inference")

            # --- prepare per-session inference log paths ---
            inferPath = os.path.join(outDir, baseName + "_infer.msgpk")

            # Header with fixed metadata + class order
            emotionLabels = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry']
            header = {
                "session": sessionID,
                "player": playerID,
                "round": int(roundNumber),
                "emotions": emotionLabels,
            }

            # --- start inference writer thread ---
            writerQueue = queue.Queue(maxsize=256) # Holds inference records for the writer to save
            inferenceWriterThread = Thread(
                target=inferenceWriter,
                args=(writerQueue, stopEvent, inferPath, header),
                daemon=True
            )
            inferenceWriterThread.start()
            print("✅ Recorder thread started")

            await faceInference(websocket, cam, terminalOffset, writerQueue, stopEvent) # Start face inference

        elif matchType == "faceToFace":
            cam.UserSetSelector.SetValue("UserSet2")
            cam.UserSetLoad.Execute()
            cam.StartGrabbing(pylon.GrabStrategy_OneByOne)
            print("Camera loaded, starting face recording")

            # Start recording on a background worker
            recorderThread = Thread(
                target=faceRecording,
                args=(cam, sessionID, playerID, roundNumber, outDir, baseName, terminalStart, terminalOffset),
                kwargs={"stopEvent": stopEvent},
                daemon=True
            )
            recorderThread.start()

            # WS watcher: just wait for disconnect
            while True:
                # any small await keeps the task cancellable & detects close
                await websocket.receive_text()
        else:
            return # Skip

    except WebSocketDisconnect:
        print("🔌 WebSocket disconnected")

    except Exception as e:
        # Catch-all so an unexpected error doesn't crash the server task
        try:
            await websocket.send_text(json.dumps({"error": f"internal error: {type(e).__name__}"}))
        except Exception:
            pass
        print("❌ Unhandled error in websocket handler:", repr(e))

    finally:
        # Signal stop and cleanly stop camera
        stopEvent.set()
        try:
            cam.StopGrabbing()
        except Exception:
            pass
        if recorderThread:
            recorderThread.join(timeout=2.0)
        if inferenceWriterThread:
            inferenceWriterThread.join(timeout=2.0)
        