# Face inference

### Camera acquisition, live facial inference, and behavioral time-series processing

**Face inference** connects experimental interactions to facial measurements. It controls laboratory cameras, supplies live facial-expression estimates to animated avatars, and processes recordings into timestamped emotion, action-unit, and gaze outputs.

Developed for the [**Treasure Hunters**](https://github.com/nicolasgonzalezr/treasure-hunters-bargaining-game) bargaining game, the application integrates pretrained computer-vision models with camera acquisition, experimental timing, recording, and offline processing. During avatar negotiation, a local WebSocket service supplies the facial signal used by [**Avatar animation**](https://github.com/nicolasgonzalezr/avatar-animation). During face-to-face negotiation, it records camera data for subsequent analysis.

## Original models and implementations

The facial-analysis models and pretrained weights come from the following projects. Their authors' repositories provide the original methods, checkpoints, licenses, and citation guidance.

| Model / original repository | Role in this pipeline |
|------------------------------------|------------------------------------|
| [**RetinaFace — PyTorch implementation**](https://github.com/biubug6/Pytorch_Retinaface) | Face detection and landmarks used for alignment and crop preparation. This is the implementation referenced by the local alignment code. |
| [**DDAMFN**](https://github.com/SainingZhang/DDAMFN) | Facial expression recognition. The live service uses the AffectNet seven-class checkpoint; offline processing includes AffectNet seven/eight-class and RAF-DB checkpoints. |
| [**ME-GraphAU**](https://github.com/CVI-SZU/ME-GraphAU) | Facial action-unit recognition using BP4D and DISFA models. |
| [**OpenGraphAU**](https://github.com/lingjivoo/OpenGraphAU) | Facial action-unit recognition built on the ME-GraphAU architecture and pretrained on a large image bank. |
| [**OpenFace 3.0**](https://github.com/CMU-MultiComp-Lab/OpenFace-3.0) | Offline emotion recognition, action-unit detection, and gaze estimation. |

This repository brings these components together through acquisition and processing code. Cite the relevant original model papers when using their outputs.

## Part of a connected research platform

| Project | Responsibility |
|------------------------------------|------------------------------------|
| [**Treasure Hunters**](https://github.com/nicolasgonzalezr/treasure-hunters-bargaining-game) | Experimental task, condition assignment, participant matching, questionnaires, offers, and outcomes. |
| [**Avatar animation**](https://github.com/nicolasgonzalezr/avatar-animation) | Statistical stimulus generation, avatar rendering, and controlled facial-response dynamics. |
| **Face inference — this repository** | Camera control, live FER, facial recordings, offline model inference, and inspection tools. |

The same local service supports two experimental modes:

``` mermaid
flowchart TB
    C["Basler camera"] -->|"Avatar phase"| L["Live alignment and FER"]
    C -->|"Face-to-face phase"| R["Raw recording"]
    L <-->|"Facial estimates and avatar state"| B["oTree browser + avatar animation"]
    R --> P["Offline preparation and inference"]
    L -->|"Inference logs"| D["Timestamped research outputs"]
    P -->|"Emotion, AU and gaze estimates"| D
```

## Live acquisition and inference

The FastAPI application in [`main.py`](main.py) runs on each participant terminal and accepts WebSocket connections at `/ws`. The browser controlled by the avatar animation script opens the WebSocket and passes the metadata from the oTree app for the recording logs. Supported modes are `avatar` and `faceToFace`. The service uses the supplied round-start time to reference local timestamps to the experimental round.

**Avatar mode.** Camera frames are rotated, aligned using RetinaFace, and classified using DDAMFN. Emotion probabilities are sent to the browser, where the avatar animation uses the `Happy` estimate to drive its experimental smile response. Avatar smile, smoothed mimicry, and spontaneous-smile values return to the service and are recorded with the inference results.

The classifier uses PyTorch on available CUDA or Apple MPS devices and OpenVINO on the CPU. RetinaFace alignment runs on the CPU.

**Face-to-face mode.** The service switches to raw camera recording, storing Bayer pixel data with per-frame timestamps and session, participant, and round metadata.

*Note.* Frames are rotated because cameras were set up in vertical position in this project to maximize pixels across the face. The main file will create a `/Recordings` folder inside the project where all raw videos and logs will be stored.

## Offline processing

### Prepare the recordings

After an experimental session, [`recordingProcessing.py`](recordingProcessing.py) reads raw video recordings and metadata from face-to-face rounds, rotates and debayers frames for face detection, and computes alignment transforms (i.e., extracts face crops) for efficient storage. It preserves the corresponding **12-bit Bayer face crops in 16-bit containers**, with Zstandard compression, timestamps, original frame indices, and crop-local alignment matrices. Separate MP4 previews support visual inspection. Frames without a usable face crop are counted and excluded from the processed recording. A processing index tracks completed files.

The script reads all the files in `/Recordings` and creates a `/ProcessedRecordings` folder inside the project with the finished output for extraction. `transferData.ps1` handles data transfer with SHA-256 integrity verification from session terminals to external storage.

### Extract facial features

[`FERAUPPG.py`](FERAUPPG.py) reconstructs aligned images from the compressed crops and runs batched model inference. Outputs are stored as compressed NumPy arrays with timestamps and experiment identifiers. Per-model processing records support subsequent runs and merging of additional outputs.

*Note.* Some models may be commented out of the pipeline because the GPU cluster used for offline processing in this project cannot accommodate instances of all the models simultaneously.

### Inspect the results

[`inferencePreview.py`](inferencePreview.py) combines the processed face video with the extracted emotion and action-unit estimates to generate annotated MP4 previews.

<https://github.com/user-attachments/assets/3bd7757d-dade-4a35-893b-71e1387c79b6>

*Note.* Bottom bars show selected AU estimates from OpenGraphAU; right-hand bars show AffectNet-7 emotion estimates from DDAMFN.

## Setup and operation

The application was developed with **Python 3.11.13**. Create and activate a virtual environment, then install the pinned dependencies from the repository root:

``` bash
python -m pip install -r requirements.txt
```

### Camera and model configuration

1.  **Connect a compatible Basler camera.** The live service opens the first camera detected through `pypylon`. The Treasure Hunters laboratory used Basler pulse puA1280-54uc cameras.
2.  **Configure the camera presets.** The current project used `UserSet3` for avatar inference and `UserSet2` for face-to-face recording. Configure each preset as required or directly supply camera settings in code. The recording pipeline expects 12-bit Bayer data stored as `uint16`.
3.  **Supply ME-GraphAU and OpenFace 3.0 separately for offline inference.** Download the sources from the [ME-GraphAU repository](https://github.com/CVI-SZU/ME-GraphAU) into `MEGraphAU/` and the [OpenFace 3.0 repository](https://github.com/CMU-MultiComp-Lab/OpenFace-3.0) into `OpenFace3/`, both at this repository’s root. Obtain the required pretrained checkpoints following each project’s instructions and place them at the paths listed below. `FERAUPPG.py` imports the models from these folders.
4.  **Prepare the model checkpoints.** Obtain the required weights from the original model repositories above and use the paths expected by the loaders.

| Model resources | Expected location |
|------------------------------------|------------------------------------|
| RetinaFace MobileNet checkpoint | `ddamfn/Retinaface_alignment_ver2_0/weights/mobilenet0.25_Final.pth` |
| DDAMFN checkpoints | `ddamfn/checkpoints_ver2.0/` |
| DDAMFN pretrained backbone | `ddamfn/pretrained/MFN_msceleb.pth` |
| Converted DDAMFN OpenVINO model | `ddamfn/openvino/ddamfn_affectnet7.xml` and its `.bin` file |
| ME-GraphAU / OpenGraphAU checkpoints | `MEGraphAU/checkpoints/` |
| OpenFace multitask checkpoint | `OpenFace3/weights/MLT_backbone.pth` |

Use `convert_ddamfn_to_openvino.py` to regenerate the OpenVINO model when needed. Offline processing loads its configured models at startup, so their checkpoints must be available before running the script.

### Start the local service

For a local HTTP setup:

``` bash
python -m uvicorn main:app --host 127.0.0.1 --port 8001 --workers 1 --no-access-log
```

Platform launchers are provided in `Start-FaceInference.ps1` and `Start-FaceInference.sh`. The PowerShell launcher uses HTTPS, expects local `localhost.pem` and `localhost-key.pem` files, and sets CPU affinity for efficient real-time inference on the CPU. Adjust its processor selection for the terminal hardware. The shell launcher starts the HTTP service.

The WebSocket scheme must match the browser setup: `ws://localhost:8001/ws` for HTTP or `wss://localhost:8001/ws` with a trusted local certificate for HTTPS. The service's origin allowlist includes the laboratory hostname `treasurehunters.local` and local/private-network addresses.

### Run offline processing

From the repository root, after configuring the input and output paths:

``` bash
python recordingProcessing.py
python FERAUPPG.py
python inferencePreview.py
```

The preparation and inference stages use separate directory settings. Transfer or synchronize the prepared recordings to the inference input location before running the second stage.

| Stage | Current path configuration |
|------------------------------------|------------------------------------|
| Acquisition | Writes to `Recordings/Nico/`. Adjust subfolder name as required. |
| Recording preparation | Reads `Recordings/` and writes `ProcessedRecordings/`, preserving subfolders. |
| Batch inference | Reads `RECORDINGS_ROOT` and writes `PROCESSED_ROOT` in `FERAUPPG.py`; defaults use the laboratory's mounted storage under `~/mnt/gdrive/TesisData/`. Adjust as required. |
| Inference previews | Uses the corresponding storage paths in `inferencePreview.py`. |

Adjust CPU affinity and worker counts in `recordingProcessing.py` for the processing machine. Batch inference selects CUDA, then Apple MPS, then CPU according to availability.

## Data and experiment integration

| Output | Format | Contents |
|------------------------|------------------------|------------------------|
| Live inference logs | `.msgpk` + JSON header | Emotion probabilities, avatar state, timestamps, and identifiers. |
| Raw recordings | `.mraw` + JSON metadata | Camera pixel data, dimensions, pixel format, timestamps, and experiment metadata. |
| Prepared recordings | `.mraw.zst` + JSON metadata | Compressed Bayer crops, alignment transforms, timestamps, original-frame mapping, and rejected-frame counts. |
| Extracted facial signals | `.npz` + JSON processing record | Model-specific arrays, timestamps, identifiers, and completed-model records. |
| Inspection videos | `.mp4` | Aligned-face previews and annotated inference videos. |

## Attribution and licensing

Model architectures, upstream implementations, and pretrained weights retain their authors' licenses and attribution requirements. Refer to the **original repositories linked above** for the applicable terms and model citations.

## Contact

For questions about acquisition, model integration, or recording processing:

**Nicolás González-Raposo** — [nicgonzalezr\@udd.cl](mailto:nicgonzalezr@udd.cl) · [ORCID](https://orcid.org/0000-0002-2723-0354)
