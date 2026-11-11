import torch
from torchvision import transforms
from .networks.DDAM import DDAMNet
from PIL import Image
from typing import Dict
from openvino import Core
import numpy as np




data_transforms_val = transforms.Compose([
#    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])]) 



class7_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry'] 
class8_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry', 'Contempt'] 


def load_ddamfn_model(device: torch.device, 
                      model_path: str,
                      num_class: int = 7,
                      num_head: int = 2,
                      ):
    """
    Load the DDAMFN model with given config.
    """
    model = DDAMNet(num_class=num_class, num_head=num_head, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model
  

try:
    _inf = torch.inference_mode()
except AttributeError:
    _inf = torch.no_grad()

@_inf
def predict_emotion(
    pil_img: Image.Image,
    model: torch.nn.Module,
    device: torch.device,
    transform: transforms.Compose = data_transforms_val,
    num_class: int = 7,
) -> Dict[str, float]:
  
    """Return {label: prob} for a single PIL image using DDAMFN."""

    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    logits, _, _ = model(img_tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    labels = class7_names if num_class == 7 else class8_names

    return {label: float(prob) for label, prob in zip(labels, probs)}


# OPENVINO ALTERNATIVE

def load_ddamfn_model_ov(
    model_path: str = "./ddamfn/openvino/ddamfn_affectnet7.xml",
    device_name: str = "CPU",
    num_threads: int = 4,
):
    core = Core()
    core.set_property(
        "CPU",
        {
            "INFERENCE_NUM_THREADS": num_threads,
        }
    )
    ov_model = core.read_model(model_path)
    compiled = core.compile_model(ov_model, device_name)
    inp = compiled.input(0)
    out = compiled.output(0)
    return compiled, inp, out


def predict_emotion_ov(
    pil_img: Image.Image,
    compiled_model,
    input_info,
    output_info,
    transform: transforms.Compose = data_transforms_val,
    num_class: int = 7,
) -> Dict[str, float]:
    img_tensor = transform(pil_img).unsqueeze(0)       # (1, 3, 112, 112)
    img_np = img_tensor.numpy()

    infer_request = compiled_model.create_infer_request()
    result_dict = infer_request.infer({input_info.any_name: img_np})
    logits = result_dict[output_info].squeeze(0)
    exps = np.exp(logits - np.max(logits))
    probs = exps / exps.sum()
    labels = class7_names if num_class == 7 else class8_names
    return {label: float(p) for label, p in zip(labels, probs)}
