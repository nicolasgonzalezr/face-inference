# convert_ddamfn_to_openvino.py
import torch
import openvino as ov

from ddamfn.affectnet7_test import load_ddamfn_model  # your loader :contentReference[oaicite:2]{index=2}

ddamfn_path = './ddamfn/checkpoints_ver2.0/affecnet7_epoch19_acc0.671.pth'

def main():
    device = torch.device("cpu")          # conversion on CPU is fine
    model = load_ddamfn_model(device=device, model_path=ddamfn_path)
    model.eval()

    example_input = torch.randn(1, 3, 112, 112)  # matches your current pipeline :contentReference[oaicite:3]{index=3}

    ov_model = ov.convert_model(model, example_input=example_input)  # PyTorch → OV Model :contentReference[oaicite:4]{index=4}

    # Save as OpenVINO IR; compress_to_fp16=True is default (FP16 weights) :contentReference[oaicite:5]{index=5}
    ov.save_model(ov_model, "./ddamfn/openvino/ddamfn_affectnet7.xml")

if __name__ == "__main__":
    main()