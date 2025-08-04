import os
import torch
import subprocess

def main():
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"✅ Entrenando en: {device}")

    command = [
        "python", "yolov5/train.py",
        "--img", "640",
        "--batch", "16",
        "--epochs", "50",
        "--data", "data/data.yaml",
        "--weights", "yolov5x.pt",
        "--device", device,
        "--hyp", "data/hyps/hyp.textura.yaml",
        "--project", "data/outputs/yolov5_training50v2",
        "--name", "exp_textura",
        "--exist-ok"
    ]
    subprocess.run(command)

if __name__ == "__main__":
    main()
