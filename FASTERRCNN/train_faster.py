import os
import torch
import torchvision.transforms as T
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
from sklearn.metrics import f1_score, average_precision_score, confusion_matrix
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
ANN_DIR = os.path.join(DATASET_DIR, "annotations")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "faster_model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_CLASSES = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["carton", "metal", "papel", "pila", "plastico", "vidrio"]
BATCH_SIZE = 8
NUM_EPOCHS = 150
LEARNING_RATE = 1e-4
CONF_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.45

# ✅ PATH DEL CHECKPOINT
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint_last.pth")
start_epoch = 1

# ---------------- TRANSFORM ----------------
def get_transform():
    return T.Compose([T.ToTensor()])

# ---------------- DATASET ----------------
def collate_fn(batch):
    return tuple(zip(*batch))

def get_dataset(split):
    imgs_dir = os.path.join(IMAGES_DIR, split)
    ann_file = os.path.join(ANN_DIR, f"instances_{split}.json")
    return CocoDetection(imgs_dir, ann_file, transform=get_transform())

train_loader = DataLoader(get_dataset("train"), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)
val_loader = DataLoader(get_dataset("val"), batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

# ---------------- MODEL ----------------
model = fasterrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
model.to(DEVICE)

# ---------------- OPTIMIZER ----------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# ---------------- SETUP METRICS ----------------
train_losses, f1_scores, ap_scores = [], [], []

# ✅ CARGAR CHECKPOINT SI EXISTE
if os.path.exists(CHECKPOINT_FILE):
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    train_losses = checkpoint["train_losses"]
    f1_scores = checkpoint["f1_scores"]
    ap_scores = checkpoint["ap_scores"]
    start_epoch = checkpoint["epoch"] + 1
    print(f"🔁 Checkpoint cargado. Reanudando desde la época {start_epoch}")

# ---------------- EVALUATE ----------------
def evaluate_on_loader(loader):
    y_true_all, y_pred_all, scores_all = [], [], []
    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="  ➤ Inferring on val"):
            img = images[0].to(DEVICE)
            gt_labels = [ann["category_id"] + 1 for ann in targets[0]]
            if not gt_labels:
                continue
            outs = model([img])[0]
            boxes = outs["boxes"].cpu()
            labels = outs["labels"].cpu().numpy()
            scores = outs["scores"].cpu().numpy()
            mask = scores > CONF_THRESHOLD
            boxes, labels, scores = boxes[mask], labels[mask], scores[mask]
            if len(scores) == 0:
                continue
            keep_idx = nms(boxes, torch.tensor(scores), iou_threshold=NMS_IOU_THRESHOLD)
            labels = labels[keep_idx.numpy()]
            scores = scores[keep_idx.numpy()]
            n = min(len(gt_labels), len(labels))
            y_true_all.extend(gt_labels[:n])
            y_pred_all.extend(labels[:n])
            scores_all.extend(scores[:n])
    if not y_true_all:
        return 0.0, 0.0, np.zeros((NUM_CLASSES - 1, NUM_CLASSES - 1), dtype=int)
    f1 = f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0)
    ap = average_precision_score(
        [1 if y_true_all[i] == y_pred_all[i] else 0 for i in range(len(y_true_all))],
        np.array(scores_all)
    )
    cm = confusion_matrix(y_true_all, y_pred_all, labels=list(range(1, NUM_CLASSES)))
    return f1, ap, cm

# ---------------- TRAINING LOOP ----------------
for epoch in range(start_epoch, NUM_EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for images, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
        imgs = [img.to(DEVICE) for img in images]
        formatted = []
        for anns in targets:
            boxes = torch.tensor([a["bbox"] for a in anns], dtype=torch.float32)
            boxes[:, 2:] += boxes[:, :2]
            labels = torch.tensor([a["category_id"] + 1 for a in anns], dtype=torch.int64)
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            formatted.append({"boxes": boxes[keep].to(DEVICE), "labels": labels[keep].to(DEVICE)})
        loss_dict = model(imgs, formatted)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    f1, ap, cm = evaluate_on_loader(val_loader)
    f1_scores.append(f1)
    ap_scores.append(ap)

    print(f"Epoch {epoch}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val F1: {f1:.3f} | Val AP: {ap:.3f}")
    print("Confusion Matrix:")
    print(cm)

    # ---------------- GUARDAR CHECKPOINT ----------------
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses": train_losses,
        "f1_scores": f1_scores,
        "ap_scores": ap_scores,
    }, CHECKPOINT_FILE)

# ---------------- GUARDAR MÉTRICAS Y PLOT ----------------
metrics_df = pd.DataFrame({
    "epoch": list(range(1, len(train_losses) + 1)),
    "train_loss": train_losses,
    "val_f1": f1_scores,
    "val_ap": ap_scores
})
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "metrics_summary.csv"), index=False)

def plot_and_save(x, ys, labels, title, ylabel, fname):
    plt.figure()
    for y, lbl in zip(ys, labels):
        plt.plot(x, y, label=lbl)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, fname))
    plt.close()

plot_and_save(metrics_df.epoch, [metrics_df.train_loss], ["Train Loss"], "Training Loss", "Loss", "train_loss_curve.png")
plot_and_save(metrics_df.epoch, [metrics_df.val_f1, metrics_df.val_ap], ["Val F1", "Val AP"], "Validation Metrics", "Metric", "val_metrics.png")

# ---------------- GUARDAR PESOS FINALES ----------------
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "fasterrcnn_model_final.pth"))
print("✅ Entrenamiento completo. Pesos y métricas finales guardados.")
