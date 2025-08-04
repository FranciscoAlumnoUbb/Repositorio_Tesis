#!/usr/bin/env python3
"""
Script de entrenamiento para DETR Visual Transformer
Configurado para detección de objetos de reciclaje: carton, metal, papel, pila, plastico, vidrio
"""

import os
import sys
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import argparse
import warnings

# Imports locales
from model import DETRModel, create_model, save_model, load_model
from dataset_loader import COCODataset, create_data_loaders, get_class_names
from utils import compute_metrics, visualize_predictions, create_confusion_matrix

warnings.filterwarnings("ignore")


def get_config():
    """Configuración embebida sin archivo externo"""
    return {
        # Dataset
        "train_img_dir":  "./dataset/images/train",
        "train_ann_file": "./dataset/annotations/instances_train.json",
        "val_img_dir":    "./dataset/images/val",
        "val_ann_file":   "./dataset/annotations/instances_val.json",
        "num_classes":    6,
        # Modelo
        "model_name":     "facebook/detr-resnet-50",
        # Entrenamiento
        "num_epochs":     50,
        "batch_size":     8,
        "learning_rate":  0.0001,
        "weight_decay":   0.0001,
        "lr_step_size":   50,
        "lr_gamma":       0.1,
        "num_workers":    4,
        # Rutas de salida
        "output_dir":     "./outputs",
        "checkpoint_dir": "./checkpoints",
        "log_dir":        "./logs",
    }


class DETRTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando device: {self.device}")

        os.makedirs(config["output_dir"], exist_ok=True)
        os.makedirs(config["checkpoint_dir"], exist_ok=True)
        os.makedirs(config["log_dir"], exist_ok=True)

        self.writer = SummaryWriter(log_dir=config["log_dir"])
        self.model = create_model(num_classes=config["num_classes"])
        self.model.to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config["lr_step_size"],
            gamma=config["lr_gamma"],
        )

        self.train_loader, self.val_loader = create_data_loaders(
            train_img_dir=config["train_img_dir"],
            train_ann_file=config["train_ann_file"],
            val_img_dir=config["val_img_dir"],
            val_ann_file=config["val_ann_file"],
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
        )

        self.train_losses = []
        self.val_losses = []
        self.val_maps = []
        self.best_map = 0.0
        self.start_epoch = 0
        self.class_names = get_class_names()

        print(f"Modelo inicializado con {config['num_classes']} clases")
        print(f"Dataset de entrenamiento: {len(self.train_loader.dataset)} imágenes")
        print(f"Dataset de validación: {len(self.val_loader.dataset)} imágenes")

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_loader)

        progress_bar = tqdm(
            self.train_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]}'
        )

        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(self.device)

            formatted_targets = []
            for target in targets:
                formatted_targets.append({
                    "class_labels": target["labels"].to(self.device),
                    "boxes":       target["boxes"].to(self.device),
                })

            self.optimizer.zero_grad()
            outputs = self.model(pixel_values=images, labels=formatted_targets)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            self.optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({
                "Loss":     f"{loss.item():.4f}",
                "AvgLoss":  f"{epoch_loss/(batch_idx+1):.4f}",
                "LR":       f'{self.optimizer.param_groups[0]["lr"]:.2e}',
            })

            if batch_idx % 100 == 0:
                step = epoch * num_batches + batch_idx
                self.writer.add_scalar("Train/Loss_Step", loss.item(), step)
                self.writer.add_scalar(
                    "Train/Learning_Rate", self.optimizer.param_groups[0]["lr"], step
                )

        avg_loss = epoch_loss / num_batches
        self.train_losses.append(avg_loss)
        self.writer.add_scalar("Train/Loss_Epoch", avg_loss, epoch)
        return avg_loss

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validación"):
                images = images.to(self.device)

                formatted_targets = []
                for target in targets:
                    formatted_targets.append({
                        "class_labels": target["labels"].to(self.device),
                        "boxes":       target["boxes"].to(self.device),
                    })

                outputs = self.model(pixel_values=images, labels=formatted_targets)
                loss = outputs.loss
                val_loss += loss.item()

                preds = self.model.predict(images, threshold=0.5)
                all_predictions.extend(preds)
                all_targets.extend(targets)

        avg_val_loss = val_loss / len(self.val_loader)
        self.val_losses.append(avg_val_loss)

        metrics = compute_metrics(all_predictions, all_targets, self.class_names)
        self.writer.add_scalar("Val/Loss",     avg_val_loss, epoch)
        self.writer.add_scalar("Val/mAP",     metrics["mAP"], epoch)
        self.writer.add_scalar("Val/Precision", metrics["precision"], epoch)
        self.writer.add_scalar("Val/Recall",    metrics["recall"], epoch)

        if metrics["mAP"] > self.best_map:
            self.best_map = metrics["mAP"]
            best_model_path = os.path.join(
                self.config["checkpoint_dir"], "best_model.pth"
            )
            save_model(self.model, best_model_path)
            print(f"Nuevo mejor modelo guardado con mAP: {self.best_map:.4f}")

        self.val_maps.append(metrics["mAP"])
        return avg_val_loss, metrics

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            "epoch":                    epoch,
            "model_state_dict":         self.model.state_dict(),
            "optimizer_state_dict":     self.optimizer.state_dict(),
            "scheduler_state_dict":     self.scheduler.state_dict(),
            "train_losses":             self.train_losses,
            "val_losses":               self.val_losses,
            "val_maps":                 self.val_maps,
            "best_map":                 self.best_map,
            "config":                   self.config,
        }

        ckpt_path = os.path.join(
            self.config["checkpoint_dir"], f"checkpoint_epoch_{epoch}.pth"
        )
        torch.save(checkpoint, ckpt_path)

        if is_best:
            best_ckpt = os.path.join(
                self.config["checkpoint_dir"], "best_checkpoint.pth"
            )
            torch.save(checkpoint, best_ckpt)

    def plot_training_curves(self):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses,   label="Val Loss")
        plt.xlabel("Época")
        plt.ylabel("Pérdida")
        plt.title("Curvas de Pérdida")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(self.val_maps, label="mAP")
        plt.xlabel("Época")
        plt.ylabel("mAP")
        plt.title("mAP de Validación")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 3)
        lrs = [
            self.config["learning_rate"]
            * (self.config["lr_gamma"] ** (i // self.config["lr_step_size"]))
            for i in range(len(self.train_losses))
        ]
        plt.plot(lrs, label="Learning Rate")
        plt.xlabel("Época")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate")
        plt.yscale("log")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.config["output_dir"], "training_curves.png"), dpi=300
        )
        plt.close()

    def train(self):
        print("Iniciando entrenamiento...")
        start_time = time.time()

        for epoch in range(self.start_epoch, self.config["num_epochs"]):
            print(f"\nÉpoca {epoch+1}/{self.config['num_epochs']}")

            train_loss = self.train_epoch(epoch)
            val_loss, metrics = self.validate(epoch)
            self.scheduler.step()

            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"mAP: {metrics['mAP']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")

            self.save_checkpoint(epoch, is_best=(metrics["mAP"] == self.best_map))
            if (epoch + 1) % 20 == 0:
                self.plot_training_curves()

        total_time = time.time() - start_time
        print(f"\nEntrenamiento completado en {total_time/3600:.2f} horas")
        print(f"Mejor mAP: {self.best_map:.4f}")

        self.save_checkpoint(self.config["num_epochs"] - 1, is_best=False)
        self.plot_training_curves()
        self.writer.close()

    def resume_from_checkpoint(self, checkpoint_path):
        print(f"Cargando checkpoint desde: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.train_losses = checkpoint["train_losses"]
        self.val_losses   = checkpoint["val_losses"]
        self.val_maps     = checkpoint["val_maps"]
        self.best_map     = checkpoint["best_map"]
        self.start_epoch  = checkpoint["epoch"] + 1

        print(f"Checkpoint cargado exitosamente!")
        print(f"Reanudando desde época {self.start_epoch + 1}")
        print(f"Mejor mAP hasta ahora: {self.best_map:.4f}")
        print(f"Épocas completadas: {len(self.train_losses)}")


def main():
    parser = argparse.ArgumentParser(
        description="Entrenar DETR para detección de objetos de reciclaje"
    )
    parser.add_argument("--resume", type=str, help="Ruta del checkpoint para reanudar entrenamiento")
    args = parser.parse_args()

    config = get_config()

    # —— CREAR CARPETA NUMERADA POR RUN ——
    runs = glob.glob("resultados*")
    indices = [
        int(os.path.basename(r).replace("resultados", ""))
        for r in runs
        if os.path.basename(r).replace("resultados", "").isdigit()
    ]
    next_idx = max(indices, default=0) + 1
    run_dir = f"resultados{next_idx}"

    config["output_dir"]     = os.path.join(run_dir, "outputs")
    config["checkpoint_dir"] = os.path.join(run_dir, "checkpoints")
    config["log_dir"]        = os.path.join(run_dir, "logs")

    os.makedirs(config["output_dir"],     exist_ok=True)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["log_dir"],        exist_ok=True)

    print(f"Guardando esta corrida en: {run_dir}")

    print("Configuración de entrenamiento:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    trainer = DETRTrainer(config)
    if args.resume:
        trainer.resume_from_checkpoint(args.resume)
    trainer.train()


if __name__ == "__main__":
    main()

