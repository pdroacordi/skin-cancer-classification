import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, EarlyStopping,
    TensorBoard, LearningRateScheduler
)

from preprocessing.graphic.hair_removal.config import HairRemovalConfig
from preprocessing.graphic.hair_removal.training.dataset import HairRemovalDataset
from preprocessing.graphic.hair_removal.training.metrics import (
    HybridLoss,
    F1Score,
    IoUMetric,
    DiceMetric
)


class HairRemovalTrainer:
    """Enhanced trainer with improved training strategies"""

    def __init__(self, data_root: Path, out_dir: Path, cfg: HairRemovalConfig):
        self.cfg = cfg
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.dataset = HairRemovalDataset(data_root, cfg)

        # Build model with better initialization
        from preprocessing.graphic.hair_removal.model import create_chimeranet
        self.model = create_chimeranet(cfg.img_size)

        # Initialize optimizer with gradient clipping
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=cfg.initial_lr,
            beta_1=0.9,
            beta_2=0.999,
            clipnorm=cfg.gradient_clip_norm
        )

        # Compile with enhanced loss and metrics
        self.model.compile(
            optimizer=self.optimizer,
            loss=HybridLoss(
                bce_weight=cfg.loss_weights['bce'],
                dice_weight=cfg.loss_weights['dice'],
                tversky_weight=cfg.loss_weights['tversky']
            ),
            metrics=[
                DiceMetric(name="dice"),
                IoUMetric(name="iou"),
                F1Score(name="f1_score"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall")
            ]
        )

    def cosine_annealing_scheduler(self, epoch, lr):
        """Cosine annealing learning rate scheduler"""
        epochs = self.cfg.epochs
        lr_max = self.cfg.initial_lr
        lr_min = self.cfg.min_lr

        lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(np.pi * epoch / epochs))
        return lr

    def train(self):
        """Train with enhanced strategies"""
        ds_train, ds_val = self.dataset.tf_datasets()

        total_samples = len(self.dataset)
        train_samples = int(total_samples * 0.85)
        val_samples = int(total_samples * 0.15)

        train_steps = train_samples // self.cfg.batch_size
        val_steps = max(1, val_samples // self.cfg.batch_size)

        # Enhanced callbacks
        callbacks = [
            # Best model checkpoint
            ModelCheckpoint(
                filepath=self.out_dir / "best_weights.h5",
                monitor="val_dice",
                mode="max",
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            ),

            # Checkpoint every N epochs
            ModelCheckpoint(
                filepath=str(self.out_dir / "checkpoint_{epoch:03d}.h5"),
                save_freq=25 * train_steps,  # Every 25 epochs
                save_weights_only=True,
                verbose=0
            ),

            # Learning rate scheduler
            LearningRateScheduler(self.cosine_annealing_scheduler, verbose=1),

            # Reduce on plateau as backup
            ReduceLROnPlateau(
                monitor="val_dice",
                mode="max",
                factor=0.5,
                patience=20,
                min_lr=self.cfg.min_lr,
                verbose=1
            ),

            # Early stopping with longer patience
            EarlyStopping(
                monitor="val_dice",
                mode="max",
                patience=50,
                restore_best_weights=True,
                verbose=1
            ),

            # TensorBoard logging
            TensorBoard(
                log_dir=str(self.out_dir / "logs"),
                histogram_freq=10,
                write_graph=True,
                write_images=True,
                update_freq='epoch'
            )
        ]

        # Train model
        history = self.model.fit(
            ds_train,
            validation_data=ds_val,
            epochs=self.cfg.epochs,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=1
        )

        # Save final model
        self.model.save_weights(str(self.out_dir / "final_model_weights.h5"))

        # Save training history
        np.save(str(self.out_dir / "history.npy"), history.history)

        return history

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True,
                        help="pasta hair/ com dermoscopic_image e hair_mask")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    cfg = HairRemovalConfig()
    HairRemovalTrainer(args.data, args.out, cfg).train()