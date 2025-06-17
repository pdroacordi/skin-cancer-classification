from pathlib import Path
import tensorflow as tf, tensorflow_addons as tfa
from tensorflow_addons.optimizers import AdamW

from ..config import HairRemovalConfig
from .dataset import HairRemovalDatasetKeras as HairRemovalDataset
from ..model import ChimeraNet


# ---------- DiceLoss ----------
class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth: float = 1e-6, **kwargs):
        super().__init__(name="dice_loss", **kwargs)
        self.smooth = smooth

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        inter = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
        union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])
        dice = (2.0 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class DiceMetric(tf.keras.metrics.Metric):
    """Custom Dice coefficient metric that handles serialization properly."""

    def __init__(self, name='dice', **kwargs):
        super().__init__(name=name, **kwargs)
        self.intersection = self.add_weight(name='intersection', initializer='zeros')
        self.union = self.add_weight(name='union', initializer='zeros')
        self.smooth = 1e-6

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > 0.5, tf.float32)  # Threshold at 0.5

        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)

        self.intersection.assign_add(intersection)
        self.union.assign_add(union)

    def result(self):
        dice = (2.0 * self.intersection + self.smooth) / (self.union + self.smooth)
        return dice

    def reset_state(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)

class CombinedLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.3, smooth=1e-6, **kwargs):
        super().__init__(name="bce_dice", **kwargs)
        self.alpha  = alpha
        self.smooth = smooth
        self.bce     = tf.keras.losses.BinaryCrossentropy()

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        # BCE component
        bce_loss  = self.bce(y_true, y_pred)
        # Dice component
        inter = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
        union = tf.reduce_sum(y_true + y_pred, axis=[1,2,3])
        dice_score = (2.0 * inter + self.smooth) / (union + self.smooth)
        dice_loss  = 1.0 - dice_score
        return self.alpha * bce_loss + (1.0 - self.alpha) * tf.reduce_mean(dice_loss)


# ---------- Treinador ----------
class HairRemovalTrainer:
    def __init__(self, data_root: Path, out_dir: Path, cfg: HairRemovalConfig):
        self.cfg      = cfg
        self.out_dir  = out_dir
        self.dataset  = HairRemovalDataset(data_root, cfg)
        self.model    = ChimeraNet(cfg.img_size).model

        optimizer = AdamW(weight_decay=1e-5, learning_rate=cfg.lr)
        self.model.compile(
            optimizer=optimizer,
            loss=CombinedLoss(alpha=0.3),
            metrics=[
                DiceMetric(name="dice"),
                tf.keras.metrics.MeanIoU(num_classes=2, name="iou")
            ],
        )

    def train(self):
        ds_train, ds_val = self.dataset.tf_datasets()
        ckpt = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.out_dir / "best",
            monitor="val_dice",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_dice",
            mode="max",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1
        )
        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_dice",
            mode="max",
            patience=20,
            restore_best_weights=True,
            verbose=1
        )

        self.model.fit(
            ds_train,
            validation_data=ds_val,
            epochs=self.cfg.epochs,
            callbacks=[ckpt, reduce_lr, es],
        )


# ---------- CLI ----------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True,
                        help="pasta hair/ com dermoscopic_image e hair_mask")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    cfg = HairRemovalConfig()
    HairRemovalTrainer(args.data, args.out, cfg).train()
