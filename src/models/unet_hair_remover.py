from __future__ import annotations

import argparse
import glob
from pathlib import Path, PurePath

import albumentations as A
import cv2
import numpy as np
import segmentation_models as sm
import tensorflow as tf
from keras.saving.save import load_model
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)

# ----------------------------- CONFIG ---------------------------------
IMG_SIZE   = 256
BATCH      = 8
ENCODER    = "efficientnetb3"  # or "efficientnetb0"
WEIGHTS_FN = "unet_hairmask.h5"
CACHE_DIR  = Path(__file__).resolve().parent

# ---------------------- LOSS & METRICS --------------------------------
sm.set_framework("tf.keras")  # ensure Keras backend
sm.framework()               # init

loss = sm.losses.DiceLoss() + sm.losses.BinaryFocalLoss(gamma=2)
metrics = [sm.metrics.iou_score, sm.metrics.f1_score]

# ----------------------- AUGMENTATION ---------------------------------
HAIR_OVERLAYS = list((CACHE_DIR / "hair_overlays").glob("*.png"))  # optional

def hair_overlay(img, **kwargs):
    if not HAIR_OVERLAYS or np.random.rand() > 0.3:
        return img
    h, w = img.shape[:2]
    ov = cv2.imread(str(np.random.choice(HAIR_OVERLAYS)), cv2.IMREAD_UNCHANGED)
    ov = cv2.resize(ov, (w, h), cv2.INTER_AREA)
    alpha = ov[..., 3:] / 255.0
    img = (img * (1 - alpha) + ov[..., :3] * alpha).astype(np.uint8)
    return img
    h, w = img.shape[:2]
    overlay = cv2.imread(str(np.random.choice(HAIR_OVERLAYS)), cv2.IMREAD_UNCHANGED)
    overlay = cv2.resize(overlay, (w, h), cv2.INTER_AREA)
    alpha = overlay[..., 3:] / 255.0
    img = (img * (1 - alpha) + overlay[..., :3] * alpha).astype(np.uint8)
    return img

train_aug = A.Compose([
    A.HorizontalFlip(), A.VerticalFlip(), A.RandomRotate90(),
    A.ElasticTransform(alpha=80, sigma=8, approximate=True, p=0.5),
    A.RandomBrightnessContrast(0.2, 0.2, p=0.3),
    A.GaussNoise(p=0.3),
    A.Lambda(image=hair_overlay, p=0.5),
    A.Resize(IMG_SIZE, IMG_SIZE),
])
val_aug = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE)])

# ------------------------ DATA PIPELINE -------------------------------

def load_ids(img_dir: Path, msk_dir: Path):
    imgs = sorted(glob.glob(str(img_dir / "*")))
    masks = sorted(glob.glob(str(msk_dir / "*")))
    mask_dict = {PurePath(p).stem.replace("_mask", ""): p for p in masks}
    pairs = [(i, mask_dict[PurePath(i).stem]) for i in imgs if PurePath(i).stem in mask_dict]
    return pairs

def gen(pairs, aug):
    while True:
        np.random.shuffle(pairs)
        for img_path, msk_path in pairs:
            img = cv2.imread(img_path)[:, :, ::-1]
            msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
            data = aug(image=img, mask=msk)
            x = data["image"].astype("float32") / 255.0
            y = (data["mask"] > 127).astype("float32")[..., None]
            yield x, y

def dataset(pairs, aug, batch):
    return (tf.data.Dataset.from_generator(lambda: gen(pairs, aug),
                output_signature=(tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
                                   tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32)))
            .batch(batch)
            .prefetch(tf.data.AUTOTUNE))

# ------------------------ TRAIN FUNCTION ------------------------------

def train(img_dir: str, msk_dir: str, epochs: int = 40, encoder: str = ENCODER):
    img_dir, msk_dir = Path(img_dir), Path(msk_dir)
    pairs = load_ids(img_dir, msk_dir)
    if len(pairs) < 10:
        raise RuntimeError("Poucas imagens encontradas – verifique pastas.")
    split = int(0.9 * len(pairs))
    train_ds = dataset(pairs[:split], train_aug, BATCH)
    val_ds   = dataset(pairs[split:], val_aug,  BATCH)

    model = sm.Unet(encoder, encoder_weights="imagenet", classes=1, activation="sigmoid")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss, metrics=metrics)

    ckpt = ModelCheckpoint(CACHE_DIR / WEIGHTS_FN, save_best_only=True,
                           monitor="val_iou_score", mode="max")
    es   = EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss")
    rl   = ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)

    steps = len(pairs[:split]) // BATCH
    vsteps = max(1, len(pairs[split:]) // BATCH)

    model.fit(train_ds,
              validation_data=val_ds,
              epochs=epochs,
              steps_per_epoch=steps,
              validation_steps=vsteps,
              callbacks=[ckpt, es, rl])

# ------------------------ PREDICTION API ------------------------------
_NET = None
THRESH = 0.65

def _load_net():
    global _NET
    if _NET is None:
        path = CACHE_DIR / WEIGHTS_FN
        if not path.exists():
            raise FileNotFoundError(f"Peso não encontrado: {path}\nTreine com: python unet_hair_remover.py train --img_dir ... --msk_dir ...")
        _NET = load_model(path, compile=False)
    return _NET


def _preproc(bgr: np.ndarray):
    h, w = bgr.shape[:2]
    img = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
    return img[None], (h, w)

def _postpred(pred, shape):
    h, w = shape
    mask = (pred[0, ..., 0] > THRESH).astype("uint8") * 255
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

# --- filtros rá​​pidos --------------------------------------------------

def _filter_color(mask: np.ndarray, bgr: np.ndarray, thr=20):
    diff = bgr.max(axis=2) - bgr.min(axis=2)
    mask[diff < thr] = 0
    return mask

from skimage.morphology import skeletonize

def _filter_geom(mask: np.ndarray, min_len=40, max_w=15):
    if not mask.any():
        return mask
    sk = skeletonize(mask // 255).astype("uint8")
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    sk[dist < (max_w / 2)] = 0
    num, lbl, stats, _ = cv2.connectedComponentsWithStats(sk, 8)
    keep = np.zeros_like(sk)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_len:
            keep[lbl == i] = 255
    return cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))


def predict_hair_mask(bgr: np.ndarray) -> np.ndarray:
    net = _load_net()
    x, shape = _preproc(bgr)
    pred = net.predict(x, verbose=False)
    mask = _postpred(pred, shape)
    mask = _filter_color(mask, bgr)
    mask = _filter_geom(mask)
    return mask

# ----------------------------- CLI ------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train");
    tr.add_argument("--img_dir", required=True)
    tr.add_argument("--msk_dir", required=True)
    tr.add_argument("--epochs", type=int, default=40)
    tr.add_argument("--encoder", default=ENCODER)

    pr = sub.add_parser("predict")
    pr.add_argument("image")
    pr.add_argument("outfile", nargs="?")

    args = parser.parse_args()

    if args.cmd == "train":
        train(args.img_dir, args.msk_dir, args.epochs, args.encoder)

    elif args.cmd == "predict":
        bgr = cv2.imread(args.image)
        mask = predict_hair_mask(bgr)
        if args.outfile:
            cv2.imwrite(args.outfile, mask)
        else:
            cv2.imshow("mask", mask); cv2.waitKey(0)
