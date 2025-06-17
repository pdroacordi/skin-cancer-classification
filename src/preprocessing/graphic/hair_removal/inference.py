import numpy as np, cv2
from .config import HairRemovalConfig
from .model import ChimeraNet

class HairRemover:
    """Wrap de inferência + TTA e inpainting."""
    def __init__(self, cfg: HairRemovalConfig):
        self.cfg = cfg
        self.model = ChimeraNet(cfg.img_size)
        if cfg.model_weights:
            self.model.load_weights(str(cfg.model_weights))

    # ---------- TTA ----------
    def _predict_tta(self, img):
        aug = [img,
               np.flip(img,1), np.flip(img,0),
               np.rot90(img,1), np.rot90(img,-1)]
        outs = []
        for i, im in enumerate(aug):
            p = self.model.predict(im[None], verbose=0)[0,:,:,0]
            if i==1: p = np.flip(p,1)
            elif i==2: p = np.flip(p,0)
            elif i==3: p = np.rot90(p,-1)
            elif i==4: p = np.rot90(p,1)
            outs.append(p)
        return np.mean(outs, axis=0)

    # ---------- API ----------
    def remove_hair(self, bgr_np):
        h,w,_ = bgr_np.shape
        img = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.cfg.img_size, self.cfg.img_size))
        img = img.astype(np.float32)/255.0

        mask = self._predict_tta(img) if self.cfg.tta else \
               self.model.predict(img[None])[0,:,:,0]
        mask = (mask > .5).astype(np.uint8)*255
        mask = cv2.resize(mask, (w, h))

        clean = cv2.inpaint(bgr_np, mask, 5, cv2.INPAINT_TELEA)
        return clean, mask
