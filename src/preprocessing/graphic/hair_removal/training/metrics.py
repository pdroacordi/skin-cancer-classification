import tensorflow as tf
import tensorflow.keras.backend as K

class DiceLoss(tf.keras.losses.Loss):
    """Enhanced Dice loss with better numerical stability and class balancing"""

    def __init__(self, smooth=1.0, gamma=2.0, **kwargs):
        super().__init__(name="enhanced_dice_loss", **kwargs)
        self.smooth = smooth
        self.gamma = gamma  # For focal component

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Flatten the tensors
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])

        # Calculate intersection and union
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)

        # Dice coefficient with improved stability
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Add focal component for hard examples
        focal_weight = tf.pow(1 - dice, self.gamma)

        return focal_weight * (1 - dice)


class TverskyLoss(tf.keras.losses.Loss):
    """Tversky loss for better handling of class imbalance"""

    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6, **kwargs):
        super().__init__(name="tversky_loss", **kwargs)
        self.alpha = alpha  # Weight for false negatives
        self.beta = beta  # Weight for false positives
        self.smooth = smooth

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Flatten
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])

        # Calculate true positives, false negatives, and false positives
        true_pos = tf.reduce_sum(y_true_f * y_pred_f)
        false_neg = tf.reduce_sum(y_true_f * (1 - y_pred_f))
        false_pos = tf.reduce_sum((1 - y_true_f) * y_pred_f)

        tversky = (true_pos + self.smooth) / (true_pos + self.alpha * false_neg +
                                              self.beta * false_pos + self.smooth)

        return 1 - tversky


class HybridLoss(tf.keras.losses.Loss):
    """Combination of BCE, Dice and Tversky, with dtype unification."""
    def __init__(self,
                 bce_weight=0.3,
                 dice_weight=0.4,
                 tversky_weight=0.3,
                 smooth=1e-6,
                 **kwargs):
        super().__init__(name="hybrid_loss", **kwargs)
        self.bce_weight     = bce_weight
        self.dice_weight    = dice_weight
        self.tversky_weight = tversky_weight
        self.smooth         = smooth

        self.bce     = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        # You already have DiceLoss and TverskyLoss in this file
        self.dice    = DiceLoss(smooth=self.smooth)
        self.tversky = TverskyLoss(alpha=0.7, beta=0.3, smooth=self.smooth)

    def call(self, y_true, y_pred):
        # make sure everything starts as float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # compute each component
        bce_loss     = self.bce(y_true, y_pred)
        dice_loss    = self.dice(y_true, y_pred)
        tversky_loss = self.tversky(y_true, y_pred)

        # cast them explicitly to float32
        bce_loss     = tf.cast(bce_loss, tf.float32)
        dice_loss    = tf.cast(dice_loss, tf.float32)
        tversky_loss = tf.cast(tversky_loss, tf.float32)

        # now summation is safe
        total = (
            self.bce_weight     * bce_loss +
            self.dice_weight    * dice_loss +
            self.tversky_weight * tversky_loss
        )

        return total


# Enhanced metrics
class IoUMetric(tf.keras.metrics.Metric):
    """Intersection over Union metric"""

    def __init__(self, threshold=0.5, name='iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.intersection = self.add_weight(name='intersection', initializer='zeros')
        self.union = self.add_weight(name='union', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)

        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection

        self.intersection.assign_add(intersection)
        self.union.assign_add(union)

    def result(self):
        return self.intersection / (self.union + K.epsilon())

    def reset_state(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)


class F1Score(tf.keras.metrics.Metric):
    """F1 Score metric for binary segmentation"""

    def __init__(self, threshold=0.5, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.precision_metric = tf.keras.metrics.Precision(thresholds=threshold)
        self.recall_metric = tf.keras.metrics.Recall(thresholds=threshold)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision_metric.update_state(y_true, y_pred, sample_weight)
        self.recall_metric.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision_metric.result()
        recall = self.recall_metric.result()
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    def reset_state(self):
        self.precision_metric.reset_state()
        self.recall_metric.reset_state()

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