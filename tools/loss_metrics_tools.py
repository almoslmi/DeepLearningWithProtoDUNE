import tensorflow as tf
from keras import backend as K

def weighted_categorical_crossentropy(weights):
    """
    Weighted version of keras.objectives.categorical_crossentropy.
    Use this loss function with median frequency weights for class balance.
    """
    # Convert weights to a constant tensor
    weights = tf.constant(weights, dtype=tf.float32)

    def loss(y_true, y_pred):
       # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)

        # Clip to prevent NaN's and Inf's
        _epsilon = tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, _epsilon, 1.0 - _epsilon)

        # Do the loss calculation
        loss = y_true * tf.log(y_pred) * 1.0/weights
        return -tf.reduce_sum(loss, axis=-1)

    return loss

def focal_loss(alpha=0.25, gamma=2.0):
    """
    alpha: Scale the focal weight with alpha.
    gamma: Take the power of the focal weight with gamma.
    """
    def loss(y_true, y_pred):
        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)

        # Clip to prevent NaN's and Inf's
        _epsilon = tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, _epsilon, 1.0 - _epsilon)

        # Do the loss calculation
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        loss = alpha * tf.pow(1.0 - pt, gamma) * tf.log(pt)
        return -tf.reduce_sum(loss, axis=-1)

    return loss

def weighted_focal_loss(weights, gamma=2.0):
    """
    weights: Median frequency weights for class balance
    gamma: Take the power of the focal weight with gamma.
    """
    # Convert weights to a constant tensor
    weights = tf.constant(weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        alpha=1.0/weights

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)

        # Clip to prevent NaN's and Inf's
        _epsilon = tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, _epsilon, 1.0 - _epsilon)

        # Do the loss calculation
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        loss = alpha * tf.pow(1.0 - pt, gamma) * tf.log(pt)
        return -tf.reduce_sum(loss, axis=-1)

    return loss

# For Keras, custom metrics can be passed at the compilation step but
# the function would need to take (y_true, y_pred) as arguments and return a single tensor value.
# Note: seems like this implementation is not stable; it sometimes returns 0 in standalone tests
def three_classes_mean_iou(y_true, y_pred):
    """
    Calculate per-step mean Intersection-Over-Union (mIOU).
    Computes the IOU for each semantic class and then computes the average over classes.
    """
    num_classes = 3
    score, up_opt = tf.metrics.mean_iou(y_true, y_pred, num_classes)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score
