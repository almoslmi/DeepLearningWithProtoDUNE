import tensorflow as tf
from keras import backend as K

def weighted_categorical_crossentropy(weights):
    """
    Weighted version of keras.objectives.categorical_crossentropy.
    Use this loss function with median frequency coefficients weights for class balance.
    """
    # Convert weights to a variable instance (with Keras metadata included)
    weights = K.variable(weights)
    def loss(y_true, y_pred):
        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        # Do the loss calculation
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, axis=-1)

        return loss

    return loss


# For Keras, custom metrics can be passed at the compilation step but
# the function would need to take (y_true, y_pred) as arguments and return a single tensor value.
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

def four_classes_mean_iou(y_true, y_pred):
    """
    Calculate per-step mean Intersection-Over-Union (mIOU).
    Computes the IOU for each semantic class and then computes the average over classes.
    """
    num_classes = 4
    score, up_opt = tf.metrics.mean_iou(y_true, y_pred, num_classes)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score
