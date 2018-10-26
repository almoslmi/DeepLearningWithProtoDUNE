import tensorflow as tf
from keras import backend as K

def weighted_loss(num_classes, coefficients):
    """
    Gets weighted categorical cross entropy.
    Use this loss function with median frequency coefficients weights for class balance.
    """
    coefficients = tf.constant(coefficients)
    num_classes = tf.constant(num_classes)

    def loss(labels, logits):
        with tf.name_scope('loss_1'):
            logits = tf.reshape(logits, (-1, num_classes))
            epsilon = tf.constant(value=1e-10)

            logits = logits + epsilon
            # consturct one-hot label array
            labels = tf.to_float(tf.reshape(labels, (-1, num_classes)))
            softmax = tf.nn.softmax(logits)

            cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax + epsilon),
                                                       coefficients), reduction_indices=[1])
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

            tf.add_to_collection('losses', cross_entropy_mean)
            loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
            #loss = cross_entropy_mean
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
