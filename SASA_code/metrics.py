import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf


def rmse_loss(y_true, y_pred):

    with tf.name_scope('RMSE'):
        y_true = tf.expand_dims(y_true,axis=-1)
        se = tf.square(tf.subtract(y_true, y_pred))
        return tf.sqrt(tf.reduce_mean(se))


def rmse(y_true, y_pred):

    y_true = np.expand_dims(y_true, axis=-1)
    se = np.square(np.subtract(y_true, y_pred))
    mse = np.mean(se)
    return np.sqrt(mse)


def cross_entropy_loss(y_true, y_pred):
    '''
    :param y_true: [batch_size]
    :param y_pred: [batch_size, num_classes]
    :return:
    '''
    cross_entropy = \
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred,name="label_logits_cross_entropy")

    loss = tf.reduce_mean(cross_entropy, name="label_loss")
    return loss


def cal_roc_auc_score(logist_pred, y_true, multi_class=False):

    if not multi_class:
        pos_probability_list = [list(lp)[1] for lp in logist_pred]
        true_label_list = np.argmax(y_true, axis=1)
        roc_auc = roc_auc_score(y_true=true_label_list, y_score=pos_probability_list)
    else:
        ## multiclass

        # TODO:
        raise Exception('TODO Multiclass...')
    return roc_auc


def mmd_loss(source_samples, target_samples, weight):
    '''
    mmd loss about linear kernel.
    '''
    delta = source_samples - target_samples
    loss_value = tf.reduce_mean(tf.matmul(delta, tf.transpose(delta, perm=[1, 0])))
    loss_value = tf.maximum(1e-6, loss_value) * weight
    return loss_value