from modules.loss import loss_function, label_error_rate
import tensorflow as tf


def valid_step_teacher_forcing(inp, tar, model, loss_tb, ler_tb, acc_tb):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    predictions, _, attention_weights = model.model(inp, tar_inp, False)

    loss = loss_function(tar_real, predictions)

    tar_weight = tf.cast(tf.logical_not(tf.math.equal(tar_real, 0)), tf.int32)
    tar_len = tf.reduce_sum(tar_weight, axis=-1)
    ler = label_error_rate(tar[:, 1:], predictions, tar_len)

    loss_tb(loss)
    ler_tb(ler)
    acc_tb(tar_real, predictions, sample_weight=tar_weight)

    return predictions.numpy()[0], tf.stack(attention_weights, axis=-1).numpy()[0]
