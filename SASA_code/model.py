import tensorflow as tf
import tensorflow.contrib.layers as tf_contrib_layers
from tensorflow.contrib.rnn import LSTMCell
from metrics import mmd_loss

class SASA(object):

    def __init__(self, params_config):

        # [batch_size, input_dim, segments_num, window_size, 1]
        self.x = tf.placeholder(
            dtype=tf.float32, shape=[None, params_config.input_dim, params_config.segments_num, params_config.window_size,
                                     1], name='input')
        if params_config.classification:
            self.y = tf.placeholder(dtype=tf.int32, shape=[None], name="label")
        else:
            self.y = tf.placeholder(dtype=tf.float32, shape=[None], name="label")

        self.seq_length = tf.placeholder(dtype=tf.int32, shape=[None], name="seq_len")

        self.max_lag = params_config.window_size

        self.learning_rate = tf.constant(params_config.learning_rate, dtype=tf.float32)
        self.training = tf.placeholder(dtype=tf.bool, shape=[], name="is_train")
        self.lossfn = params_config.loss_fn

        self.labeled_size = tf.placeholder(dtype=tf.int32, shape=(), name='labeled_size')
        self.coeff = tf.constant(params_config.coeff, dtype=tf.float32)

        self._build_graph(classification=params_config.classification, max_length = params_config.window_size,
                          segments_num= params_config.segments_num, input_dim=params_config.input_dim,
                          class_num=params_config.class_num, h_dim=params_config.h_dim, dense_dim=params_config.dense_dim,
                          drop_prob=params_config.drop_prob)

    def _build_graph(self, classification, max_length, segments_num, input_dim, class_num,
                     h_dim, dense_dim, drop_prob):


        def self_attention_fn(Q,K,scaled=True,sparse=True):
            '''
            :param Q: [batch_size, segments_num, hidden_dim]
            :param K: [batch_size, segments_num, hidden_dim]
            :return: [batch_size, segments_num, segments_num]
            '''
            attention_weight = tf.matmul(Q, K, transpose_b=True)

            attention_weight = tf.reduce_mean(attention_weight, axis=1,
                                                  keepdims=True)  # attention_weight[batch_size, 1, segments_num]
            # sacled
            if scaled:
                d_k = tf.cast(tf.shape(K)[-1], dtype=tf.float32)
                attention_weight = tf.divide(attention_weight, tf.sqrt(d_k))  # [batch_size, segments_num, segments_num]

            if sparse:
                # The implementation of sparsemax requires the tensor of rank 2
                # The sparsemax algorithm is presented in https://arxiv.org/abs/1602.02068.
                attention_weight_sparse = tf.contrib.sparsemax.sparsemax(tf.reshape(attention_weight, shape=[-1, tf.shape(attention_weight)[-1]]))  #[batch_size*window_n um, segments_num]
                attention_weight = tf.reshape(attention_weight_sparse, shape=[-1, tf.shape(attention_weight)[1], tf.shape(attention_weight)[2]])
            else:
                attention_weight = tf.nn.softmax(attention_weight, axis=-1)

            return attention_weight
        def attention_fn(Q,K,scaled=False,sparse=True):

            # cosine
            attention_weight = tf.matmul(tf.nn.l2_normalize(Q, dim=-1), tf.nn.l2_normalize(K, dim=-1), transpose_b=True)

            if scaled:
                d_k = tf.cast(tf.shape(K)[-1], dtype=tf.float32)
                attention_weight = tf.divide(attention_weight, tf.sqrt(d_k))

            if sparse:
                attention_weight_sparse = tf.contrib.sparsemax.sparsemax(tf.reshape(attention_weight,shape=[-1,tf.shape(attention_weight)[-1]]))
                attention_weight = tf.reshape(attention_weight_sparse,shape=[-1, tf.shape(attention_weight)[1],tf.shape(attention_weight)[2]])
            else:
                attention_weight = tf.nn.softmax(attention_weight, axis=-1)
            return attention_weight

        final_feature = []
        hidden_state_list, segments_representation = [], []
        self.domain_loss_alpha = []
        self.domain_loss_beta = []
        for i in range(0,input_dim):

            with tf.variable_scope("lstm_%s"% str(i+1)):
                # [batch_size, input_dim, segments_num, window_size, 1]
                univariate_x = tf.reshape(self.x[:, i, :, :, :], shape=[-1, max_length, 1])

                cell = LSTMCell(num_units=h_dim, use_peepholes=True)

                # MIMIC-III
                # cell = LSTMCell(num_units=h_dim, use_peepholes=True,
                #                      initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0, dtype=tf.float32))

                outputs, final_state = tf.nn.dynamic_rnn(cell, univariate_x, dtype=tf.float32, sequence_length=self.seq_length)

                final_hidden_state = tf.reshape(final_state[1], shape=[-1, segments_num, h_dim])

                hidden_state_list.append(final_hidden_state)

            with tf.variable_scope("self-attention", reuse=tf.AUTO_REUSE):
                Q = tf.layers.dense(final_hidden_state, units=h_dim,
                                    kernel_regularizer=tf_contrib_layers.l2_regularizer(scale=0.5),
                                    )
                K = tf.layers.dense(final_hidden_state, units=h_dim,
                                    kernel_regularizer=tf_contrib_layers.l2_regularizer(scale=0.5),
                                    )
                V = tf.layers.dense(final_hidden_state, units=h_dim,
                                    kernel_regularizer=tf_contrib_layers.l2_regularizer(scale=0.5),
                                    )

                #  sparsemax
                attention_weight = self_attention_fn(Q, K, sparse=True)

                Z_i = tf.matmul(attention_weight, V)

                segments_representation.append(Z_i)

                # loss alpha
                src_structure, tgt_structure = tf.split(tf.squeeze(attention_weight, axis=1), 2, axis=0)
                domain_loss_intra = mmd_loss(src_structure, tgt_structure, self.coeff)
                self.domain_loss_alpha.append(domain_loss_intra)

        # attention for inter-feature structure
        for i in range(0, input_dim):
            with tf.variable_scope('attention_%s' % str(i+1)):

                Z_i = segments_representation[i]

                other_hidden_state = [hidden_state_list[j] for j in range(0, input_dim) if j!=i]
                other_hidden_state = tf.concat(other_hidden_state, axis=1)

                attention_weight = attention_fn(Z_i, other_hidden_state, sparse=True)

                U_i = tf.matmul(attention_weight, other_hidden_state)

                Hi = tf.squeeze(tf.concat([Z_i, U_i], axis=-1), axis=1)

                final_feature.append(Hi)

                src_structure, tgt_structure = tf.split(tf.squeeze(attention_weight, axis=1), 2, axis=0)
                domain_loss_inter = mmd_loss(src_structure, tgt_structure, self.coeff)
                self.domain_loss_beta.append(domain_loss_inter)


        with tf.variable_scope("label_predictor"):

            # mask target label
            final_feature = tf.reshape(tf.slice(tf.concat(final_feature, axis=-1), begin=[0, 0], size=[self.labeled_size, -1]),
                                        shape=[self.labeled_size, input_dim * 2 * h_dim])
            self.labeled_y = tf.slice(self.y, begin=[0], size=[self.labeled_size])

            self.final_feature = final_feature

            label_hidden_1 = tf.layers.dense(inputs=final_feature, units=dense_dim,
                                             activation=tf.nn.relu,
                                             kernel_initializer=tf_contrib_layers.xavier_initializer(),
                                             kernel_regularizer=tf_contrib_layers.l2_regularizer(scale=0.5),
                                             name="label_hidden_layer_1")

            label_hidden_dropout_1 = tf.layers.dropout(label_hidden_1, drop_prob, training=self.training,
                                                       name="label_hidden_dropout_layer_1")

            self.y_pred = tf.layers.dense(inputs=label_hidden_dropout_1, units=class_num,
                                          activation=None,
                                          kernel_initializer=tf_contrib_layers.xavier_initializer(),
                                          kernel_regularizer=tf_contrib_layers.l2_regularizer(scale=0.5),
                                          name="label_hidden_layer_2")

            self.label_loss = self.lossfn(y_true=self.labeled_y, y_pred=self.y_pred)

        if classification:
            self.logits_pred = tf.nn.softmax(self.y_pred)
            self.onehot_pred = tf.one_hot(tf.argmax(self.y_pred, 1), depth=class_num)
            self.onehot_true = tf.one_hot(self.labeled_y, depth=class_num)


        self.domain_loss = tf.add_n(self.domain_loss_alpha) + tf.add_n(self.domain_loss_beta)

        self.total_loss = self.label_loss + self.domain_loss

        self.opt = tf.train.AdamOptimizer(self.learning_rate)

        self.training_op = self.opt.minimize(self.total_loss)

        self.saver = tf.train.Saver(max_to_keep=1)
