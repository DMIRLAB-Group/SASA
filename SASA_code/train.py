import os
import math
import logging
import numpy as np
import argparse
import tensorflow as tf
from tabulate import tabulate
from params_config import Config_Params_SASA
from data_utils import get_dataset_size, data_generator
from model import SASA

parser = argparse.ArgumentParser(description='train')
parser.add_argument('-cuda_device', type=str, default='0', help='which gpu to use ')
# # training params
parser.add_argument('-lr', type=float, default=0.0005, help='initial learning rate [default: 0.0005]')
parser.add_argument('-drop_prob', type=float, default=0.3, help='the probability for dropout [default: 0.3]')
parser.add_argument('-rep_idx', type=int, default=1, help='An index used to mark repeated experiments.'
                                                                 'save_dir like model_save/Air/SASA/...._rep_idx ')
parser.add_argument('-collect_score', type=bool, default=False, help='whether collect the best score to file')


parser.add_argument('-model', type=str, default='SASA', help='which model')
parser.add_argument('-dataset', type=str, default='Air', help='which dataset')
parser.add_argument('-source_to_target', type=str, default='S2T', help='the source to the target')
parser.add_argument('-src_train_path', type=str, default='Beijing_train.csv', help='source domain train path')
parser.add_argument('-src_test_path', type=str, default='Beijing_test.csv', help='source domain test path')
parser.add_argument('-tgt_train_path', type=str, default='Tianjin_train.csv', help='target domain train path')
parser.add_argument('-tgt_test_path', type=str, default='Tianjin_test.csv', help='target domain test path')

# parser.add_argument('-model', type=str, default='SASA', help='which model')
# parser.add_argument('-dataset', type=str, default='MIMIC-III', help='which dataset')
# parser.add_argument('-source_to_target', type=str, default='3to2', help='the source to the target')
# parser.add_argument('-src_train_path', type=str, default='3_train.npy', help='source domain train path')
# parser.add_argument('-src_test_path', type=str, default='3_test.npy', help='source domain test path')
# parser.add_argument('-tgt_train_path', type=str, default='2_train.npy', help='target domain train path')
# parser.add_argument('-tgt_test_path', type=str, default='2_test.npy', help='target domain test path')

# parser.add_argument('-model', type=str, default='SASA', help='which model ')
# parser.add_argument('-dataset', type=str, default='Boiler', help='which dataset')
# parser.add_argument('-source_to_target', type=str, default='1to2', help='the source to the target')
# parser.add_argument('-src_train_path', type=str, default='1/train.csv', help='source domain train path')
# parser.add_argument('-src_test_path', type=str, default='1/test.csv', help='source domain test path')
# parser.add_argument('-tgt_train_path', type=str, default='Boiler_#02.csv', help='target domain train path')
# parser.add_argument('-tgt_test_path', type=str, default='Boiler_#02.csv', help='target domain test path')



args = parser.parse_args()


def init_logger(log_file):
    logger = logging.getLogger('Running-Logger')
    logger.setLevel(logging.DEBUG)
    log_file_handler = logging.FileHandler(filename=os.path.join(log_file))
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    log_file_handler.setFormatter(formatter)
    logger.addHandler(log_file_handler)
    return logger


def save_best_score(best_score, config):
    '''
    different training parameters best results save to one file
    :param best_score:
    :param config:
    :return:
    '''
    file_name = '%s_%s.txt' % (config.model, config.source_to_target)  #'SASA_3to8.txt'
    content_name = config.log_save_file.split('.txt')[0]

    # save
    with open(os.path.join(config.log_save_path, file_name), 'a') as file:
        file.write(str(best_score) + ' ' + content_name + '\n')
        file.close()

def evaluate_classification(session):
    total_tgt_test_label_loss = 0.0
    tgt_test_epoch = int(math.ceil(tgt_test_set_size / float(config.batch_size)))

    tgt_test_logits_pred_list = list()
    tgt_test_y_pred_list = list()
    tgt_test_y_true_list = list()
    for _ in range(tgt_test_epoch):
        test_batch_tgt_x, test_batch_tgt_y, test_batch_tgt_l = tgt_test_generator.__next__()

        tgt_test_label_loss, \
        tgt_test_logits_pred, tgt_test_label_pred, tgt_test_label_true, ff = \
            session.run([model.label_loss, model.logits_pred,
                         model.onehot_pred, model.onehot_true, model.final_feature],
                        feed_dict={model.x: test_batch_tgt_x,
                                   model.y: test_batch_tgt_y,
                                   model.seq_length: test_batch_tgt_l,
                                   model.labeled_size: test_batch_tgt_y.shape[0],
                                   model.training: False})

        total_tgt_test_label_loss += tgt_test_label_loss
        tgt_test_logits_pred_list.extend(tgt_test_logits_pred)
        tgt_test_y_pred_list.extend(tgt_test_label_pred)
        tgt_test_y_true_list.extend(tgt_test_label_true)

    # target test
    score = \
        config.eval_fn(logist_pred=tgt_test_logits_pred_list, y_true=tgt_test_y_true_list)
    mean_tgt_test_label_loss = total_tgt_test_label_loss / tgt_test_epoch
    return mean_tgt_test_label_loss, score

def evaluate_regression(session):
    total_tgt_test_label_loss = 0.0
    tgt_test_epoch = int(math.ceil(tgt_test_set_size / float(config.batch_size)))

    # tgt_test_logits_pred_list = list()
    tgt_test_y_pred_list = list()
    tgt_test_y_true_list = list()
    for _ in range(tgt_test_epoch):
        test_batch_tgt_x, test_batch_tgt_y, test_batch_tgt_l = tgt_test_generator.__next__()

        tgt_test_label_loss, \
        tgt_test_label_pred, tgt_test_label_true = \
            session.run([model.label_loss, model.y_pred,
                         model.labeled_y],
                        feed_dict={model.x: test_batch_tgt_x,
                                   model.y: test_batch_tgt_y,
                                   model.seq_length: test_batch_tgt_l,
                                   model.labeled_size: test_batch_tgt_y.shape[0],
                                   model.training: False})

        total_tgt_test_label_loss += tgt_test_label_loss
        tgt_test_y_pred_list.extend(tgt_test_label_pred)
        tgt_test_y_true_list.extend(tgt_test_label_true)

    # target test
    score = \
        config.eval_fn(y_pred=tgt_test_y_pred_list, y_true=tgt_test_y_true_list)
    mean_tgt_test_label_loss = total_tgt_test_label_loss / tgt_test_epoch
    return mean_tgt_test_label_loss, score

def train_classification(config):

    print('start training...')
    with tf.Session(config=config_proto) as session:

        session.run(tf.global_variables_initializer())

        global_step = 0

        total_train_label_loss = 0.0
        total_train_domain_loss = 0.0

        best_score = 0.0
        best_step = 0
        while global_step < config.training_steps:

            src_train_batch_x, src_train_batch_y, src_train_batch_l = src_train_generator.__next__()
            # src_train_batch_y = np.reshape(src_train_batch_y, [-1, 1])

            tgt_train_batch_x, tgt_train_batch_y, tgt_train_batch_l = tgt_train_generator.__next__()
            # tgt_train_batch_y = np.reshape(tgt_train_batch_y, [-1, 1])

            if src_train_batch_y.shape[0] != tgt_train_batch_y.shape[0]: #
                continue

            train_batch_x = np.vstack([src_train_batch_x, tgt_train_batch_x])
            train_batch_y = np.append(src_train_batch_y, tgt_train_batch_y)
            train_batch_l = np.append(src_train_batch_l, tgt_train_batch_l)

            _, train_label_loss, train_domain_loss, pred\
                = session.run([model.training_op, model.label_loss, model.domain_loss, model.logits_pred],
                              feed_dict={model.x: train_batch_x, model.y: train_batch_y, model.seq_length: train_batch_l,
                                         model.labeled_size: src_train_batch_y.shape[0],
                                         model.training: True})
            total_train_label_loss += train_label_loss
            total_train_domain_loss += train_domain_loss

            ## test test test
            if global_step % config.test_per_step == 0 and global_step != 0:

                mean_tgt_test_label_loss, auc_score = evaluate_classification(session)


                # source train
                mean_train_label_loss = total_train_label_loss / config.test_per_step
                mean_train_domain_loss = total_train_domain_loss / config.test_per_step

                header = ["Steps=%d" % (global_step ), "label_loss", "domain_loss", "AUC", ]
                table = \
                    [["train", mean_train_label_loss, mean_train_domain_loss,""],

                     ["tgt_test", mean_tgt_test_label_loss, "", auc_score ]]

                print(tabulate(table, header, tablefmt="grid"), "\n")
                logger.info(tabulate(table, header, tablefmt="grid"))

                if best_score < auc_score:
                    best_score = auc_score
                    best_step = global_step
                    model.saver.save(session, save_path=os.path.join(config.model_save_path, 'best')) # only save the best model
                elif global_step - best_step >= config.early_stop:
                    print('early stop by {} steps.'.format(config.early_stop))
                    logger.info('early stop ')
                    break

                print('best score:',best_score,'best step:',best_step, config.model_save_path)

                logger.info((best_score, best_step, config.model_save_path))

                total_train_label_loss = 0.0
                total_train_domain_loss = 0.0

            global_step += 1
    return best_score

def train_regression(config):

    print('start training...')
    with tf.Session(config=config_proto) as session:

        session.run(tf.global_variables_initializer())

        global_step = 0

        total_train_label_loss = 0.0
        total_train_domain_loss = 0.0

        best_score = 1000000.0
        best_step = 0
        last_tgt_test_y_pred_list = []
        while global_step < config.training_steps:

            src_train_batch_x, src_train_batch_y, src_train_batch_l = src_train_generator.__next__()
            # src_train_batch_y = np.reshape(src_train_batch_y, [-1, 1])

            tgt_train_batch_x, tgt_train_batch_y, tgt_train_batch_l = tgt_train_generator.__next__()
            # tgt_train_batch_y = np.reshape(tgt_train_batch_y, [-1, 1])

            if src_train_batch_y.shape[0] != tgt_train_batch_y.shape[0]: #
                continue

            train_batch_x = np.vstack([src_train_batch_x, tgt_train_batch_x])
            train_batch_y = np.append(src_train_batch_y, tgt_train_batch_y)
            train_batch_l = np.append(src_train_batch_l, tgt_train_batch_l)

            _, train_label_loss, train_domain_loss \
                = session.run([model.training_op, model.label_loss, model.domain_loss],
                              feed_dict={model.x: train_batch_x, model.y: train_batch_y, model.seq_length: train_batch_l,
                                         model.labeled_size: src_train_batch_y.shape[0],
                                         model.training: True})
            total_train_label_loss += train_label_loss
            total_train_domain_loss += train_domain_loss

            ## test test test
            if global_step % config.test_per_step == 0 and global_step != 0:

                mean_tgt_test_label_loss, score = evaluate_regression(session)


                # source train
                mean_train_label_loss = total_train_label_loss / config.test_per_step
                mean_train_domain_loss = total_train_domain_loss / config.test_per_step

                header = ["Steps=%d" % (global_step ), "label_loss", "domain_loss", "RMSE", ]
                table = \
                    [["train", mean_train_label_loss, mean_train_domain_loss,""],

                     ["tgt_test", mean_tgt_test_label_loss, "", score ]]

                print(tabulate(table, header, tablefmt="grid"), "\n")
                logger.info(tabulate(table, header, tablefmt="grid"))

                if best_score > score:
                    best_score = score
                    best_step = global_step
                    model.saver.save(session, save_path=os.path.join(config.model_save_path, 'best')) # only save the best model
                elif global_step - best_step >= config.early_stop:
                    print('early stop by {} steps.'.format(config.early_stop))
                    logger.info('early stop ')
                    break

                print('best score:',best_score,'best step:',best_step, config.model_save_path)

                logger.info((best_score, best_step, config.model_save_path))

                total_train_label_loss = 0.0
                total_train_domain_loss = 0.0

            global_step += 1
    return best_score




if __name__=='__main__':
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    config = Config_Params_SASA(args)
    logger = init_logger(config.log_save_file)

    config_proto = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config_proto.gpu_options.allow_growth = True


    ## data preparing..
    src_train_generator = data_generator(data_path=config.src_train_path, segments_length=config.segments_length,
                                         window_size=config.window_size,
                                         batch_size=config.batch_size // 2, dataset=config.dataset, is_shuffle=True, )
    tgt_train_generator = data_generator(data_path=config.tgt_train_path, segments_length=config.segments_length,
                                         window_size=config.window_size,
                                         batch_size=config.batch_size // 2, dataset=config.dataset, is_shuffle=True, )
    tgt_test_generator = data_generator(data_path=config.tgt_test_path, segments_length=config.segments_length,
                                        window_size=config.window_size,
                                        batch_size=config.batch_size, dataset=config.dataset, is_shuffle=False, )

    tgt_test_set_size = get_dataset_size(config.tgt_test_path, config.dataset, config.window_size)

    model = SASA(config)

    if config.classification:
        best_score = train_classification(config)
    else:
        best_score = train_regression(config)

    if args.collect_score:
        save_best_score(best_score, config)
