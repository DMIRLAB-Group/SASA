import os
import metrics as metrics

class Config_Params(object):
    def __init__(self, args):

        self.model = args.model
        self.model_save_base_path = 'model_save'
        self.log_save_base_path = 'log_save'

        # data info
        self.dataset = args.dataset
        self.source_to_target = args.source_to_target

        # learning params
        self.drop_prob = args.drop_prob
        self.learning_rate = args.lr

        # training config
        self.batch_size = 128
        self.test_per_step = 50
        self.early_stop = 5000
        self.training_steps = 50000
        print(self.dataset)
        if self.dataset == 'Air':

            # data path
            data_base_path = '../datasets/Air'
            self.src_train_path, self.src_test_path = \
                os.path.join(data_base_path, args.src_train_path), os.path.join(data_base_path, args.src_test_path)
            self.tgt_train_path, self.tgt_test_path = \
                os.path.join(data_base_path, args.tgt_train_path), os.path.join(data_base_path, args.tgt_test_path)

            # data info
            self.classification = False
            self.input_dim = 11
            self.class_num = 1 #the number of classes
            self.window_size = 6  # the length of sample

            self.loss_fn, self.eval_fn = metrics.rmse_loss, metrics.rmse
            self.coeff = 1000

        elif self.dataset == 'MIMIC-III':
            # data path
            data_base_path = '../datasets/MIMIC-III'
            self.src_train_path, self.src_test_path = \
                os.path.join(data_base_path, args.src_train_path), os.path.join(data_base_path, args.src_test_path)
            self.tgt_train_path, self.tgt_test_path = \
                os.path.join(data_base_path, args.tgt_train_path), os.path.join(data_base_path, args.tgt_test_path)

            # data info
            self.classification = True
            self.input_dim = 12
            self.class_num = 2  #the number of classes
            self.window_size = 24

            self.loss_fn, self.eval_fn = metrics.cross_entropy_loss, metrics.cal_roc_auc_score
            self.coeff = 5

        elif self.dataset=='Boiler':
            # data path
            data_base_path = '../datasets/Boiler'
            self.src_train_path, self.src_test_path = \
                os.path.join(data_base_path, args.src_train_path), os.path.join(data_base_path, args.src_test_path)
            self.tgt_train_path, self.tgt_test_path = \
                os.path.join(data_base_path, args.tgt_train_path), os.path.join(data_base_path, args.tgt_test_path)

            # data info
            self.classification = True
            self.input_dim = 20
            self.class_num = 2  #the number of classes
            self.window_size = 6

            self.loss_fn, self.eval_fn = metrics.cross_entropy_loss, metrics.cal_roc_auc_score
            self.coeff = 5

        else:
            raise Exception('unknown dataset!')

    def build_save_path(self, args):
        # save path
        # 'model_save/Air/B2T_...'
        self.model_save_path = os.path.join(self.model_save_base_path, self.dataset, self.model,
                                            f'{self.source_to_target}_lr{self.learning_rate}_'
                                            f'dp{self.drop_prob}_rept{args.rep_idx}')
        self.log_save_path = os.path.join(self.log_save_base_path, self.dataset, self.model)
        self.log_save_file = os.path.join(self.log_save_path,
                                          f'{self.source_to_target}_lr{self.learning_rate}_'
                                          f'dp{self.drop_prob}_rept{args.rep_idx}.txt')

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        if not os.path.exists(self.log_save_path):
            os.makedirs(self.log_save_path)


class Config_Params_SASA(Config_Params):
    def __init__(self, args):
        super(Config_Params_SASA, self).__init__(args)

        if self.dataset == 'Air':
            self.segments_length = [1, 2, 3, 4, 5, 6]  # the length of each segments
            self.segments_num = len(self.segments_length)

            self.h_dim = 15
            self.dense_dim = 100
        elif self.dataset=='MIMIC-III':
            self.segments_length = [3, 6, 9, 12, 15, 18, 21, 24] #coarse-grained segements
            self.segments_num = len(self.segments_length)

            self.h_dim = 20
            self.dense_dim = 100
        elif self.dataset == 'Boiler':
            self.segments_length = [1, 2, 3, 4, 5, 6]  # the length of each segments
            self.segments_num = len(self.segments_length)

            self.h_dim = 15
            self.dense_dim = 100
        else:
            raise Exception('unknown dataset!')

        self.build_save_path(args)