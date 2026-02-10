import configparser


class Config:
    def __init__(self, config_path, mode='train'):
        config = configparser.ConfigParser()
        config.read(config_path)

        if mode == 'train':
            train_config = config['TRAIN']
            self.batch_size = int(train_config['BATCH_SIZE'])
            self.max_epochs = int(train_config['MAX_EPOCHS'])
            self.log_interval = int(train_config['LOG_INTERVAL'])
            self.num_samples = int(train_config['NUM_SAMPLES'])
            self.drop_p = float(train_config['DROP_P'])

            opt_config = config['OPTIMIZER']
            self.init_lr = float(opt_config['INIT_LR'])
            self.adam_eps = float(opt_config['ADAM_EPS'])
            self.adam_weight_decay = float(opt_config['ADAM_WEIGHT_DECAY'])

            gcn_config = config['GCN']
            self.hidden_size = int(gcn_config['HIDDEN_SIZE'])
            self.num_stages = int(gcn_config['NUM_STAGES'])

        elif mode == 'test':
            # Load both TRAIN and TEST sections for test mode
            train_config = config['TRAIN']
            self.num_samples = int(train_config['NUM_SAMPLES'])
            self.drop_p = float(train_config['DROP_P'])

            gcn_config = config['GCN']
            self.hidden_size = int(gcn_config['HIDDEN_SIZE'])
            self.num_stages = int(gcn_config['NUM_STAGES'])

            test_config = config['TEST']
            self.test_split = test_config.get('TEST_SPLIT')
            self.resume = test_config.get('RESUME')
            self.test_batch_size = int(test_config.get('TEST_BATCH_SIZE', 1))
            self.num_classes = int(test_config.get('NUM_CLASSES', 100))
            self.keypoints_path = test_config.get('KEYPOINTS_PATH', '')

        else:
            raise ValueError(f"Invalid mode: {mode}. Choose 'train' or 'test'.")

    def __str__(self):
        attrs = vars(self)
        return ', '.join(f'{k}={v}' for k, v in attrs.items())


if __name__ == '__main__':
    # example usage
    config_path = '/home/dxli/workspace/nslt/code/VGG-GRU/configs/test.ini'
    cfg = Config(config_path, mode='test')
    print(cfg)
