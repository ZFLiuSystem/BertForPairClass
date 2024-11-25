import argparse


class Args:
    @staticmethod
    def parser():
        parser = argparse.ArgumentParser()
        return parser
        pass

    @staticmethod
    def initialize(parser):
        file_path = 'your address'
        parser.add_argument('--train_file', default=file_path + 'data/train.txt', type=str)
        parser.add_argument('--valid_file', default=file_path + 'data/val.txt', type=str)
        parser.add_argument('--test_file', default=file_path + 'data/test.txt', type=str)
        parser.add_argument('--label_file', default=file_path + 'data/label.txt', type=str)
        parser.add_argument('--vocab', default=file_path + 'data/vocab.txt', type=str)
        parser.add_argument('--split_seq', default='_!_', type=str)
        parser.add_argument('--checkpoint', default=file_path + 'checkpoint/best_bert_cls.pth')
        # loader
        parser.add_argument('--batch_size', default=32, type=int)
        parser.add_argument('--eval_batch_size', default=32, type=int)
        parser.add_argument('--shuffle', default=False, type=bool)
        # model parameters
        parser.add_argument('--pretrained_model', default=file_path + 'bert-based-uncased', type=str)
        parser.add_argument('--num_labels', default=3, type=int)
        parser.add_argument('--hidden_dropout', default=0.3, type=float)
        parser.add_argument('--hidden_size', default=768, type=int)
        parser.add_argument('--max_len', default=70, type=int)
        # hyperparameters
        parser.add_argument('--device', default='cuda:0', type=str)
        parser.add_argument('--epoch', default=30, type=int)
        parser.add_argument('--lr', default=0.00005, type=float)
        parser.add_argument('--gradient_accumulation_steps', default=4, type=int)
        parser.add_argument('--max_grad_norm', default=1.0, type=float)
        parser.add_argument('--num_warmup', default=300, type=int)
        parser.add_argument('--weight_decay', default=0.001, type=float)
        # trained model
        parser.add_argument('--save_model', default='', type=str)
        return parser

    def get_parser(self):
        parser = self.parser()
        parser = self.initialize(parser)
        return parser.parse_args()
        pass
