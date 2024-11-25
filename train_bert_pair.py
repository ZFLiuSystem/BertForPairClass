import torch
from transformers import get_scheduler, AdamW
from transformers import BertTokenizer
from process_sentence import LoadingPairSentenceClassificationSet
from config import Args
from bert_model import BertForPairClass
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score, f1_score


class Trainer:
    def __init__(self, model_config, train_loader, valid_loader, test_loader):
        self.model_config = model_config

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.model = BertForPairClass(self.model_config, bert_out_mode='mean')
        print(self.model)
        self.device = torch.device(model_config.device)

        model_parameters = self.set_parameters()
        self.optimizer = AdamW(params=model_parameters, lr=self.model_config.lr, eps=1e-9, no_deprecation_warning=True)
        self.scheduler = get_scheduler(name='linear', optimizer=self.optimizer,
                                       num_warmup_steps=self.model_config.num_warmup,
                                       num_training_steps=len(train_loader) * model_config.epoch)
        pass

    def set_parameters(self):
        no_decay = ['bias', 'LayerNorm', 'Layer.weight']
        model_parameters = list(self.model.named_parameters())
        optimized_param = [
            {'params': [p for p_name, p in model_parameters if not any(nd in p_name for nd in no_decay)],
             'weight_decay': self.model_config.weight_decay},
            {'params': [p for p_name, p in model_parameters if any(nd in p_name for nd in no_decay)],
             'weight_decay': .0}]
        return optimized_param

    @staticmethod
    def calculate_acc(pred: list, labels: list):
        accuracy = {}
        acc = accuracy_score(labels, pred)
        f1_macro = f1_score(labels, pred, average='macro')
        f1_micro = f1_score(labels, pred, average='micro')
        accuracy['acc'] = acc
        accuracy['f1_macro'] = f1_macro
        accuracy['f1_micro'] = f1_micro
        return accuracy

    @staticmethod
    def generate_masks_type_ids(a_batch_samples):
        batch_size, sequence = a_batch_samples.shape
        conditions_masks = (a_batch_samples != 0)
        masks_ = torch.ones([batch_size, sequence], dtype=torch.long)
        masks = torch.where(conditions_masks, masks_, 0)
        return masks
        pass

    def train(self):
        self.model.to(self.device)
        best_valid_acc = .0
        for epoch in range(1, self.model_config.epoch + 1):
            self.model.train()
            batch_total_loss = .0
            batch_train_pred = []
            batch_train_label = []
            print(80 * '-')
            print('Epoch[{}/{}]'.format(self.model_config.epoch, epoch))
            start_time = time.time()
            for i_batch, (a_batch_samples, a_batch_labels, a_batch_type_ids) in enumerate(tqdm(self.train_loader)):
                attn_masks = self.generate_masks_type_ids(a_batch_samples).to(self.device)
                token_type_ids = a_batch_type_ids.to(self.device)
                inputs_ids = a_batch_samples.to(self.device)
                labels = a_batch_labels.to(self.device)

                loss, logits = self.model(input_ids=inputs_ids,
                                          token_type_ids=token_type_ids,
                                          attention_mask=attn_masks,
                                          labels=labels,)

                batch_total_loss += loss.item()
                loss.backward()

                if (epoch + 1) % self.model_config.gradient_accumulation_steps == 0 \
                        or len(self.train_loader) == (epoch + 1):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.model_config.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                pred = torch.argmax(logits, dim=-1)
                batch_train_pred.extend(pred.cpu().numpy())
                batch_train_label.extend(labels.cpu().numpy())

            end_time = time.time()
            train_time = end_time - start_time
            train_loss = batch_total_loss / len(self.train_loader)
            train_accuracy = self.calculate_acc(batch_train_pred, batch_train_label)
            prompts = "Train Time: {:.2f}secs, Train Loss: {:.2f},\n" \
                      + "Train Accuracy: {:.3f}, Train F1_micro: {:.3f}, "\
                      + "Train F1_macro: {:.3f}."
            print(prompts.format(train_time, train_loss, train_accuracy['acc'],
                                 train_accuracy['f1_micro'], train_accuracy['f1_macro']))
            self.model.eval()
            valid_time, valid_loss, valid_pred, valid_labels = self.validate()
            valid_accuracy = self.calculate_acc(valid_pred, valid_labels)
            prompts = "Valid Time: {:.2f}secs, Valid Loss: {:.2f},\n" \
                      + "Valid Accuracy: {:.3f}, Valid F1_micro: {:.3f}, " \
                      + "Valid F1_macro: {:.3f}."
            print(prompts.format(valid_time, valid_loss, valid_accuracy['acc'],
                                 valid_accuracy['f1_micro'], valid_accuracy['f1_macro']))
            if valid_accuracy['acc'] > best_valid_acc:
                check_point = {
                    'epoch': epoch,
                    'loss': valid_loss,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                }
                torch.save(check_point, self.model_config.checkpoint)
                best_valid_acc = valid_accuracy['acc']
            print(80 * '-')
        pass

    def validate(self):
        batch_loss = .0
        batch_pred = []
        batch_labels = []
        start_time = time.time()
        with torch.no_grad():
            for i_batch, (a_batch_samples, a_batch_labels, a_batch_type_ids) in enumerate(tqdm(self.valid_loader)):
                attn_masks = self.generate_masks_type_ids(a_batch_samples).to(self.device)
                token_type_ids = a_batch_type_ids.to(self.device)
                inputs_ids = a_batch_samples.to(self.device)
                labels = a_batch_labels.to(self.device)

                loss, logits = self.model(input_ids=inputs_ids, attention_mask=attn_masks,
                                          token_type_ids=token_type_ids, labels=labels)
                batch_loss += loss.item()

                pred = torch.argmax(logits, dim=-1)
                batch_pred.extend(pred.cpu().numpy())
                batch_labels.extend(labels.cpu().numpy())
        batch_loss = batch_loss / len(self.valid_loader)
        end_time = time.time()
        valid_time = end_time - start_time
        return valid_time, batch_loss, batch_pred, batch_labels
        pass

    def test(self, id2label: dict = None, convert_id2label: bool = False):
        checkpoint = torch.load(self.model_config.checkpoint)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.model.to(self.device)
        test_loss = .0
        test_pred = []
        test_labels = []
        start_time = time.time()
        with torch.no_grad():
            for i_batch, (a_batch_samples, a_batch_labels, a_batch_type_ids) in enumerate(tqdm(self.test_loader)):
                attn_masks = self.generate_masks_type_ids(a_batch_samples).to(self.device)
                token_type_ids = a_batch_type_ids.to(self.device)
                inputs_ids = a_batch_samples.to(self.device)
                labels = a_batch_labels.to(self.device)

                loss, logits = self.model(input_ids=inputs_ids, attention_mask=attn_masks,
                                          token_type_ids=token_type_ids, labels=labels)
                test_loss += loss.item()
                pred = torch.argmax(logits, dim=-1)
                test_pred.extend(pred.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
        end_time = time.time()
        test_time = end_time - start_time
        test_loss = test_loss / len(self.test_loader)
        test_accuracy = self.calculate_acc(test_pred, test_labels)
        if convert_id2label and id2label is not None:
            test_labels = [id2label[i] for i in test_labels]
            pred_labels = [id2label[i] for i in test_pred]
            print(f"There are {len(test_labels)} tested samples totally.")
            for i, (label, pred) in enumerate(zip(test_labels, pred_labels)):
                print(80 * '-')
                if i + 1 == 1:
                    prompts = 'The tested {}st sample: Actual Label: {}, Predicted Label: {}.'
                elif i + 1 == 2:
                    prompts = 'The tested {}nd sample: Actual Label: {}, Predicted Label: {}.'
                elif i + 1 == 3:
                    prompts = 'The tested {}rd sample: Actual Label: {}, Predicted Label: {}.'
                else:
                    prompts = 'The tested {}th sample: Actual Label: {}, Predicted Label: {}.'
                if label == pred:
                    print('This judgement is correct.')
                else:
                    print('This prediction exists some errors.')
                print(prompts.format(i + 1, label, pred))
                print(80 * '-')
        print(self.model)
        return test_time, test_loss, test_accuracy
        pass


def main():
    parser = Args()
    model_config = parser.get_parser()

    id2label = {}
    label2id = {}
    with open(model_config.label_file, 'r', encoding='utf-8') as f:
        labels = f.read().strip().split('\n')
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label
    bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=model_config.pretrained_model)

    train_set = LoadingPairSentenceClassificationSet(set_file=model_config.train_file,
                                                     vocab_file=model_config.vocab,
                                                     tokenizer=bert_tokenizer,
                                                     split_sentence=model_config.split_seq,
                                                     max_len=model_config.max_len,
                                                     batch_size=model_config.batch_size
                                                     )
    train_loader = train_set.set_loader(mode='train')
    valid_set = LoadingPairSentenceClassificationSet(set_file=model_config.valid_file,
                                                     vocab_file=model_config.vocab,
                                                     tokenizer=bert_tokenizer,
                                                     split_sentence=model_config.split_seq,
                                                     max_len=model_config.max_len,
                                                     batch_size=model_config.eval_batch_size
                                                     )
    valid_loader = valid_set.set_loader()
    test_set = LoadingPairSentenceClassificationSet(set_file=model_config.test_file,
                                                    vocab_file=model_config.vocab,
                                                    tokenizer=bert_tokenizer,
                                                    split_sentence=model_config.split_seq,
                                                    max_len=model_config.max_len,
                                                    batch_size=model_config.eval_batch_size
                                                    )
    test_loader = test_set.set_loader()
    trainer = Trainer(model_config, train_loader, valid_loader, test_loader)
#   trainer.train()

    test_time, test_loss, test_accuracy = trainer.test(id2label, True)
    prompts = "Test Time: {:.2f}secs, Test Loss: {:.2f},\n" \
              + "Test Accuracy: {:.3f},Test F1_micro: {:.3f}, "\
              + "Test F1_macro: {:.3f}."
    print(prompts.format(test_time, test_loss, test_accuracy['acc'],
                         test_accuracy['f1_micro'], test_accuracy['f1_macro']))
    pass


if __name__ == '__main__':
    main()
    pass
