import torch
from torch.utils.data import Dataset, RandomSampler, DataLoader
from torch.nn.utils.rnn import pad_sequence
from vocab import Vocab


class PairSentenceClassificationSet(Dataset):
    def __init__(self, samples):
        self.input_ids = samples['input_ids']
        self.token_type_ids = samples['token_type_ids']
        self.labels = samples['labels']
        pass

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, ids):
        length = self.__len__()
        assert length >= ids + 1
        input_sample = self.input_ids[ids]
        a_label = self.labels[ids]
        a_type_ids = self.token_type_ids[ids]
        return input_sample, a_label, a_type_ids


class LoadingPairSentenceClassificationSet:
    def __init__(self, set_file, vocab_file, tokenizer, split_sentence, max_len, batch_size):
        self.batch_size = batch_size
        self.max_len = max_len
        self.set_file = set_file
        self.vocab = Vocab(vocab_file)
        self.split_sentence = split_sentence
        self.cls_ids = self.vocab['[CLS]']
        self.sep_ids = self.vocab['[SEP]']
        self.tokenizer = tokenizer
        self.pad_ids = tokenizer.pad_token_id
        self.torch_set = PairSentenceClassificationSet
        pass

    def read_raw_set(self, mode=None):
        samples = {}
        with open(self.set_file, 'r', encoding='utf-8') as file:
            raw_set = file.readlines()
        token_ids = []
        labels = []
        token_type_ids = []
        length = len(raw_set)
        if mode is not None and mode == 'train':
            ratio = 0.01
        else:
            ratio = 1.0
        for raw_sentence in raw_set[:int(ratio * length)]:
            raw_sentence = raw_sentence.rstrip('\n').split(self.split_sentence)
            assert len(raw_sentence) == 3
            sen_1, sen_2, label = raw_sentence[0].lower(), raw_sentence[1].lower(), raw_sentence[2]
            token_group_1 = [self.cls_ids] + [self.vocab[word] for word in self.tokenizer.tokenize(sen_1)]
            type_a = [0 for e in range(len(token_group_1))]
            token_group_2 = ([self.sep_ids] + [self.vocab[word] for word in self.tokenizer.tokenize(sen_2.lower())]
                             + [self.sep_ids])
            type_b = [1 for e in range(len(token_group_2))]
            token_ids.append((token_group_1 + token_group_2))
            token_type_ids.append((type_a + type_b))
            labels.append(int(label))
        samples['input_ids'], samples['labels'], samples['token_type_ids'] = token_ids, labels, token_type_ids
        return samples

    def batch_process(self, batch: list):
        a_batch_samples, a_batch_labels, a_batch_type_ids = [], [], []
        batch_samples_container = list()
        batch_type_ids_container = list()
        for a_group in batch:
            a_sample, a_label, a_type_ids = a_group[0], a_group[1], a_group[2]
            a_batch_samples.append(a_sample)
            a_batch_type_ids.append(a_type_ids)
            a_batch_labels.append(a_label)
        for a_pair_samples, a_pair_type_ids in zip(a_batch_samples, a_batch_type_ids):
            if len(a_pair_samples) != len(a_pair_type_ids):
                break
            else:
                if len(a_pair_samples) > self.max_len:
                    a_pair = a_pair_samples[:self.max_len]
                    batch_samples_container.append(torch.tensor(data=a_pair, dtype=torch.long))
                    a_pair = a_pair_type_ids[:self.max_len]
                    batch_type_ids_container.append(torch.tensor(data=a_pair, dtype=torch.long))
                else:
                    batch_samples_container.append(torch.tensor(data=a_pair_samples, dtype=torch.long))
                    batch_type_ids_container.append(torch.tensor(data=a_pair_type_ids, dtype=torch.long))
        a_batch_samples = pad_sequence(sequences= batch_samples_container,
                                       batch_first=True,
                                       padding_value=self.vocab['[PAD]'])
        a_batch_type_ids = pad_sequence(sequences=batch_type_ids_container,
                                        batch_first=True,
                                        padding_value=self.vocab['[PAD]'])
        a_batch_labels = torch.tensor(data=a_batch_labels, dtype=torch.long)
        return a_batch_samples, a_batch_labels, a_batch_type_ids
        pass

    def set_loader(self, mode=None):
        samples = self.read_raw_set(mode)
        torch_set = self.torch_set(samples)
        if mode is not None and mode == 'train':
            set_sampler = RandomSampler(torch_set)
        else:
            set_sampler = None
        set_loader = DataLoader(dataset=torch_set, sampler=set_sampler, batch_size=self.batch_size,
                                num_workers=0, collate_fn=self.batch_process)
        return set_loader
        pass

