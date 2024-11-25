class Vocab:
    unk = '[UNK]'

    def __init__(self, vocab_path):
        self.stoi = {}
        self.itos = []
        with open(vocab_path, 'r', encoding='utf-8') as words:
            for i, word in enumerate(words):
                word = word.rstrip('\n')
                self.stoi[word] = i
                self.itos.append(word)

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, word):
        return self.stoi.get(word, self.stoi.get(Vocab.unk))
