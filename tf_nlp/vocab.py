from collections import defaultdict

from .utils import flatten

class Vocab(object):
    UNK = '**UNK**'
    EOS = '**EOS**'

    @staticmethod
    def keep_n_most_frequent(words, n):
        counts = defaultdict(lambda: 0)
        for word in words:
            counts[word] += 1
        counts = list(counts.items())
        counts.sort(key=lambda x: x[1], reverse=True)
        return [word for word, cnt in counts[:n]]


    def __init__(self, words=None, add_eos=True, add_unk=True):
        self.index2word = []
        self.word2index = {}
        self.eos = None
        self.unk = None
        if add_unk:
            self.add(Vocab.UNK)
        if add_eos:
            self.add(Vocab.EOS)

        if words:
            self.add(words)


    def __contains__(self, key):
        if isinstance(key, int):
            return key in range(len(self.index2word))
        elif isinstance(key, str):
            return key in self.word2index
        else:
            raise ValueError("expected(index or string)")

    def add(self, words):
        for word in flatten(words):
            idx = self.word2index.get(word)
            if idx is None:
                idx = len(self.index2word)
                self.index2word.append(word)
                self.word2index[word] = idx
                if word is Vocab.UNK:
                    self.unk = idx
                if word is Vocab.EOS:
                    self.eos = idx

    def fraction_unknown(self, words):
        num_unknown = 0
        num_total   = 0
        for word in flatten(words):
            if word not in self.word2index:
                num_unknown += 1
            num_total += 1
        return num_unknown / num_total

    def words(self):
        return self.word2index.keys()

    def __len__(self):
        return len(self.index2word)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.index2word[index]
        elif isinstance(index, str):
            if self.unk is not None:
                return self.word2index.get(index) or self.unk
            else:
                return self.word2index[index]
        else:
            raise ValueError("expected(index or string)")

    def decode(self, encoded_words, strip_eos=False, decode_type=int):
        encoded_words = flatten(encoded_words)

        words = [self.index2word[word_idx] for word_idx in encoded_words]
        if strip_eos:
            assert self.eos is not None
            return [el for el in words if el != Vocab.EOS]
        else:
            return words

    def encode(self, words, pad_eos=None):
        def encode_word(word):
            if self.unk is not None:
                return self.word2index.get(word) or self.unk
            else:
                return self.word2index[word]

        words = flatten(words)
        encoded_words = [encode_word(word) for word in words]

        if pad_eos is not None:
            assert self.eos is not None
            assert len(encoded_words) <= pad_eos + 1, \
                "Sentence %s is so long it cannot be padded with even single EOS :( " % (words,)
            padded_words = [encode_word(self.eos)] * pad_eos
            for i, encoded_word in enumerate(encoded_words):
                padded_words[i] = encoded_word
            return padded_words
        else:
            return encoded_words
