from collections import namedtuple

QaQcDatum = namedtuple("QaQcDatum", ["question", "main_cat", "sub_cat"])

def qa_qc(train_set, test_set):
    def load_dataset(fname):
        dataset = []

        with open(fname, "rb") as f:
            for line in f:
                line = line.replace(b"sister\xf0city", b"sister city")
                line = line.decode("utf-8")

                cats, question = line.split(' ', 1)
                maincat, subcat = cats.split(':')
                question = tuple([token for token in question.split() if token != ''])
                dataset.append(QaQcDatum(question, maincat, subcat))
        return dataset
    return load_dataset(train_set), load_dataset(test_set)
