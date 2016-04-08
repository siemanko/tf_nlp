import math
import random

import progressbar

def get_pb(name):
    return progressbar.ProgressBar(widgets=[name,
                                            progressbar.AdaptiveETA(),
                                            ' ',
                                            progressbar.Bar(),
                                            ' ',
                                            progressbar.Percentage(),
                                            ])

def make_batches(examples, batch_size, sorting_key=None, enforce_size=False, shuffle=True):
    examples = examples[:]
    if sorting_key:
        examples.sort(key=sorting_key)

    batches = [examples[bs:(bs+batch_size)] for bs in range(0,len(examples), batch_size)]
    if shuffle:
        random.shuffle(batches)

    if enforce_size:
        batches = [b for b in batches if len(b) == batch_size]
    return batches

def flatten(thing):
    """Takes arbitrary nested lists and returns a flat list.
    Examples:
         1             => [1]
         [1,[2],[3,[4]]] => [1,2,3,4]
         [1,2,3,4]     => [1,2, 3, 4]
    """
    if isinstance(thing, (list, tuple)):
        return sum([flatten(thing_item) for thing_item in thing], [])
    else:
        return [thing]


def validation_split(data, validation_fraction=0.1):
    assert(0 <= validation_fraction <= 1.0)
    validation_size = round(validation_fraction * len(data))
    return data[:-validation_size], data[-validation_size:]

def find_common_examples(ds1, ds2):
    for example in ds1:
        if example in ds2:
            yield example
