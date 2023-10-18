from __future__ import annotations

from typing import List

import attr
from gensim.models.word2vec import Word2Vec as W2V

from pyrdf2vec.embedders import Embedder
from pyrdf2vec.typings import Embeddings, Entities, SWalk

import random
import json
import os

class MyWalks(object):
    def __init__(self, dirname):
        self.dirname = dirname
        self.files = os.listdir(self.dirname)
        random.shuffle(self.files)

    def __iter__(self):
        for fwalk in self.files:
            with open(os.path.join(self.dirname, fwalk), "r") as f:
                try:
                    # If Walks are saved as json:
                    #walk = tuple(json.load(f))

                    # If walks are saved as txt integers
                    walk = f.readlines()[0].split(' ')

                except:
                    print("PROBLEM WITH: ", fwalk)
                    raise
                    pass
            yield walk


@attr.s(init=False)
class Word2Vec(Embedder):
    """Defines the Word2Vec embedding technique.

    SEE: https://radimrehurek.com/gensim/models/word2vec.html

    Attributes:
        _model: The gensim.models.word2vec model.
            Defaults to None.
        kwargs: The keyword arguments dictionary.
            Defaults to { min_count=0 }.

    """

    kwargs = attr.ib(init=False, default=None)
    _model = attr.ib(init=False, type=W2V, default=None, repr=False)

    def __init__(self, **kwargs):
        self.kwargs = {
            "min_count": 0,
            **kwargs,
        }
        self._model = W2V(**self.kwargs)

    def fit(
        self, walks: List[List[SWalk]], is_update: bool = False
    ) -> Embedder:
        """Fits the Word2Vec model based on provided walks.

        Args:
            walks: The walks to create the corpus to to fit the model.
            is_update: True if the new walks should be added to old model's
                walks, False otherwise.
                Defaults to False.

        Returns:
            The fitted Word2Vec model.

        """
        #corpus = [walk for entity_walks in walks for walk in entity_walks]
        corpus = MyWalks("/media/tim/vol2/walks")

        print("Building Vocabulary...")
        self._model.build_vocab(corpus, update=is_update)
        print("Training Model...")
        self._model.train(
            corpus,
            total_examples=self._model.corpus_count,
            epochs=self._model.epochs,
            #epochs=1,

        )
        return self

    def transform(self, entities: Entities) -> Embeddings:
        """The features vector of the provided entities.

            Args:
                entities: The entities including test entities to create the
                embeddings. Since RDF2Vec is unsupervised, there is no label
                leakage.

        Returns:
            The features vector of the provided entities.

        """


        if not all([entity in self._model.wv for entity in entities]):
            raise ValueError(
                "The entities must have been provided to fit() first "
                "before they can be transformed into a numerical vector."
            )
        return [self._model.wv.get_vector(entity) for entity in entities]
