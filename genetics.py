from Finch.layers.universal_layers import *
from Finch.selectors import RandomSelection, RankBasedSelection
from Finch.generic import GenePool, Layer, Individual
from Finch.rates import make_callable
from distance import l_replace, l_shift, similarity
from markov import AutoLexicon, BiMarkovChain
import numpy as np
import random


class LaGene:
    def __init__(self, source: np.ndarray, target: np.ndarray, shift: int, positional_rank: float):
        self.source = source
        self.target = target
        self.shift = shift
        self.positional_rank = positional_rank

    def apply(self, sequence_text: np.ndarray):

        new_sequence = l_replace(sequence_text, self.source, self.target)
        # if self.shift != 0:
        #     new_sequence = l_shift(new_sequence, self.target, self.shift)

        return new_sequence


class LaGenePool(GenePool):
    def __init__(self, source_file: str, target_file: str, temperature: Union[Callable, float],
                 fitness_function: Callable):
        super().__init__(generator_function=self.generate_lagene, fitness_function=fitness_function)

        self.temperature = make_callable(temperature)

        with open(source_file) as f:
            self.source_text = f.read().lower()
            self.source_lines = self.source_text.split('\n')
        with open(target_file) as f:
            self.target_text = f.read().lower()
            self.target_lines = self.target_text.split('\n')

        self.markov = BiMarkovChain()

        split_source_lines = []
        split_target_lines = []

        for source_line, target_line in zip(self.source_lines, self.target_lines):
            split_source = self.tokenize(source_line)
            split_target = self.tokenize(target_line)

            self.markov.add_seq(split_source)
            self.markov.add_seq(split_target)

            split_source_lines.append(split_source)
            split_target_lines.append(split_target)

        self.AutoLexicon = AutoLexicon(split_source_lines, split_target_lines)

        self.source_text = self.tokenize(self.source_text)
        self.target_text = self.tokenize(self.target_text)
    def tokenize(self, text):
        self.temperature = self.temperature
        return text.split(" ")

    def generate_lagene(self):
        source, target = self.AutoLexicon.get_random_pair(self.temperature())
        if random.random() > .5:
            source, target = target, source
        lagene = LaGene(source=[source], target=[target], shift=random.choice([-1, 0, 1]), positional_rank=random.random())
        individual = Individual(item=lagene, fitness_function=self.fitness_function)

        return individual
