from Finch.layers.universal_layers import *
from Finch.generic import GenePool, Individual
from Finch.rates import make_callable
from distance import l_replace, l_shift
from markov import AutoLexicon, BiMarkovChain
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uuid


class LaGene:
    def __init__(self, source: np.ndarray, target: np.ndarray, shift: int, positional_rank: float):
        self.source = source
        self.target = target
        self.shift = shift
        self.positional_rank = positional_rank
        self.name = str(uuid.uuid4())
        self.order_boost = 0  # TODO: implement mutations and crossover for this. (maybe)

    def apply(self, sequence_text: np.ndarray):
        new_sequence = l_replace(sequence_text, self.source, self.target)
        # if self.shift != 0:  # TODO: figure out how to do this quickly
        #     new_sequence = l_shift(new_sequence, self.target, self.shift)

        return new_sequence

    def __str__(self):
        return f"{self.name}: ({self.positional_rank}) + {self.order_boost}  | {' '.join(self.source)} <fromto> {' '.join(self.target)}"

    def dict(self):
        return {'name': self.name, 'positional_rank': self.positional_rank, 'order_boost': self.order_boost,
                'source': self.source, 'target': self.target}


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

        # Create TF-IDF matrix for source sentences
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.source_lines)

    def tokenize(self, text):
        self.temperature = self.temperature
        return text.split(" ")

    def generate_lagene(self):
        source, target = self.AutoLexicon.get_random_pair(self.temperature())
        if random.random() > .5:
            source, target = target, source
        lagene = LaGene(source=[source], target=[target], shift=0,
                        positional_rank=random.random())
        individual = Individual(item=lagene, fitness_function=self.fitness_function)
        return individual

    def find_samples(self, query, n):
        top_n_indices, _ = self.find_inds(query, n)

        source_samples = [self.source_lines[i] for i in top_n_indices]
        target_samples = [self.target_lines[i] for i in top_n_indices]

        source_str = " ".join(source_samples)
        target_str = " ".join(target_samples)

        return source_str, target_str

    def find_inds(self, query, n):
        query_vector = self.tfidf_vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_n_indices = np.argsort(cosine_similarities)[-n:][::-1]
        return top_n_indices, cosine_similarities[top_n_indices]
