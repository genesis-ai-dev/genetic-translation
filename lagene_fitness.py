from typing import List
import numpy as np
from difflib import SequenceMatcher
from Finch.generic import Environment, Individual
from genetics import LaGenePool, LaGene
from layers import AfterLife


def similarity(a: List[str], b: List[str], source) -> float:

    return (SequenceMatcher(None, a, b).ratio() * 100)


def get_closest_lagenes(items: List[Individual], reference: Individual, n: int, include_reference: bool = False) -> \
List[Individual]:
    distances = [(item, abs(item.item.positional_rank - reference.item.positional_rank))
                 for item in items if item.fitness > 0.0 or item.item.name == reference.item.name]
    sorted_items = sorted(distances, key=lambda x: x[1])
    filtered_items = [item for item, _ in sorted_items if item.item.name != reference.item.name]

    num_to_select = n - 1 if include_reference else n
    closest_items = filtered_items[:num_to_select]
    closest_items.sort(key=lambda item: len(item.item.source), reverse=True)

    if include_reference:
        closest_items =  closest_items + [reference]

    return closest_items


class CommunalFitness:
    def __init__(self, environment: Environment, gene_pool: LaGenePool, n_texts: int, n_lagenes: int,
                 afterlife: AfterLife, query_text: str = None):
        self.environment = environment
        self.gene_pool = gene_pool
        self.afterlife = afterlife
        self.fitness_history = []
        self.useful_lagenes = {}
        self.fitness_memory = {}
        self.n_texts = n_texts
        self.n_lagenes = n_lagenes
        self.reset_count = 0
        self.updates = 0
        self.query_text = query_text.split() if query_text else None

    def fitness(self, individual: Individual) -> float:
        fitness_key = f"{individual.item.source} - {individual.item.target}"

        # if fitness_key in self.fitness_memory:
        #     stored_fitness, stored_name = self.fitness_memory[fitness_key]
        #     if stored_name != individual.item.name:
        #         return individual.fitness * 0.8

        query = ' '.join(individual.item.source)
        source_sample, target_sample = self.gene_pool.find_samples(query, n=self.n_texts)
        source_sample, target_sample = source_sample.split(), target_sample.split()

        other_lagenes = get_closest_lagenes(self.environment.individuals, individual, self.n_lagenes)
        all_lagenes = get_closest_lagenes(self.environment.individuals, individual, self.n_lagenes,
                                          include_reference=True)

        translation_without = self.apply_lagenes(other_lagenes, source_sample)
        translation_with = self.apply_lagenes(all_lagenes, source_sample)

        similarity_without = similarity(translation_without, target_sample, source_sample)
        similarity_with = similarity(translation_with, target_sample, source_sample)

        self.update_fitness_history()

        improvement = similarity_with - similarity_without
        # self.fitness_memory[fitness_key] = (improvement, individual.item.name)

        if improvement < 0:
            self.environment.individuals = [ind for ind in self.environment.individuals if ind != individual]
        elif improvement > 1:
            self.update_useful_lagenes(individual, improvement)

        if self.query_text:
            before = self.query_text.copy()
            after = self.apply_lagenes([individual], before.copy())
            if before != after:
                return improvement * 2
            else:
                return improvement
        else:
            return improvement

    def update_fitness_history(self):
        current_length = len(self.environment.history['population'])
        if current_length != self.reset_count:
            all_lagenes = self.environment.individuals + self.afterlife.individuals
            translation = self.apply_lagenes(all_lagenes, self.gene_pool.source_text)
            total_similarity = similarity(translation, self.gene_pool.target_text, self.gene_pool.source_text)
            self.fitness_history.append(total_similarity)
            self.reset_count = current_length

    def update_useful_lagenes(self, individual: Individual, improvement: float):
        key = ' '.join(individual.item.source)

        if key not in self.useful_lagenes or self.useful_lagenes[key][1] < improvement:
            self.useful_lagenes[key] = (individual.item.dict(), improvement)
            self.updates += 1


    def apply_lagenes(self, lagenes: List[LaGene], text: List[str]) -> List[str]:
        sorted_lagenes = sorted(lagenes, key=lambda x: len(x.item.source) + x.item.order_boost)
        for lagene in sorted_lagenes:
            text = lagene.item.apply(text)
        return text

    def final(self) -> str:
        all_lagenes = self.afterlife.individuals
        if self.query_text:
            translation = self.apply_lagenes(all_lagenes, self.query_text)
        else:
            translation = self.apply_lagenes(all_lagenes, self.gene_pool.source_text)


        return " ".join(translation)

    def plot(self):
        import matplotlib.pyplot as plt
        plt.plot(self.fitness_history)
        plt.ylabel('Communal Fitness')
        plt.xlabel("Generation")
        plt.show()