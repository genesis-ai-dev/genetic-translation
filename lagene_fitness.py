from Finch.generic import Environment, Individual
from genetics import LaGenePool, LaGene
from layers import AfterLife
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
from typing import List
import numpy as np

def sim(a, b):
    return SequenceMatcher(None, a, b).ratio() * 100

def close_lagenes(items, reference_item, n, include_reference=False):

    # Calculate the distance of each item from the reference item
    distances = [(item, abs(item.item.positional_rank - reference_item.item.positional_rank)) for item in items if
                 item.fitness > 0 or item.item.name == reference_item.item.name]
    # Sort the items based on the calculated distance
    sorted_items = sorted(distances, key=lambda x: x[1])

    # Filter out the reference item if it's included in the distances and we're not including it in the final list
    if not include_reference:
        sorted_items = [item for item in sorted_items if item[0].item.name != reference_item.item.name]
        n -= 1
    # Select the top n closest items
    closest_items = [item[0] for item in sorted_items[:n]]

    # If include_reference is True, ensure the reference_item is in the correct position
    if include_reference and reference_item not in closest_items:
        closest_items.append(reference_item)
        closest_items.sort(key=lambda item: abs(item.item.positional_rank - reference_item.item.positional_rank))

    return closest_items
class CommunalFitness:
    def __init__(self, environment: Environment, gene_pool: LaGenePool, n_texts: int, n_lagenes: int,
                 afterlife: AfterLife):
        self.environment = environment
        self.gene_pool = gene_pool
        self.fitness_history = []
        self.reset = 0
        self.fitness_memory = {}
        self.n_texts = n_texts
        self.n_lagenes = n_lagenes
        self.afterlife = afterlife

    def fitness(self, individual: Individual) -> float:
        global global_history, fitness_history

        fitness_rep = f"{individual.item.source} - {individual.item.target}"
        memory = self.fitness_memory.get(fitness_rep, None)

        if memory:
            fitness, name = memory
            if name != individual.item.name:
                return 0
        query = ' '.join(individual.item.source)

        source_sample, target_sample = self.gene_pool.find_samples(query, n=self.n_texts)
        source_sample, target_sample = source_sample.split(), target_sample.split()
        other_lagenes = close_lagenes(items=self.environment.individuals, reference_item=individual, n=self.n_lagenes)
        # Translation without the current LaGene
        translation_without = self.apply_lagenes(other_lagenes, source_sample)
        similarity_without = sim(translation_without, target_sample)
        # Translation with the current LaGene
        all_lagenes = close_lagenes(items=self.environment.individuals, reference_item=individual, n=self.n_lagenes,
                                    include_reference=True)

        translation_with = self.apply_lagenes(all_lagenes, source_sample)

        similarity_with = sim(translation_with, target_sample)
        length = len(self.environment.history['population'])

        if length != self.reset:
            all_lagenes += self.afterlife.individuals
            translation_with = self.apply_lagenes(all_lagenes, self.gene_pool.source_text)

            total = sim(translation_with, self.gene_pool.target_text)
            self.fitness_history.append(total)
            self.reset = length


        improvement = similarity_with - similarity_without
        if not self.fitness_memory.get(fitness_rep, None):
            self.fitness_memory[fitness_rep] = (improvement, individual.item.name)

        if improvement < 0:
            self.environment.individuals = [ind for ind in self.environment.individuals if ind != individual]
        return improvement

    def final(self):
        other_lagenes = [ind for ind in self.environment.individuals + self.afterlife.individuals]
        translation_without = self.apply_lagenes(other_lagenes, self.gene_pool.source_text)
        return " ".join(translation_without)

    def apply_lagenes(self, lagenes: List[LaGene], text: np.ndarray) -> str:
        for individual in sorted(lagenes, key=lambda x: len(x.item.source) + x.item.order_boost):
            text = individual.item.apply(text)
        return text

    def plot(self):
        plt.plot(self.fitness_history)
        plt.ylabel('Communal Fitness')
        plt.xlabel("Generation")
        plt.show()