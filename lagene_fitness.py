from Finch.generic import Environment, Individual
from genetics import LaGenePool, LaGene
from layers import AfterLife
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
from typing import List
import numpy as np

def sim(a, b):

    difference = 0  # (len(a) - len(b)) * 2 TODO: find out if this is helpful
    return (SequenceMatcher(None, a, b).ratio() * 100) - difference

# def sim(a, b):
#     m = min(len(a), len(b))
#     score = 0
#     for i in range(m):
#         score += 1 if a[i] == b[i] else 0
#     return score
def close_lagenes(items, reference_item, n, include_reference=False):
    # Calculate the distance of each item from the reference item
    distances = [(item, abs(item.item.positional_rank - reference_item.item.positional_rank)) for item in items if
                 item.fitness > 0.0 or item.item.name == reference_item.item.name]

    # Sort the items based on the calculated distance
    sorted_items = sorted(distances, key=lambda x: x[1])

    # Filter out the reference item from the sorted list
    sorted_items = [item for item in sorted_items if item[0].item.name != reference_item.item.name]

    # Determine how many items to select
    num_to_select = n - 1 if include_reference else n

    # Select the top num_to_select closest items
    closest_items = [item[0] for item in sorted_items[:num_to_select]]

    # Sort the closest items based on the length of item.item.source, longest first
    closest_items.sort(key=lambda item: len(item.item.source), reverse=True)

    # If include_reference is True, add the reference_item at the end
    if include_reference:
        closest_items = [reference_item] + closest_items

    return closest_items
class CommunalFitness:
    def __init__(self, environment: Environment, gene_pool: LaGenePool, n_texts: int, n_lagenes: int,
                 afterlife: AfterLife):
        self.environment = environment
        self.gene_pool = gene_pool
        self.fitness_history = []
        self.reset = 0
        self.useful_lagenes = {}
        self.fitness_memory = {}
        self.n_texts = n_texts
        self.n_lagenes = n_lagenes
        self.afterlife = afterlife

    def fitness(self, individual: Individual) -> float:
        global global_history, fitness_history

        fitness_rep = f"{individual.item.source} - {individual.item.target}"
        memory = self.fitness_memory.get(fitness_rep, None)

        # if memory:
        #     fitness, name = memory
        #     if name != individual.item.name:
        #         return individual.fitness * .8
        query = ' '.join(individual.item.source)
        #div = min(len(individual.item.source), len(individual.item.target))

        source_sample, target_sample = self.gene_pool.find_samples(query, n=self.n_texts)
        source_sample, target_sample = source_sample.split(), target_sample.split()
        other_lagenes = close_lagenes(items=self.environment.individuals, reference_item=individual, n=self.n_lagenes)
        # Translation without the current LaGene
        translation_without = self.apply_lagenes(other_lagenes, source_sample)
        similarity_without = sim(translation_without, target_sample) #* div
        # Translation with the current LaGene
        all_lagenes = close_lagenes(items=self.environment.individuals, reference_item=individual, n=self.n_lagenes,
                                    include_reference=True)

        translation_with = self.apply_lagenes(all_lagenes, source_sample)

        similarity_with = sim(translation_with, target_sample) #* div
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

        if improvement > 1:
            key = ' '.join(individual.item.source)
            value = ' '.join(individual.item.target)
            previous_useful = self.useful_lagenes.get(key, None)

            if previous_useful is None:
                self.useful_lagenes.update({key: (value, improvement)})
                return improvement
            if previous_useful[0] == value:
                return improvement
            if previous_useful[1] < improvement:
                print("replaceing")
                print(key + ' -> ' + self.useful_lagenes[key][0])
                print(key + ' -> ' +value)
                self.useful_lagenes[key] = (value, improvement)
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