from typing import List

import numpy as np

from Finch.generic import Environment, Individual
from Finch.layers.universal_layers import *
from Finch.selectors import RandomSelection, RankBasedSelection, TournamentSelection
from genetics import LaGenePool, LaGene
from layers import LexiconMutation, SimpleLaGeneCrossover, MarkovMutation
from distance import similarity
import matplotlib.pyplot as plt
from difflib import SequenceMatcher

def sim(a, b):
    return SequenceMatcher(None, a, b).ratio() * 100

class CommunalFitness:
    def __init__(self, environment: Environment, gene_pool: LaGenePool):
        self.environment = environment
        self.gene_pool = gene_pool
        self.fitness_history = []


    def fitness(self, individual: Individual) -> float:
        global global_history
        lagene = individual.item
        other_lagenes = [ind.item for ind in self.environment.individuals if ind.item != lagene]

        # Translation without the current LaGene
        translation_without = self.apply_lagenes(other_lagenes, self.gene_pool.source_text)
        similarity_without = sim(translation_without, self.gene_pool.target_text)

        # Translation with the current LaGene
        all_lagenes = other_lagenes + [lagene]
        translation_with = self.apply_lagenes(all_lagenes, self.gene_pool.source_text)
        similarity_with = sim(translation_with, self.gene_pool.target_text)
        self.fitness_history.append(similarity_with)
        improvement = similarity_with - similarity_without

        if improvement < 0:
            self.environment.individuals = [ind for ind in self.environment.individuals if ind.item != lagene]
        return improvement

    def final(self):
        other_lagenes = [ind.item for ind in self.environment.individuals]
        translation_without = self.apply_lagenes(other_lagenes, self.gene_pool.source_text)
        return " ".join(translation_without)

    def apply_lagenes(self, lagenes: List[LaGene], text: np.ndarray) -> str:
        for lagene in sorted(lagenes, key=lambda x: x.positional_rank):
            text = lagene.apply(text)
        return text

    def plot(self):
        plt.plot(self.fitness_history)
        plt.ylabel('Fitness')
        plt.xlabel("Generation")
        plt.show()


# Config
start_pop = 90
max_pop = 100
mutation_selection = RankBasedSelection(factor=-40, amount_to_select=8)
crossover_selection = RankBasedSelection(factor=40, amount_to_select=2)
total_children = 8
op_mutation = True
generations = 200


# Setup
pool = LaGenePool(source_file='source.txt', target_file='target.txt', temperature=1,
                  fitness_function=lambda x: 100)  # placeholder fitness that will be replaced
env = Environment(layers=[
    Populate(population=start_pop, gene_pool=pool),
], individuals=[])


fitness = CommunalFitness(environment=env, gene_pool=pool)
pool.fitness_function = fitness.fitness  # reassign the fitness function

# Create remaining layers
crossover = SimpleLaGeneCrossover(parent_selection=crossover_selection.select, total_children=total_children,
                                  gene_pool=pool)
mutation = LexiconMutation(selection=mutation_selection.select, gene_pool=pool, overpowered=op_mutation)
markov_mutation = MarkovMutation(selection=mutation_selection.select, gene_pool=pool, overpowered=op_mutation)
sorting_layer = SortByFitness()
cap_layer = CapPopulation(max_population=max_pop)

# Add layers to environment

env.add_layer(crossover)
env.add_layer(cap_layer)

env.add_layer(markov_mutation)
env.add_layer(mutation)

env.add_layer(sorting_layer)
env.add_layer(cap_layer)
env.compile()
env.evolve(generations=generations)
env.plot()
fitness.plot()
print(fitness.final())

for ind in env.individuals:
    print(f"{ind.item.source} -> {ind.item.target}: {ind.item.shift}")