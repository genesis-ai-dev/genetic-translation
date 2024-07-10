from typing import List

import numpy as np

from Finch.generic import Environment, Individual
from Finch.layers.universal_layers import *
from Finch.selectors import RandomSelection, RankBasedSelection, TournamentSelection
from genetics import LaGenePool, LaGene
from layers import LexiconMutation, SimpleLaGeneCrossover, MarkovMutation, MassExtinction, MigrationLayer
from distance import similarity
import matplotlib.pyplot as plt
from difflib import SequenceMatcher




# Config
start_pop = 50
max_pop = 300
mutation_selection = RankBasedSelection(factor=-1, amount_to_select=8)
crossover_selection = RankBasedSelection(factor=3, amount_to_select=2)
total_children = 4
migration_selection = RandomSelection(percent_to_select=.2)
op_mutation = True
generations = 300

# Setup
pool = LaGenePool(source_file='source.txt', target_file='target.txt', temperature=10,
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
extinction = MassExtinction(period=15)
migration = MigrationLayer(selection=migration_selection.select, gene_pool=pool, hard_reset=True, scope_size=4)
# Add layers to environment

env.add_layer(crossover)
env.add_layer(migration)
env.add_layer(cap_layer)

env.add_layer(markov_mutation)
env.add_layer(mutation)
env.add_layer(extinction)
env.add_layer(sorting_layer)
env.add_layer(cap_layer)
env.compile()
env.evolve(generations=generations)
env.plot()
fitness.plot()

plt.plot(env.history['population'])
plt.ylabel('Population')
plt.xlabel('Generation')
plt.show()

print(fitness.final())

for ind in env.individuals:
    print(f"{ind.item.source} -> {ind.item.target}: {ind.item.shift}")
