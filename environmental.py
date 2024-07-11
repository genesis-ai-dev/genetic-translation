from Finch.generic import Environment
from Finch.layers.universal_layers import *
from Finch.selectors import RandomSelection, RankBasedSelection
from lagene_selectors import PositionalSelection
from lagene_fitness import CommunalFitness
from genetics import LaGenePool
from layers import LexiconMutation, SimpleLaGeneCrossover, MarkovMutation, MassExtinction, MigrationLayer, AfterLife
import matplotlib.pyplot as plt

# Config
start_pop = 50
max_pop = 100
markov_mutation_selection = RankBasedSelection(factor=5, amount_to_select=12)
lexicon_mutation_selection = RankBasedSelection(factor=-5, amount_to_select=12)

crossover_selection = RankBasedSelection(factor=3, amount_to_select=2)
positional_crossover_selection = PositionalSelection(2, temperature=.2)

total_children = 15
migration_selection = RandomSelection(percent_to_select=1)
op_mutation = True
generations = 40

# Setup
pool = LaGenePool(source_file='source.txt', target_file='target.txt', temperature=7,
                  fitness_function=lambda x: 100)  # placeholder fitness that will be replaced
env = Environment(layers=[Populate(population=start_pop, gene_pool=pool)], individuals=[])
afterlife = AfterLife(start_at=10, n_best=1, period=5, threshold=20)
fitness = CommunalFitness(environment=env, gene_pool=pool, n_texts=2, n_lagenes=2, afterlife=afterlife)
pool.fitness_function = fitness.fitness  # reassign the fitness function

# Create remaining layers
crossover = SimpleLaGeneCrossover(parent_selection=positional_crossover_selection.select, total_children=total_children,
                                  gene_pool=pool)
mutation = LexiconMutation(selection=lexicon_mutation_selection.select, gene_pool=pool, overpowered=op_mutation)
markov_mutation = MarkovMutation(selection=markov_mutation_selection.select, gene_pool=pool, overpowered=op_mutation)
sorting_layer = SortByFitness()
cap_layer = CapPopulation(max_population=max_pop)
extinction = MassExtinction(period=7)
migration = MigrationLayer(selection=migration_selection.select, gene_pool=pool, hard_reset=True, scope_size=3)
# Add layers to environment

env.add_layer(crossover)
env.add_layer(migration)
env.add_layer(markov_mutation)  # Super needed
#env.add_layer(mutation)
env.add_layer(extinction)
env.add_layer(sorting_layer)
env.add_layer(afterlife)
env.add_layer(cap_layer)
env.compile()
env.evolve(generations=generations)

fitness.plot()
#
# plt.plot(env.history['population'])
# plt.ylabel('Population')
# plt.xlabel('Generation')
# plt.show()

print(fitness.final())

for ind in env.individuals:
    print(f"{ind.item.source} -> {ind.item.target}: {ind.fitness}")

print("After life: ")
for ind in afterlife.individuals:
    print(f"{ind.item.source} -> {ind.item.target}: {ind.fitness}")

print(f"Every useful: {len(fitness.useful_lagenes)}")
print(fitness.useful_lagenes)
