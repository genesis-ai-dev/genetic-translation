from Finch.generic import Environment
from Finch.layers.universal_layers import *
from Finch.selectors import RandomSelection, RankBasedSelection
from lagene_selectors import PositionalSelection
from lagene_fitness import CommunalFitness
from genetics import LaGenePool
from layers import LexiconMutation, NPointLaGeneCrossover, MarkovMutation, MassExtinction, MigrationLayer, AfterLife
import matplotlib.pyplot as plt

# Config
start_pop = 50
max_pop = 1000

markov_mutation_selection = RankBasedSelection(factor=-10, amount_to_select=8)
lexicon_mutation_selection = RankBasedSelection(factor=-10, amount_to_select=8)
crossover_selection = RankBasedSelection(factor=3, amount_to_select=2)
positional_crossover_selection = PositionalSelection(2, temperature=.02)

children = 30
families = 20
migration_selection = RandomSelection(percent_to_select=1)
op_mutation = False
generations = 1000
# Setup
pool = LaGenePool(source_file='source.txt', target_file='target.txt', temperature=15,
                  fitness_function=lambda x: 100)  # placeholder fitness that will be replaced
env = Environment(layers=[Populate(population=start_pop, gene_pool=pool)], individuals=[], verbose_every=5)
afterlife = AfterLife(start_at=10, n_best=1, period=5, threshold=100)
fitness = CommunalFitness(environment=env, gene_pool=pool, n_texts=8, n_lagenes=4, afterlife=afterlife, query_text='die deutsche tourismusbranche ist ein wichtiger wirtschaftsfaktor mit vielen attraktionen')
pool.fitness_function = fitness.fitness  # reassign the fitness function

# Create remaining layers
#crossover = SimpleLaGeneCrossover(parent_selection=positional_crossover_selection.select, total_children=total_children,
#                                  gene_pool=pool)
crossover = NPointLaGeneCrossover(parent_selection=positional_crossover_selection.select, families = families, children_per_family=children, gene_pool=pool, n_points=3)
mutation = LexiconMutation(selection=lexicon_mutation_selection.select, gene_pool=pool, overpowered=op_mutation)
markov_mutation = MarkovMutation(selection=markov_mutation_selection.select, gene_pool=pool, overpowered=False,
                                 mutation_obedience=.8)
sorting_layer = SortByFitness()
cap_layer = CapPopulation(max_population=max_pop)
extinction = MassExtinction(period=10)
migration = MigrationLayer(selection=migration_selection.select, gene_pool=pool, hard_reset=True, scope_size=1)
# Add layers to environment

env.add_layer(crossover)
env.add_layer(migration)
env.add_layer(markov_mutation)  # Super needed
env.add_layer(mutation)
env.add_layer(extinction)
env.add_layer(sorting_layer)
env.add_layer(afterlife)
env.add_layer(cap_layer)
env.compile()
env.evolve(generations=generations)

fitness.plot()
#
plt.plot(env.history['population'])
plt.ylabel('Population')
plt.xlabel('Generation')
plt.show()

print(fitness.final())

for ind in env.individuals:
    print(f"{ind.item.source} -> {ind.item.target}: {ind.fitness}")

print("After life: ")
for ind in afterlife.individuals:
    print(f"{ind.item.source} -> {ind.item.target}: {ind.fitness}")

print(f"Useful updates: {fitness.updates}")


def sort_and_replace(data_dict, query_text):
    # Sort the dictionary items by the length of the source text (first element of the tuple)
    sorted_items = sorted(data_dict.items(), key=lambda x: len(x[1][0]), reverse=False)

    # Perform text replacement
    result = query_text
    for key, (replace_text, _) in sorted_items:
        result = result.replace(key, replace_text)

    return result


while 1:
    try:
        query = input("Enter: ")
        if query == "exit":
            exit()
        print(sort_and_replace(fitness.useful_lagenes, query))
    except:
        print("er")
