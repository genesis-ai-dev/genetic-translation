from Finch.generic import Environment
from Finch.layers.universal_layers import Populate, SortByFitness, CapPopulation
from Finch.selectors import RandomSelection, RankBasedSelection
from lagene_selectors import PositionalRankSelection
from lagene_fitness import CommunalFitness
from genetics import LaGenePool
from layers import LexiconMutation, NPointLaGeneCrossover, MarkovMutation, MassExtinction, MigrationLayer, AfterLife
import matplotlib.pyplot as plt


class TranslationEvolver:
    def __init__(self, source_file, target_file, query_text, start_pop=50, max_pop=500, generations=100):
        self.source_file = source_file
        self.target_file = target_file
        self.query_text = query_text
        self.start_pop = start_pop
        self.max_pop = max_pop
        self.generations = generations

        self.setup_environment()

    def setup_environment(self):
        # Setup gene pool
        self.pool = LaGenePool(source_file=self.source_file, target_file=self.target_file, temperature=15,
                               fitness_function=lambda x: 100)  # placeholder fitness

        # Setup environment
        self.env = Environment(layers=[Populate(population=self.start_pop, gene_pool=self.pool)],
                               individuals=[], verbose_every=5)

        # Setup layers
        self.afterlife = AfterLife(start_at=1, n_best=1, period=1, threshold=10)
        self.fitness = CommunalFitness(environment=self.env, gene_pool=self.pool, n_texts=3, n_lagenes=1,
                                       afterlife=self.afterlife, query_text=self.query_text)
        self.pool.fitness_function = self.fitness.fitness  # reassign the fitness function

        # Selection methods
        markov_mutation_selection = RankBasedSelection(factor=1, amount_to_select=8)
        lexicon_mutation_selection = RankBasedSelection(factor=1, amount_to_select=8)
        positional_crossover_selection = PositionalRankSelection(2, temperature=.1)
        migration_selection = RandomSelection(percent_to_select=1)

        # Create layers
        crossover = NPointLaGeneCrossover(parent_selection=positional_crossover_selection.select,
                                          families=8, children_per_family=4, gene_pool=self.pool, n_points=2)
        mutation = LexiconMutation(selection=lexicon_mutation_selection.select, gene_pool=self.pool, overpowered=True)
        markov_mutation = MarkovMutation(selection=markov_mutation_selection.select, gene_pool=self.pool,
                                         overpowered=True, mutation_obedience=.8)
        sorting_layer = SortByFitness()
        cap_layer = CapPopulation(max_population=self.max_pop)
        extinction = MassExtinction(period=5)
        migration = MigrationLayer(selection=migration_selection.select, gene_pool=self.pool,
                                   hard_reset=True, scope_size=5)

        # Add layers to environment
        self.env.add_layer(migration)
        self.env.add_layer(crossover)
        self.env.add_layer(markov_mutation)
        self.env.add_layer(extinction)
        self.env.add_layer(sorting_layer)
        self.env.add_layer(cap_layer)
        self.env.compile()

    def evolve(self):
        self.env.evolve(generations=self.generations)

    def plot_fitness(self):
        self.fitness.plot()

    def plot_population(self):
        plt.figure()
        plt.plot(self.env.history['population'])
        plt.ylabel('Population')
        plt.xlabel('Generation')
        plt.show()

    def print_results(self):
        print(self.fitness.final())

        print("Individuals:")
        for ind in self.env.individuals:
            print(f"{ind.item.source} -> {ind.item.target}: {ind.fitness}")

        print("\nAfterlife:")
        for ind in self.afterlife.individuals:
            print(f"{ind.item.source} -> {ind.item.target}: {ind.fitness}")

    def run(self):
        self.evolve()
        self.plot_fitness()
        self.plot_population()
        self.print_results()

# Usage example:
# evolver = TranslationEvolver('source.txt', 'target.txt', 'deutsche rechtssystem unterscheidet')
# evolver.run()