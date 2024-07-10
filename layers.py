import copy
import random
import numpy
from Finch.generic import Layer, Individual
from genetics import LaGene, LaGenePool
from typing import Callable, Union, List


class SimpleLaGeneCrossover(Layer):
    """
    Performs crossover operation on LaGene individuals.
    """

    def __init__(self, parent_selection: Union[Callable, int], total_children: int, gene_pool: LaGenePool):
        super().__init__(application_function=self.crossover, selection_function=parent_selection,
                         repeat=total_children, refit=False)
        self.gene_pool = gene_pool

    def crossover(self, individuals: List[Individual]) -> Individual:
        """
        Performs crossover between two parent individuals.
        """
        parent1, parent2 = individuals
        assert parent1.item.__class__ == LaGene and parent2.item.__class__ == LaGene, \
            "Individual Items must be LaGenes"

        source1, source2 = parent1.item.source, parent2.item.source
        target1, target2 = parent1.item.target, parent2.item.target
        rank1, rank2 = parent1.item.positional_rank, parent2.item.positional_rank

        source = copy.deepcopy(random.choice([source1, source2]))
        target = copy.deepcopy(random.choice([target1, target2]))

        child_lagene = LaGene(source=source, target=target, shift=0, positional_rank=max([rank1, rank2]) + .05)
        child_lagene = Individual(child_lagene, parent1.fitness_function)
        child_lagene.fit()

        if child_lagene.fitness < max(parent1.fitness, parent2.fitness) or child_lagene.fitness <= 0:
            return

        parent1.fit()
        parent2.fit()

        child_lagene.item.name = parent1.item.name if child_lagene.item.source == parent1.item.source else parent2.item.name
        self.environment.add_individuals([child_lagene])

        return child_lagene


class LexiconMutation(Layer):
    """
    Performs mutation operations on individuals using a lexicon.
    """

    def __init__(self, selection: Union[Callable, int], gene_pool: LaGenePool, overpowered: bool = False):
        super().__init__(application_function=self.mutate_all, selection_function=selection, refit=False)
        self.gene_pool = gene_pool
        self.overpowered = overpowered

    def mutate_all(self, individuals: List[Individual]):
        """
        Applies mutation to all individuals in the list.
        """
        for individual in individuals:
            self.mutate(individual)

    def mutate_source_target(self, individual: Individual):
        """
        Mutates either the source or target of an individual.
        """
        choice = random.choice(['source'])
        source, target = individual.item.source, individual.item.target
        index = random.randint(0, len(source) - 1)

        try:
            if choice == 'source':
                new_sources = self.gene_pool.AutoLexicon.get_occurrences_from_target(target[index])
                new_source = random.choices(
                    [word for word, score in new_sources],
                    weights=[score for word, score in new_sources],
                    k=1
                )[0]
                individual.item.source[index] = new_source
            elif choice == 'target':
                new_targets = self.gene_pool.AutoLexicon.get_occurrences(source[index])
                new_target = random.choices(
                    [word for word, score in new_targets],
                    weights=[score for word, score in new_targets],
                    k=1
                )[0]
                individual.item.target[index] = new_target
        except IndexError:
            pass

    def mutate_rank(self, individual: Individual):
        """
        Mutates the positional rank of an individual.
        """
        individual.item.positional_rank += random.choice([-.02, .02])

    def mutate_shift(self, individual: Individual):
        """
        Mutates the shift of an individual.
        """
        individual.item.shift = random.choice([-1, 1])

    def mutate(self, individual: Individual):
        """
        Applies a random mutation to an individual.
        """
        mutation_type = random.choice(["text", "shift"])

        if self.overpowered:
            old_individual = individual.copy()
            old_fitness = old_individual.fitness

        mapping = {
            "shift": self.mutate_shift,
            "rank": self.mutate_rank,
            "text": self.mutate_source_target
        }

        mapping[mutation_type](individual)

        if self.overpowered:
            new_fitness = individual.fit()
            if new_fitness < old_fitness:
                individual.fitness = old_individual.fitness
                individual.item = old_individual.item


class MarkovMutation(Layer):
    """
    Performs Markov chain-based mutations on individuals.
    """

    def __init__(self, selection: Union[Callable, int], gene_pool: LaGenePool, overpowered: bool = False,
                 mutation_obedience: float = .96):
        super().__init__(application_function=self.mutate_all, selection_function=selection, refit=False)
        self.gene_pool = gene_pool
        self.overpowered = overpowered
        self.mutation_obedience = mutation_obedience

    def mutate_all(self, individuals: List[Individual]):
        """
        Applies mutation to all individuals in the list.
        """
        for individual in individuals:
            self.mutate(individual)

    def mutate(self, individual: Individual):
        """
        Applies Markov chain-based mutation to an individual.
        """
        if self.overpowered:
            old_individual = individual.copy()
            old_fitness = old_individual.fitness

        t = 'add' if len(individual.item.source) == 1 or len(individual.item.target) == 1 else \
            'delete' if max(len(individual.item.source), len(individual.item.target)) > 3 else \
                random.choice(['add', 'delete'])

        self._mutate_sequence(individual.item.source, t)
        self._mutate_sequence(individual.item.target, t)

        if self.overpowered:
            new_fitness = individual.fit()
            if new_fitness < old_fitness:
                individual.fitness = old_individual.fitness
                individual.item = old_individual.item

    def _mutate_sequence(self, sequence, mutation_type):
        """
        Helper method to mutate a sequence (either source or target).
        """
        if mutation_type == 'add' and (self.mutation_obedience == 1 or random.random() < self.mutation_obedience):
            last_item = sequence[-1]
            new_item = self.gene_pool.markov.rand_next(last_item)
            sequence.append(new_item)
        elif mutation_type == 'delete' and random.random() > .5:
            index = random.randint(0, len(sequence) - 1)
            sequence.pop(index)


class MassExtinction(Layer):
    """
    Performs periodic mass extinction events in the population.
    """

    def __init__(self, period: int):
        super().__init__(application_function=self.mass_extinction, selection_function=lambda x: x)
        self.n = 0
        self.period = period

    def mass_extinction(self, individuals: List[Individual]):
        """
        Removes individuals with non-positive fitness every 'period' generations.
        """
        self.n += 1
        if self.n >= self.period:
            self.n = 0
            self.environment.individuals = [ind for ind in individuals if ind.fitness > 0]


class MigrationLayer(Layer):
    """
    Performs migration operations on individuals based on their similarity to other parts of the gene pool.
    """

    def __init__(self, selection: Union[Callable, int], gene_pool: LaGenePool, hard_reset: bool = False,
                 scope_size: int = 4):
        super().__init__(application_function=self.migrate, selection_function=selection)
        self.hard_reset = hard_reset
        self.gene_pool = gene_pool
        self.scope_size = scope_size

    def weighted_position(self, indices, scores, max_index):
        """
        Calculates a weighted position based on indices and scores.
        """
        if len(indices) != len(scores):
            raise ValueError("Indices and scores must be of the same length")
        if max_index == 0:
            raise ValueError("Maximum index value must not be zero")

        normalized_indices = numpy.array(indices) / max_index
        scores = numpy.array(scores)

        total_weight = numpy.sum(scores)
        if total_weight == 0:
            return 0
        weighted_sum = numpy.sum(normalized_indices * scores)
        return weighted_sum / total_weight

    def migrate(self, individuals: List[Individual]):
        """
        Adjusts the positional rank of individuals based on their similarity to other parts of the gene pool.
        """
        total_length = len(self.gene_pool.source_lines)
        for individual in individuals:
            query = ' '.join(individual.item.source)
            ids, scores = self.gene_pool.find_inds(query, self.scope_size)

            suggested_position = self.weighted_position(ids, scores, total_length)
            if self.hard_reset:
                individual.item.positional_rank = suggested_position
            else:
                individual.item.positional_rank = (individual.item.positional_rank + suggested_position) / 2


class AfterLife(Layer):
    def __init__(self, start_at: int, n_best: int, period: int):
        """A positive way to die"""
        super().__init__(application_function=self.save, selection_function=lambda x: x)
        self.n_best = n_best
        self.start_at = start_at
        self.individuals = []
        self.year = 0
        self.period = period

    def save(self, individuals: List[Individual]):
        self.year += 1
        if self.year > self.start_at and self.year % self.period == 0:

            self.environment.individuals = list(sorted(individuals, key=lambda individual: -individual.fitness))
            if len(self.environment.individuals) > self.n_best:
                for i in range(self.n_best):
                    self.individuals.append(self.environment.individuals.pop())

    def bring_back(self):
        self.environment.add_individuals(self.individuals)
        self.individuals = []


