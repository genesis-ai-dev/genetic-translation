import copy
import random

import numpy

from Finch.generic import Layer, Individual
from genetics import LaGene, LaGenePool
from typing import Callable, Union, List


class SimpleLaGeneCrossover(Layer):
    def __init__(self, parent_selection: Union[Callable, int], total_children: int, gene_pool: LaGenePool):
        super().__init__(application_function=self.crossover, selection_function=parent_selection,
                         repeat=total_children, refit=False)
        self.gene_pool = gene_pool

    def crossover(self, individuals: List[Individual]) -> Individual:
        parent1, parent2 = individuals
        assert parent1.item.__class__ == LaGene and parent2.item.__class__ == LaGene, \
            "Individual Items must be LaGenes"

        source1, source2 = parent1.item.source, parent2.item.source
        target1, target2 = parent1.item.target, parent2.item.target

        rank1, rank2 = parent1.item.positional_rank, parent2.item.positional_rank

        source = copy.deepcopy(random.choice([source1, source2]))
        target = copy.deepcopy(random.choice([target1, target2]))

        child_lagene = LaGene(source=source, target=target, shift=0, positional_rank=max([rank1, rank2]) + .05)
        # self.environment.individuals = [ind for ind in self.environment.individuals if ind != parent1 and ind != parent1]
        child_lagene = Individual(child_lagene, parent1.fitness_function)
        child_lagene.fit()
        if child_lagene.fitness < max(parent1.fitness, parent2.fitness) or child_lagene.fitness <= 0:  #
            return
        parent2.fit()
        parent1.fit()
        self.environment.add_individuals([child_lagene])

        return child_lagene


class LexiconMutation(Layer):
    def __init__(self, selection: Union[Callable, int], gene_pool: LaGenePool, overpowered: bool = False):
        super().__init__(application_function=self.mutate_all, selection_function=selection, refit=False)
        self.gene_pool = gene_pool
        self.overpowered = overpowered

    def mutate_all(self, individuals: List[Individual]):
        for individual in individuals:
            self.mutate(individual)

    def mutate_source_target(self, individual: Individual):
        choice = random.choice(['source'])

        source = individual.item.source
        target = individual.item.target
        index = random.randint(0, len(source) - 1)
        try:
            if choice == 'source':

                new_sources = self.gene_pool.AutoLexicon.get_occurrences_from_target(target[index])
                # select a random source from the list of tuples using score as weights (word, score)
                try:
                    new_source = random.choices(
                        [word for word, score in new_sources],
                        weights=[score for word, score in new_sources],
                        k=1
                    )[0]
                except:
                    return
                individual.item.source[index] = new_source

            if choice == 'target':
                new_targets = self.gene_pool.AutoLexicon.get_occurrences(source[index])
                # select a random target from the list of tuples using score as weights (word, score)
                new_target = random.choices(
                    [word for word, score in new_targets],
                    weights=[score for word, score in new_targets],
                    k=1
                )[0]
                individual.item.target[index] = new_target
        except IndexError:
            pass

    def mutate_rank(self, individual: Individual):
        individual.item.positional_rank += random.choice([-.02, .02])  # TODO: does this need to be smoother?

    def mutate_shift(self, individual: Individual):
        shift_shift = random.choice([-1, 1])
        individual.item.shift = shift_shift

    def mutate(self, individual: Individual):
        mutation_type = random.choice(["rank", "text"])

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
    def __init__(self, selection: Union[Callable, int], gene_pool: LaGenePool, overpowered: bool = False):
        super().__init__(application_function=self.mutate_all, selection_function=selection, refit=False)
        self.gene_pool = gene_pool
        self.overpowered = overpowered

    def mutate_all(self, individuals: List[Individual]):
        for individual in individuals:
            self.mutate(individual)

    def mutate(self, individual: Individual):

        if self.overpowered:
            old_individual = individual.copy()
            old_fitness = old_individual.fitness

        t = random.choice(['add', 'delete'])
        if max(len(individual.item.source), len(individual.item.target)) > 3:
            t = 'delete'
        if len(individual.item.source) == 1 or len(individual.item.target) == 1:
            t = 'add'

        if t == 'add':
            last_item = individual.item.source[-1]
            new_item = self.gene_pool.markov.rand_next(last_item)
            individual.item.source += [new_item]
        else:
            if random.random() > .5:
                index = random.randint(0, len(individual.item.source) - 1)
                individual.item.source.pop(index)

        if t == 'add':
            last_item = individual.item.target[-1]
            new_item = self.gene_pool.markov.rand_next(last_item)
            individual.item.target += [new_item]
        else:
            if random.random() > .5:
                index = random.randint(0, len(individual.item.target) - 1)
                individual.item.target.pop(index)

        if self.overpowered:
            new_fitness = individual.fit()
            if new_fitness < old_fitness:
                individual.fitness = old_individual.fitness
                individual.item = old_individual.item


class MassExtinction(Layer):
    def __init__(self, period: int):
        super().__init__(application_function=self.mass_extinction, selection_function=lambda x: x)
        self.n = 0
        self.period = period

    def mass_extinction(self, individuals: List[Individual]):
        self.n += 1
        if self.n >= self.period:
            self.n = 0
            self.environment.individuals = [ind for ind in individuals if ind.fitness > 0]


class MigrationLayer(Layer):
    def __init__(self, selection: Union[Callable, int], gene_pool: LaGenePool, hard_reset: bool = False,
                 scope_size: int = 4):
        super().__init__(application_function=self.migrate, selection_function=selection)
        self.hard_reset = hard_reset
        self.gene_pool = gene_pool
        self.scope_size = scope_size

    def weighted_position(self, indices, scores, max_index):
        if len(indices) != len(scores):
            raise ValueError("Indices and scores must be of the same length")

        if max_index == 0:
            raise ValueError("Maximum index value must not be zero")

        # Normalize indices to 0-1 range
        normalized_indices = numpy.array(indices) / max_index
        scores = numpy.array(scores)

        # Compute weighted average of normalized indices using scores as weights
        total_weight = numpy.sum(scores)
        if total_weight == 0:
            return 0
        weighted_sum = numpy.sum(normalized_indices * scores)
        return weighted_sum / total_weight
    def migrate(self, individuals: List[Individual]):
        total_length = len(self.gene_pool.source_lines)
        for individual in individuals:
            query = ' '.join(individual.item.source)
            ids, scores = self.gene_pool.find_inds(query, self.scope_size)

            suggested_position = self.weighted_position(ids, scores, total_length)
            if self.hard_reset:
                individual.item.positional_rank = suggested_position
            else:
                individual.item.positional_rank = (individual.item.positional_rank + suggested_position) / 2

