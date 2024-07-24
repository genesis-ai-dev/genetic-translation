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


import random
from typing import List, Union, Callable
from Finch.generic import Layer, Individual
from genetics import LaGene, LaGenePool


class NPointLaGeneCrossover(Layer):
    """
    Performs flexible N-point crossover operation on LaGene individuals.
    """

    def __init__(self, parent_selection: Union[Callable, int], families: int, children_per_family: int,
                 gene_pool: LaGenePool, n_points: int = 2):
        super().__init__(application_function=self.crossover_family, selection_function=parent_selection,
                         repeat=families, refit=False)
        self.gene_pool = gene_pool
        self.children_per_family = children_per_family
        self.n_points = n_points

    def crossover_family(self, parents: List[Individual]) -> List[Individual]:
        """
        Performs crossover between two parent individuals to create multiple children.
        """
        parent1, parent2 = parents
        assert parent1.item.__class__ == LaGene and parent2.item.__class__ == LaGene, \
            "Individual Items must be LaGenes"

        children = []
        for _ in range(self.children_per_family):
            child = self.flexible_n_point_crossover(parent1, parent2)
            if child is not None:
                children.append(child)

        self.environment.add_individuals(children)
        return children

    def flexible_n_point_crossover(self, parent1: Individual, parent2: Individual) -> Union[Individual, None]:
        """
        Performs flexible N-point crossover between two parents.
        """
        source1, source2 = parent1.item.source, parent2.item.source
        target1, target2 = parent1.item.target, parent2.item.target

        # Determine the number of crossover points based on sequence lengths
        min_length = min(len(source1), len(source2), len(target1), len(target2))
        max_points = min(min_length - 1, self.n_points)

        if max_points <= 0:
            # If sequences are too short for crossover, randomly choose one parent
            chosen_parent = random.choice([parent1, parent2])
            child_source = chosen_parent.item.source.copy()
            child_target = chosen_parent.item.target.copy()
        else:
            # Perform crossover
            num_points = random.randint(1, max_points)
            crossover_points = sorted(random.sample(range(1, min_length), num_points))

            child_source = []
            child_target = []
            start = 0
            parent_switch = False

            for point in crossover_points + [None]:  # Add None to process the last segment
                if parent_switch:
                    child_source.extend(source2[start:point])
                    child_target.extend(target2[start:point])
                else:
                    child_source.extend(source1[start:point])
                    child_target.extend(target1[start:point])
                start = point
                parent_switch = not parent_switch

        # Calculate new positional rank
        rank1, rank2 = parent1.item.positional_rank, parent2.item.positional_rank
        new_rank = (rank1 + rank2) / 2 + random.uniform(-0.05, 0.05)

        child_lagene = LaGene(source=child_source, target=child_target, shift=0, positional_rank=new_rank)
        child_individual = Individual(child_lagene, parent1.fitness_function)
        child_individual.fit()

        if child_individual.fitness <= max(parent1.fitness, parent2.fitness) or child_individual.fitness <= 0:
            return None

        child_individual.item.name = f"{parent1.item.name}_{parent2.item.name}_child"
        return child_individual

    def __str__(self):
        return f"FlexibleNPointLaGeneCrossover(families={self.repeat}, children_per_family={self.children_per_family}, n_points={self.n_points})"

class LexiconMutation(Layer):
    """
    Performs mutation operations on individuals using a lexicon.
    """

    def __init__(self, selection: Union[Callable, int], gene_pool: LaGenePool, overpowered: bool = False):
        super().__init__(application_function=self.mutate_all, selection_function=selection, refit=not overpowered)
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
        super().__init__(application_function=self.mutate_all, selection_function=selection, refit=not overpowered)
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
            if new_fitness <= old_fitness:
                individual.fitness = old_individual.fitness
                individual.item = old_individual.item

    def _mutate_sequence(self, sequence, mutation_type):
        """
        Helper method to mutate a sequence (either source or target).
        """
        if mutation_type == 'add' and (self.mutation_obedience == 1 or random.random() < self.mutation_obedience):
            if random.random() < .5:
                last_item = sequence[-1]
                new_item = self.gene_pool.markov.rand_next(last_item)
                sequence.append(new_item)
            else:
                first_item = sequence[0]
                new_item = self.gene_pool.markov.rand_prev(first_item)
                sequence.insert(0, new_item)
        elif mutation_type == 'delete' and random.random() > .5:
            index = random.randint(0, len(sequence) - 1)
            sequence.pop(index)


class MassExtinction(Layer):
    """
    Performs periodic mass extinction events in the population.
    """

    def __init__(self, period: int):
        super().__init__(application_function=self.mass_extinction, selection_function=lambda x: x, refit=False)
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
        super().__init__(application_function=self.migrate, selection_function=selection, refit=False)
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
    def __init__(self, start_at: int, n_best: int, period: int, threshold: float = 12):
        """A positive way to die"""
        super().__init__(application_function=self.save, selection_function=lambda x: x, refit=False)
        self.n_best = n_best
        self.start_at = start_at
        self.individuals = []
        self.year = 0
        self.period = period
        self.threshold = threshold
    def save(self, individuals: List[Individual]):
        self.year += 1
        if self.year > self.start_at and self.year % self.period == 0:

            self.environment.individuals = list(sorted(individuals, key=lambda individual: -individual.fitness))
            if len(self.environment.individuals) > self.n_best:
                for i in range(self.n_best):
                    if self.environment.individuals[0].fitness > self.threshold:
                        self.individuals.append(self.environment.individuals.pop(0))

    def bring_back(self):
        self.environment.add_individuals(self.individuals)
        self.individuals = []


