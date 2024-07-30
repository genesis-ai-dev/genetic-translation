from Finch.selectors import Select
from Finch.generic import Individual
import random
from typing import Callable, List, Tuple, Union
from Finch.selectors import Select
from Finch.generic import Individual
import random
from typing import Callable, List, Union

class PositionalRankSelection(Select):
    """
    Combined positional and rank-based selection strategy.

    Parameters:
    - amount: The number of individuals to select.
    - percent: The percentage of individuals to select (alternative to amount).
    - position_weight: Weight given to positional selection (0 to 1).
    - rank_weight: Weight given to rank-based selection (0 to 1).
    - temperature: A factor that controls the randomness of selection.

    Usage:
    ```
    selector = PositionalRankSelection(amount=10, position_weight=0.6, rank_weight=0.4, temperature=0.1)
    selected_individuals = selector.select(individuals)
    ```
    """

    def __init__(self,
                 amount: Union[int, Callable] = None,
                 percent: Union[float, Callable] = None,
                 position_weight: float = 0.5,
                 rank_weight: float = 0.5,
                 temperature: Union[float, Callable] = 1.0):
        super().__init__(percent_to_select=percent, amount_to_select=amount)
        self.position_weight = position_weight
        self.rank_weight = rank_weight
        self.temperature = temperature if callable(temperature) else lambda: temperature

        if abs(self.position_weight + self.rank_weight - 1.0) > 1e-6:
            raise ValueError("The sum of position_weight and rank_weight must be 1.0")

    def select(self, individuals: List[Individual]) -> List[Individual]:
        """
        Select individuals based on their positional_rank and fitness rank.

        Parameters:
        - individuals: List of individuals to select from.

        Returns:
        - list[Individual]: Selected individuals.
        """
        if not individuals:
            return []

        if self.percent_to_select is not None:
            amount = int(self.percent_to_select() * len(individuals))
        else:
            amount = self.amount_to_select()

        # Sort individuals by positional_rank and fitness
        sorted_by_position = sorted(individuals, key=lambda ind: ind.item.positional_rank)
        sorted_by_fitness = sorted(individuals, key=lambda ind: ind.fitness, reverse=True)

        # Calculate selection probabilities
        position_ranks = {ind: i for i, ind in enumerate(sorted_by_position)}
        fitness_ranks = {ind: i for i, ind in enumerate(sorted_by_fitness)}

        temp = self.temperature()
        total_individuals = len(individuals)

        probabilities = []
        for ind in individuals:
            position_prob = 1 / (1 + temp * position_ranks[ind])
            fitness_prob = 1 / (1 + temp * fitness_ranks[ind])
            combined_prob = (self.position_weight * position_prob +
                             self.rank_weight * fitness_prob)
            probabilities.append(combined_prob)

        # Normalize probabilities
        total = sum(probabilities)
        normalized_probabilities = [p / total for p in probabilities]

        # Select individuals
        selected_individuals = random.choices(individuals, weights=normalized_probabilities, k=amount)

        return selected_individuals

    def __str__(self):
        return (f"PositionalRankSelection(amount={self.amount_to_select}, "
                f"percent={self.percent_to_select}, "
                f"position_weight={self.position_weight}, "
                f"rank_weight={self.rank_weight}, "
                f"temperature={self.temperature})")

class PositionalSelection(Select):
    """
    Positional selection strategy based on individual's positional_rank.

    Parameters:
    - amount: The number of individuals to select.
    - percent: The percentage of individuals to select (alternative to amount).
    - temperature: A factor that controls the randomness of selection.
                   Higher values increase randomness, lower values make selection more deterministic.

    Usage:
    ```
    selector = PositionalSelection(amount=10, temperature=0.1)
    selected_individuals = selector.select(individuals)
    ```
    """

    def __init__(self, amount: Union[int, Callable] = None,
                 percent: Union[float, Callable] = None,
                 temperature: Union[float, Callable] = 1.0):
        super().__init__(percent_to_select=percent, amount_to_select=amount)
        self.temperature = temperature if callable(temperature) else lambda: temperature

    def select(self, individuals: List[Individual]) -> List[Individual]:
        """
        Select individuals based on their positional_rank.

        Parameters:
        - individuals: List of individuals to select from.

        Returns:
        - list[Individual]: Selected individuals.
        """
        if not individuals:
            return []

        if self.percent_to_select is not None:
            amount = int(self.percent_to_select() * len(individuals))
        else:
            amount = self.amount_to_select()

        # Sort individuals by positional_rank
        sorted_individuals = sorted(individuals, key=lambda ind: ind.item.positional_rank)

        # Calculate selection probabilities
        ranks = [i for i in range(len(sorted_individuals))]
        temp = self.temperature()
        probabilities = [1 / (1 + temp * rank) for rank in ranks]

        # Normalize probabilities
        total = sum(probabilities)
        normalized_probabilities = [p / total for p in probabilities]

        # Select individuals
        selected_individuals = random.choices(sorted_individuals, weights=normalized_probabilities, k=amount)

        return selected_individuals

    def __str__(self):
        return f"PositionalSelection(amount={self.amount_to_select}, percent={self.percent_to_select}, temperature={self.temperature})"
