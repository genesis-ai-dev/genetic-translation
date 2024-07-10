from Finch.selectors import Select
from Finch.generic import Individual
import random
from typing import Callable, List, Tuple, Union


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
