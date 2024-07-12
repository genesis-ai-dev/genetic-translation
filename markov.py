import math
import random

import numpy as np
from collections import defaultdict
from typing import List, Tuple


class BiMarkovChain:
    def __init__(self):
        self.fwd_chain = defaultdict(lambda: defaultdict(int))
        self.bwd_chain = defaultdict(lambda: defaultdict(int))
        self.items = set()

    def add_seq(self, sequence):
        for i in range(len(sequence) - 1):
            curr, next_item = sequence[i], sequence[i + 1]
            self.fwd_chain[curr][next_item] += 1
            self.bwd_chain[next_item][curr] += 1
            self.items.update([curr, next_item])

    def _build_matrices(self):
        self.item_to_idx = {item: idx for idx, item in enumerate(self.items)}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        n = len(self.items)

        self.fwd_matrix = np.zeros((n, n))
        self.bwd_matrix = np.zeros((n, n))

        for curr, nexts in self.fwd_chain.items():
            for next_item, count in nexts.items():
                i, j = self.item_to_idx[curr], self.item_to_idx[next_item]
                self.fwd_matrix[i, j] = count
                self.bwd_matrix[j, i] = count

        # Normalize
        self.fwd_matrix /= np.sum(self.fwd_matrix, axis=1, keepdims=True)
        self.bwd_matrix /= np.sum(self.bwd_matrix, axis=1, keepdims=True)

        # Replace NaNs with 0
        self.fwd_matrix = np.nan_to_num(self.fwd_matrix)
        self.bwd_matrix = np.nan_to_num(self.bwd_matrix)

    def next(self, item):
        if not hasattr(self, 'fwd_matrix'):
            self._build_matrices()
        idx = self.item_to_idx[item]
        next_idx = np.argmax(self.fwd_matrix[idx])
        return self.idx_to_item[next_idx]

    def prev(self, item):
        if not hasattr(self, 'bwd_matrix'):
            self._build_matrices()
        idx = self.item_to_idx[item]
        prev_idx = np.argmax(self.bwd_matrix[idx])
        return self.idx_to_item[prev_idx]

    def rand_next(self, item):
        if not hasattr(self, 'fwd_matrix'):
            self._build_matrices()
        try:
            idx = self.item_to_idx[item]
        except KeyError:
            return ''
        if sum(self.fwd_matrix[idx]) < 1:
            return ''
        next_idx = random.choices(range(len(self.items)), weights=self.fwd_matrix[idx])[0]
        return self.idx_to_item[next_idx]

    def rand_prev(self, item):
        if not hasattr(self, 'bwd_matrix'):
            self._build_matrices()
        try:
            idx = self.item_to_idx[item]
            prev_idx = np.random.choice(len(self.items), p=self.bwd_matrix[idx])
        except:
            return ''
        return self.idx_to_item[prev_idx]

    def generate_sequences(self, words):
        return list(permutations(words))

    def sequence_probability(self, sequence):
        if not hasattr(self, 'fwd_matrix'):
            self._build_matrices()

        probability = 1.0
        for i in range(len(sequence) - 1):
            current_word = sequence[i]
            next_word = sequence[i + 1]
            try:
                curr_idx = self.item_to_idx[current_word]
                next_idx = self.item_to_idx[next_word]
                probability *= self.fwd_matrix[curr_idx, next_idx]
            except KeyError:
                probability *= 0.0001  # Small probability for unseen transitions
        return probability

    def rank_sequences(self, sequences):
        ranked = [(seq, self.sequence_probability(seq)) for seq in sequences]
        return sorted(ranked, key=lambda x: x[1], reverse=True)

    def order_words(self, words, top_n=5):
        sequences = self.generate_sequences(words)
        ranked_sequences = self.rank_sequences(sequences)
        return ranked_sequences[:top_n]


class AutoLexicon:
    def __init__(self, source: List[List[str]], target: List[List[str]]):
        if len(source) != len(target):
            raise ValueError("Source and target lists must have the same length")

        self.source = source
        self.target = target
        self.word_pairs = defaultdict(lambda: defaultdict(float))
        self.source_word_counts = defaultdict(int)

        self._build_word_pairs()
        self._post_process_word_pairs()

    def _build_word_pairs(self):
        for src_sentence, tgt_sentence in zip(self.source, self.target):
            unique_src_words = set(src_sentence)
            unique_tgt_words = set(tgt_sentence)

            for src_word in unique_src_words:
                self.source_word_counts[src_word] += 1
                for tgt_word in unique_tgt_words:
                    self.word_pairs[src_word][tgt_word] += 1

    def _post_process_word_pairs(self):
        # Find the maximum score for each source and target word
        src_max_scores = {src: max(tgt_dict.values()) for src, tgt_dict in self.word_pairs.items()}
        tgt_max_scores = defaultdict(float)
        for src_dict in self.word_pairs.values():
            for tgt, score in src_dict.items():
                tgt_max_scores[tgt] = max(tgt_max_scores[tgt], score)

        # Apply the subtraction
        for src_word, tgt_dict in self.word_pairs.items():
            for tgt_word, score in tgt_dict.items():
                src_max = src_max_scores[src_word]
                tgt_max = tgt_max_scores[tgt_word]

                if src_max > score and tgt_max > score:
                    # Both words have higher-scoring partners
                    subtract = max(src_max, tgt_max)
                elif src_max > score or tgt_max > score:
                    # Only one word has a higher-scoring partner
                    subtract = min(src_max, tgt_max)
                else:
                    # Neither word has a higher-scoring partner
                    subtract = 0

                new_score = max(0, score - subtract)
                self.word_pairs[src_word][tgt_word] = new_score

    def get_occurrences(self, source_word: str) -> List[Tuple[str, float]]:
        if source_word not in self.word_pairs:
            return []

        occurrences = self.word_pairs[source_word]
        ranked_occurrences = sorted(occurrences.items(), key=lambda x: x[1], reverse=True)

        return ranked_occurrences

    def get_occurrences_from_target(self, target_word: str) -> List[Tuple[str, float]]:
        occurrences = []
        for src_word, tgt_dict in self.word_pairs.items():
            if target_word in tgt_dict and tgt_dict[target_word] > 0:
                occurrences.append((src_word, tgt_dict[target_word]))

        ranked_occurrences = sorted(occurrences, key=lambda x: x[1], reverse=True)

        return ranked_occurrences

    def get_most_frequent_pairs(self) -> List[Tuple[str, str, float]]:
        all_pairs = []
        for src_word, tgt_words in self.word_pairs.items():
            for tgt_word, score in tgt_words.items():
                if score > 0:  # Only include pairs with positive scores
                    all_pairs.append((src_word, tgt_word, score))

        # Sort by score in descending order
        ranked_pairs = sorted(all_pairs, key=lambda x: x[2], reverse=True)

        return ranked_pairs

    def get_random_pair(self, temperature=1.0):
        all_pairs = []
        scores = []

        for src_word, tgt_dict in self.word_pairs.items():
            for tgt_word, score in tgt_dict.items():
                if score > 0:
                    all_pairs.append((src_word, tgt_word))
                    scores.append(score)

        if not all_pairs:
            return None

        # Apply softmax with temperature
        exp_scores = [math.exp(score / temperature) for score in scores]
        sum_exp_scores = sum(exp_scores)
        probabilities = [exp_score / sum_exp_scores for exp_score in exp_scores]

        # Choose a random pair based on the calculated probabilities
        chosen_pair = random.choices(all_pairs, weights=probabilities, k=1)[0]

        return chosen_pair


if __name__ == "__main__":
    source = [
        ["the", "cat", "is", "black"],
        ["the", "dog", "is", "white"],
        ["the", "cat", "and", "dog", "are", "animals"],
        ["I", "love", "to", "eat", "apples"],
        ["she", "reads", "a", "book", "every", "day"],
        ["the", "sun", "is", "shining", "brightly"],
        ["we", "went", "to", "the", "park", "yesterday"],
        ["he", "plays", "football", "on", "weekends"],
        ["the", "children", "are", "playing", "in", "the", "garden"],
        ["I", "drink", "coffee", "every", "morning"]
    ]

    target = [
        ["le", "chat", "est", "noir"],
        ["le", "chien", "est", "blanc"],
        ["le", "chat", "et", "le", "chien", "sont", "des", "animaux"],
        ["j'aime", "manger", "des", "pommes"],
        ["elle", "lit", "un", "livre", "tous", "les", "jours"],
        ["le", "soleil", "brille", "fort"],
        ["nous", "sommes", "allés", "au", "parc", "hier"],
        ["il", "joue", "au", "football", "les", "weekends"],
        ["les", "enfants", "jouent", "dans", "le", "jardin"],
        ["je", "bois", "du", "café", "tous", "les", "matins"]
    ]

    lexicon = AutoLexicon(source, target)
    most_frequent_pairs = lexicon.get_most_frequent_pairs()

    # Print all pairs with non-zero scores
    print("Word pairs with non-zero scores:")
    for src, tgt, score in most_frequent_pairs:
        print(f"{src} - {tgt}: {score:.2f}")

    # Test the get_occurrences method for a few words
    test_words = ["le", "is", "cat", "dog", "to"]
    for word in test_words:
        print(f"\nOccurrences for '{word}':")
        occurrences = lexicon.get_occurrences(word)
        for tgt, score in occurrences:
            if score > 0:
                print(f"  {tgt}: {score:.2f}")

    print("\nRandom pairs (with higher temperature, more uniform distribution):")
    for _ in range(10):
        pair = lexicon.get_random_pair(temperature=1.5)
        if pair:
            src, tgt = pair
            score = lexicon.word_pairs[src][tgt]
            print(f"{src} - {tgt}: {score:.2f}")
