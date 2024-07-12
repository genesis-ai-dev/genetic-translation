import random
from collections import defaultdict
from itertools import permutations


def create_markov_chain(text):
    words = text.split()
    markov_chain = defaultdict(lambda: defaultdict(int))

    for i in range(len(words) - 1):
        current_word = words[i]
        next_word = words[i + 1]
        markov_chain[current_word][next_word] += 1

    # Convert counts to probabilities
    for word in markov_chain:
        total = sum(markov_chain[word].values())
        for next_word in markov_chain[word]:
            markov_chain[word][next_word] /= total

    return markov_chain


def generate_sequences(words):
    return list(permutations(words))


def sequence_probability(sequence, markov_chain):
    probability = 1.0
    for i in range(len(sequence) - 1):
        current_word = sequence[i]
        next_word = sequence[i + 1]
        if current_word in markov_chain and next_word in markov_chain[current_word]:
            probability *= markov_chain[current_word][next_word]
        else:
            probability *= 0.0001  # Small probability for unseen transitions
    return probability


def rank_sequences(sequences, markov_chain):
    ranked = [(seq, sequence_probability(seq, markov_chain)) for seq in sequences]
    return sorted(ranked, key=lambda x: x[1], reverse=True)


# Example usage
text = "the quick brown fox jumps over the lazy dog"
words_to_order = ["the", "quick", "brown"]

markov_chain = create_markov_chain(text)
sequences = generate_sequences(words_to_order)
ranked_sequences = rank_sequences(sequences, markov_chain)

print("Top 5 most likely sequences:")
for seq, prob in ranked_sequences[:5]:
    print(f"{' '.join(seq)}: {prob}")