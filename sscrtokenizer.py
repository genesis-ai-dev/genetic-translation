import difflib
import re


def word_level_similarity(str1, str2, window_size=3):
    # Tokenize strings into words
    words1 = re.findall(r'\b\w+\b', str1.lower())
    words2 = re.findall(r'\b\w+\b', str2.lower())

    # Calculate global similarity
    global_matcher = difflib.SequenceMatcher(None, words1, words2)
    global_similarity = global_matcher.ratio()

    # Calculate local similarities using sliding window
    local_similarities = []
    for i in range(len(words1) - window_size + 1):
        for j in range(len(words2) - window_size + 1):
            window1 = words1[i:i + window_size]
            window2 = words2[j:j + window_size]
            matcher = difflib.SequenceMatcher(None, window1, window2)
            local_similarities.append(matcher.ratio())

    # If no local similarities (short strings), use global similarity
    if not local_similarities:
        return global_similarity

    # Combine global and max local similarity
    max_local_similarity = max(local_similarities)
    combined_similarity = (global_similarity + max_local_similarity) / 2

    return combined_similarity


# Example usage
print(word_level_similarity("dog", "bog"))  # Should be 0
print(word_level_similarity("the dog", "a dog"))  # Should be > 0
print(word_level_similarity("the quick brown fox", "the quick brown dog"))  # Higher similarity
print(word_level_similarity("the quick brown fox",
                            "brown quick the fox"))  # Higher than before, less sensitive to global order
print(word_level_similarity("the quick brown fox jumps over the lazy dog",
                            "the fox jumps over the quick brown lazy dog"))  # Should handle longer strings well