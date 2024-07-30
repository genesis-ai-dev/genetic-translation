from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from genetic_translator import TranslationEvolver
import os


class TranslationSystem:
    def __init__(self, complete_source_file, complete_target_file):
        self.complete_source_file = complete_source_file
        self.complete_target_file = complete_target_file
        self.complete_source = self._load_file(complete_source_file)
        self.complete_target = self._load_file(complete_target_file)
        self._validate_files()

    def _load_file(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return f.readlines()

    def _validate_files(self):
        assert len(self.complete_source) == len(self.complete_target), \
            "Source and target files must have the same number of lines"

    def find_similar_sentences(self, query, n=10):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(self.complete_source + [query])
        similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        return similarities.argsort()[0][-n:][::-1]

    def prepare_translation_data(self, query, n=15):
        similar_indices = self.find_similar_sentences(query, n)

        with open('temp_source.txt', 'w', encoding='utf-8') as f_source, \
                open('temp_target.txt', 'w', encoding='utf-8') as f_target:
            for idx in similar_indices:
                f_source.write(self.complete_source[idx])
                f_target.write(self.complete_target[idx])

    def process_sentence(self, sentence, **evolver_kwargs):
        print(f"Processing: {sentence}")

        self.prepare_translation_data(sentence, n=100)

        evolver = TranslationEvolver('temp_source.txt', 'temp_target.txt', sentence, **evolver_kwargs)
        evolver.run()

        # Clean up temporary files
        os.remove('temp_source.txt')
        os.remove('temp_target.txt')

        print("\n" + "=" * 50 + "\n")

    def process_sentences(self, sentences, **evolver_kwargs):
        for sentence in sentences:
            self.process_sentence(sentence, **evolver_kwargs)

        print("All sentences processed.")

# Usage example:
ts = TranslationSystem('corpus/eng.txt', 'corpus/de.txt')
sentences_to_translate = [
    "then Jesus went up the mountain and prayed".lower(),
    "then moses said to the people".lower(),
]
ts.process_sentences(sentences_to_translate)