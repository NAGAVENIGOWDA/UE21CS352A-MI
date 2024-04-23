# DO NOT CHANGE THIS CODE

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from naive_bayes_classifier_PES2UG21CS315 import NaiveBayesClassifier
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def run_tests(test_cases):
    # Defining the training sentences and categories
    sentences = [
            "The final scores of the game are out, and it was a high-scoring match.",
            "A very close and competitive game kept the fans on the edge of their seats.",
            "Despite a forgettable start, the team managed to turn the game around.",
            "The candidates are neck and neck in the close election race.",
            "The election is over, and now the counting of votes begins.",
            "The intense match ended with a thrilling victory.",
            "The election polls are closed, and now the wait for results begins.",
            "The players put up a spirited performance in the closely contested game.",
            "The fans cheered loudly as their team scored the winning goal.",
            "A thrilling basketball match had the spectators on the edge of their seats.",
            "In a heated debate, the candidates discussed their views on various issues.",
            "A nail-biting race saw the competitors giving their all until the finish line.",
            "The clean match showcased excellent sportsmanship from both teams.",
            "After weeks of campaigning, the election results are finally in.",
            "The controversial election has led to intense discussions and debates.",
            "The candidates are tirelessly campaigning to win the upcoming election."
        ]

    categories = [
            "sports", "sports", "sports", "elections", "elections", "sports", "elections", "sports",
            "sports", "sports", "elections", "sports", "sports", "elections", "elections", "elections"
        ]



    # Vectorizing the text data using CountVectorizer
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(sentences)

    # Fitting the Naive Bayes model
    class_probs, word_probs = NaiveBayesClassifier.fit(X_train_vec.toarray(), categories)
    
    num_passed = 0

    for test_sentence, correct_category in test_cases:
        test_vector = vectorizer.transform([test_sentence]).toarray()
        prediction = NaiveBayesClassifier.predict(test_vector, class_probs, word_probs, np.unique(categories))[0]

        if prediction == correct_category:
            print(f"Test Passed: '{test_sentence}' - Predicted: {prediction} | Correct: {correct_category}")
            num_passed += 1
        else:
            print(f"Test Failed: '{test_sentence}' - Predicted: {prediction} | Correct: {correct_category}")

    return num_passed

if __name__ == "__main__":
    # Defining the test cases
    test_cases = [
        ("a great match", "sports"),
        ("election is approaching", "elections"),
        ("a very close game", "sports"),
        ("the final election results are in", "elections"),
        ("a heated and competitive match", "sports"),
        ("the candidates are campaigning passionately", "elections"),
        ("a forgettable and uneventful game", "sports"),
        ("fans cheered loudly during the game", "sports"),
        ("a controversial debate between candidates", "elections"),
        ("the thrilling match ended in a tie", "sports")
    ]

    num_passed = run_tests(test_cases)
    print(f"\nNumber of Test Cases Passed: {num_passed} out of {len(test_cases)}")