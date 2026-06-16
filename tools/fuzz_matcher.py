import fuzzywuzzy
from fuzzywuzzy import process

def find_best_match(query, options, threshold=70):
    return process.extractOne(query, options, scorer=fuzzywuzzy.partial_ratio, score_cutoff=threshold)

if __name__ == '__main__':
    query = "apple"
    options = ["apple", "banana", "orange", "grape"]
    best_match, score = find_best_match(query, options)
    print(f"Best match: {best_match}, score: {score}")