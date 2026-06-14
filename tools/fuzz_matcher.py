import fuzzywuzzy
from fuzzywuzzy import process

def find_best_match(string_list, search_string):
    return process.extractOne(search_string, string_list, scorer=fuzzywuzzy.partial_ratio)

if __name__ == '__main__':
    string_list = ['apple', 'banana', 'cherry', 'date', 'elderberry']
    search_string = 'bana'
    best_match, score = find_best_match(string_list, search_string)
    print(f"Best match: {best_match}, Score: {score}")