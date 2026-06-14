import json
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

class MetricsAnalysis:
    """
    A tool to analyze performance metrics and provide actionable insights for improvement.
    """
    def __init__(self, responses):
        self.responses = responses
        self.vectorizer = CountVectorizer()
        self.model = LatentDirichletAllocation(n_components=5, random_state=0)
        self.similarity_threshold = 0.6

    def preprocess(self):
        """
        Preprocess the responses by vectorizing them.
        """
        self.vectorized_responses = self.vectorizer.fit_transform(self.responses)

    def analyze_patterns(self):
        """
        Analyze the responses to identify patterns and areas of strength/weakness.
        """
        word_counts = Counter()
        for response in self.responses:
            word_counts.update(response.split())

        most_common_words = word_counts.most_common(10)
        print("Most common words:")
        for word, count in most_common_words:
            print(f"{word}: {count}")

        print("\nResponse patterns:")
        self.model.fit(self.vectorized_responses)
        topic_counts = self.model.transform(self.vectorized_responses)
        for i in range(len(self.responses)):
            topic_distribution = dict(zip(self.model.components_, topic_counts[i]))
            most_likely_topic = max(topic_distribution, key=topic_distribution.get)
            print(f"Response {i+1}: Most likely topic {most_likely_topic}")

    def find_similar_responses(self):
        """
        Find responses with high similarity to each other.
        """
        similarity_matrix = cosine_similarity(self.vectorized_responses)
        print("\nSimilar responses:")
        for i in range(len(self.responses)):
            similar_responses = [i for i in range(len(self.responses)) if similarity_matrix[i].max() > self.similarity_threshold]
            print(f"Response {i+1} is similar to responses: {', '.join(map(str, similar_responses))}")

    def provide_insights(self):
        """
        Provide actionable insights and recommendations for improvement.
        """
        print("\nInsights and recommendations:")
        print("1. Focus on diversifying your responses by using less common words.")
        print("2. Vary your response patterns by discussing different topics.")
        print("3. Avoid repeating similar responses, as they may not provide unique insights.")

    def run(self):
        """
        Run the metrics analysis tool.
        """
        self.preprocess()
        self.analyze_patterns()
        self.find_similar_responses()
        self.provide_insights()

def main():
    with open("responses.json") as file:
        responses = json.load(file)["responses"]

    metrics_analysis = MetricsAnalysis(responses)
    metrics_analysis.run()

if __name__ == "__main__":
    main()