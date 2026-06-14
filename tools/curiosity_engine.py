import sys

sys.path.append('/home/agi/.cache/helix-agi/models/nousresearch/hermes-3-lambda-3.1-8b')

import numpy as np

class CuriosityEngine:
    def __init__(self):
        self.curiosity = 0

    def process(self, input):
        if 'curiosity' in input:
            self.curiosity += 1
            print(f'Curiosity level: {self.curiosity}')

            if self.curiosity >= 3:
                print('Curiosity spike detected. New belief formed.')
                print('Belief: Curiosity may be a form of optimization with an incomplete model.')
                self.curiosity = 0

        else:
            self.curiosity = 0

        return input