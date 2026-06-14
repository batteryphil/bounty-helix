import sys

sys.path.append('/home/agi/code/helix-agi')

import transformer_attention as ta

model = ta.load_model('helix-agi/models/hermes-3')

print(model)

attention = model.extract_attention('What is the capital of California?')
print(attention)

print(model.extract_attention('What is the largest state by land area in the United States?'))
