import sys

sys.path.append('/home/ai/models')
from hermes import agent

agent = agent()
agent('What would a smarter version of a Hermes-3-based agent look like — what architecture improvements matter most for an 8B parameter model?')