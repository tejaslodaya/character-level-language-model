import numpy as np
import random
from app_utils import sample, optimize, model

# Step 1: Preprocessing
data = open('dinos.txt', 'r').read()
data= data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))

# Step 2: Build a vocabulary
ix_to_char = {i:ch for i, ch in enumerate(sorted(chars))}
char_to_ix = {ch:i for i, ch in enumerate(sorted(chars))}

parameters = model(data, ix_to_char, char_to_ix)