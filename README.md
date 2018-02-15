
# Character level language model

A language model is the one where given an input sentence, the model outputs a probability of how correct that sentence is. This is extensively used in speech recognition, sentence generation and machine translation systems where it outputs the sentences that are likely.

Steps to build a language model:

1. Build a training set using a large corpus of english text
2. Tokenize each sentence to build a vocabulary
3. Map each word in the sentence using any encoding mechanism
4. Replace uncommon words with <UNK>, in which case model the chance of the unknown word instead of the specific word.
5. Build an RNN model where output is the softmax probability for each word in the dictionary

### Training a language model
-----------------------------
<img src = "https://raw.githubusercontent.com/tejaslodaya/character-level-language-model/master/images/train.png">

At t time step, RNN is estimating P(y<t>| y<1>,y<2>,…,y<t−1>). Training set is formed in a way where x<2> = y<1> and x<3> = y<2> and so on. In short, the output sentence lags behind the input sentence by one time step. The optimization algorithm followed is always **Stochastic Gradient Descent** (one sequence at a time).

To get probability for a random sequence, break down the joint probability distribution P(y1, y2, y3, ...) as a product of conditionals, P(y1) * P(y2 | y1) * P(y3 | y1, y2).

**NOTE**: In vanilla language model as described above, word is a basic building block.
In character level language model, the basic unit/ lowest level is a character, which makes building a dictionary very easy (finite number of characters)

### Generate new text
---------------------
Once the model is trained, we can sample new text(characters). The process of generation is explained below:
<img src = "https://raw.githubusercontent.com/tejaslodaya/character-level-language-model/master/images/sample.png">

Steps:

1. Pass the network the first "dummy" input x⟨1⟩=0 ⃗ (the vector of zeros). This is the default input before we've generated any characters. We also set a⟨0⟩=0 ⃗
2. Use the probabilities output by the RNN to randomly sample a chosen word (using np.random.choice) for that time-step as y<t>
3. Pass this selected word to the next time-step as x<2> 
 
### Results
----------- 
Some of the names generated:
1. Macaersaurus
2. Edahosaurus
3. Trodonosaurus
4. Ivusanon
5. Trocemitetes

If you observe carefully, our model has learned to capture `saurus`, `don`,`aura`, `tor` at the end of every dinosaur name

### TODO
-------
1. Use LSTM in-place of RNNs with help of Keras

### DIY
-------
Place the training data (dinosaur names) in-place of `dinos.txt`. 
Run `main.py`, which follows 3 steps:
1. Preprocessing the data
2. Building a vocabulary
3. Run the model

To generate the names out-of-the-box, run
`python main.py`

### References
--------------
1. [Inspired by Andrej Karpathy's implementation](https://gist.github.com/karpathy/d4dee566867f8291f086)
2. [Karpathy's blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)