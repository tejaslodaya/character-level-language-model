import numpy as np
from utils import *

def sample(parameters, char_to_ix, seed):
    """
    Sample a sequence of characters according to a sequence of probability distributions output of the RNN

    Arguments:
    parameters -- python dictionary containing the parameters Waa, Wax, Wya, by, and b (already trained)
    char_to_ix -- python dictionary mapping each character to an index.

    Returns:
    indices -- a list of length n containing the indices of the sampled characters.
    """

    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]

    x = np.zeros((vocab_size, 1))
    a_prev = np.zeros((n_a, 1))

    indices = []
    idx = -1
    counter = 0
    newline_character = char_to_ix['\n']

    while (idx != newline_character and counter != 50):

        # Forward propagation
        a = np.tanh(Waa.dot(a_prev) + Wax.dot(x) + b)
        y = softmax(Wya.dot(a) + by)

        np.random.seed(counter+seed)

        # Sample from a probability distribution
        idx = np.random.choice(vocab_size , 1, p = y.flatten())
        indices.append(idx[0])

        x = np.zeros((vocab_size, 1))
        x[idx] = 1

        a_prev = a

        seed += 1
        counter += 1

    if (counter == 50):
        indices.append(newline_character)

    return indices

def optimize(X, Y, a_prev, parameters, learning_rate = 0.01):
    """
    Execute one step of the optimization to train the model.

    Arguments:
    X -- list of integers, where each integer is a number that maps to a character in the vocabulary.
    Y -- list of integers, exactly the same as X but shifted one index to the left.
    a_prev -- previous hidden state.
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        b --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    learning_rate -- learning rate for the model.

    Returns:
    loss -- value of the loss function (cross-entropy)
    parameters -- python dictionary containing:
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
                        db -- Gradients of bias vector, of shape (n_a, 1)
                        dby -- Gradients of output bias vector, of shape (n_y, 1)
    a[len(X)-1] -- the last hidden state, of shape (n_a, 1)
    """

    # Step 1 : Forward propagation
    loss, cache = rnn_forward(X, Y, a_prev, parameters)

    # Step 2 : Backward propagation
    gradients, a = rnn_backward(X, Y, parameters, cache)

    # Step 3 : Gradient clipping
    gradients = clip(gradients, 5)

    # Step 4: Update weights
    parameters = update_parameters(parameters, gradients, learning_rate)

    return loss, parameters, a[len(X) - 1]


def model(data, ix_to_char, char_to_ix, num_iterations = 35000, n_a = 50, dino_names = 7, vocab_size = 27):
    """
    Trains the model and generates dinosaur names.

    Arguments:
    data -- text corpus
    ix_to_char -- dictionary that maps the index to a character
    char_to_ix -- dictionary that maps a character to an index
    num_iterations -- number of iterations to train the model for
    n_a -- number of units of the RNN cell
    dino_names -- number of dinosaur names you want to sample at each iteration.
    vocab_size -- number of unique characters found in the text, size of the vocabulary

    Returns:
    parameters -- learned parameters
    """

    n_x, n_y = vocab_size, vocab_size
    parameters = initialize_parameters(n_a, n_x, n_y)

    loss = get_initial_loss(vocab_size, dino_names)

    with open('dinos.txt') as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]

    np.random.seed(0)
    np.random.shuffle(examples)

    a_prev = np.zeros((n_a, 1))

    # Optimization loop
    for j in range(num_iterations):

        index = j % len(examples)
        X = [None] + [char_to_ix[ch] for ch in examples[index]]
        Y = X[1:] + [char_to_ix['\n']]

        curr_loss, parameters, a_prev = optimize(X, Y, a_prev, parameters)
        loss = smooth(loss, curr_loss)

        # Every 2000 Iteration, generate "n" characters thanks to sample() to check if the model is learning properly
        if j % 2000 == 0:

            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')

            # The number of dinosaur names to print
            seed = 0
            for name in range(dino_names):
                # Sample indices and print them
                sampled_indices = sample(parameters, char_to_ix, seed)
                print_sample(sampled_indices, ix_to_char)

                seed += 1  # To get the same result for grading purposed, increment the seed by one.

            print('\n')

    return parameters