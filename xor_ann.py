# =============================================================
#  XOR using a Simple Neural Network (from scratch)
# =============================================================
#
#  Network Structure:
#    Input(2) --> Hidden(2) --> Output(1)
#
#  The code has 3 functions:
#    initialize()  – set up data, weights, and settings
#    train()       – forward pass + backward pass, repeated many times
#    test()        – check if the network learned XOR correctly
# =============================================================

import numpy as np


# Sigmoid activation function
# Formula:  sigmoid(x) = 1 / (1 + e^(-x))
# It squishes any number into a value between 0 and 1.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of sigmoid (needed for backpropagation)
# Formula:  sigmoid'(x) = x * (1 - x)
#   (when x is already the output of sigmoid)
# This tells us the "slope" — how sensitive the output is to a small change in input.
def sigmoid_derivative(x):
    return x * (1 - x)


# =============================================================
#  FUNCTION 1 — INITIALIZE
#  Sets up training data, weights, biases, and hyperparameters.
#  Returns everything the other functions need.
# =============================================================

def initialize():

    # --- Training Data (XOR truth table) ---

    # 4 input samples: each row is [x1, x2]
    inputs = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    # Correct XOR answers for each input
    # 0 XOR 0 = 0,  0 XOR 1 = 1,  1 XOR 0 = 1,  1 XOR 1 = 0
    expected_output = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    # --- Weights and Biases (random starting values) ---

    np.random.seed(42)  # so results are the same every time

    # Weights from Input Layer to Hidden Layer (shape: 2 inputs × 2 hidden neurons)
    weights_input_to_hidden = np.random.uniform(size=(2, 2))

    # Bias for Hidden Layer (one per hidden neuron)
    bias_hidden = np.random.uniform(size=(1, 2))

    # Weights from Hidden Layer to Output Layer (shape: 2 hidden neurons × 1 output)
    weights_hidden_to_output = np.random.uniform(size=(2, 1))

    # Bias for Output Layer
    bias_output = np.random.uniform(size=(1, 1))

    # --- Hyperparameters ---

    learning_rate = 0.5   # how big of a step we take when updating weights
    epochs = 10000        # how many times we loop through the training data

    # Return everything packed in a dictionary so other functions can use it
    return {
        "inputs": inputs,
        "expected_output": expected_output,
        "weights_input_to_hidden": weights_input_to_hidden,
        "bias_hidden": bias_hidden,
        "weights_hidden_to_output": weights_hidden_to_output,
        "bias_output": bias_output,
        "learning_rate": learning_rate,
        "epochs": epochs
    }


# =============================================================
#  FUNCTION 2 — TRAIN
#  Runs forward propagation and back propagation for each epoch.
#  Updates the weights and biases so the network learns XOR.
# =============================================================

def train(data):

    # Unpack everything from the dictionary
    inputs                   = data["inputs"]
    expected_output          = data["expected_output"]
    weights_input_to_hidden  = data["weights_input_to_hidden"]
    bias_hidden              = data["bias_hidden"]
    weights_hidden_to_output = data["weights_hidden_to_output"]
    bias_output              = data["bias_output"]
    learning_rate            = data["learning_rate"]
    epochs                   = data["epochs"]

    for epoch in range(epochs):

        # ---------------------------------------------------------
        #  FORWARD PROPAGATION  –  "predict an output"
        # ---------------------------------------------------------

        # Hidden Layer
        # Formula:  hidden_input  = (inputs × weights) + bias
        #           hidden_output = sigmoid(hidden_input)
        hidden_input  = np.dot(inputs, weights_input_to_hidden) + bias_hidden
        hidden_output = sigmoid(hidden_input)

        # Output Layer
        # Formula:  output_input = (hidden_output × weights) + bias
        #           predicted    = sigmoid(output_input)
        output_input     = np.dot(hidden_output, weights_hidden_to_output) + bias_output
        predicted_output = sigmoid(output_input)

        # ---------------------------------------------------------
        #  CALCULATE ERROR
        # ---------------------------------------------------------

        # Formula:  error = expected_output - predicted_output
        error = expected_output - predicted_output

        # ---------------------------------------------------------
        #  BACK PROPAGATION  –  "learn from mistakes"
        # ---------------------------------------------------------

        # Output layer gradient
        # Formula:  d_output = error × sigmoid'(predicted_output)
        d_output = error * sigmoid_derivative(predicted_output)

        # Hidden layer error (send the blame backwards)
        # Formula:  error_hidden = d_output × weights_hidden_to_output^T
        error_hidden = d_output.dot(weights_hidden_to_output.T)

        # Hidden layer gradient
        # Formula:  d_hidden = error_hidden × sigmoid'(hidden_output)
        d_hidden = error_hidden * sigmoid_derivative(hidden_output)

        # ---------------------------------------------------------
        #  UPDATE WEIGHTS AND BIASES
        # ---------------------------------------------------------

        # Formula:  new_weight = old_weight + (input^T × gradient) × learning_rate
        # Formula:  new_bias   = old_bias   + sum(gradient) × learning_rate

        weights_hidden_to_output += hidden_output.T.dot(d_output)           * learning_rate
        bias_output              += np.sum(d_output, axis=0, keepdims=True) * learning_rate

        weights_input_to_hidden  += inputs.T.dot(d_hidden)                  * learning_rate
        bias_hidden              += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

        # Print loss every 1000 epochs
        # Formula:  MSE = mean(error²)
        if (epoch + 1) % 1000 == 0:
            loss = np.mean(error ** 2)
            print(f"Epoch {epoch+1:>5}  |  Loss: {loss:.6f}")

    # Save the trained weights back into the dictionary
    data["weights_input_to_hidden"]  = weights_input_to_hidden
    data["bias_hidden"]              = bias_hidden
    data["weights_hidden_to_output"] = weights_hidden_to_output
    data["bias_output"]              = bias_output

    return data


# =============================================================
#  FUNCTION 3 — TEST
#  Runs each input through the trained network and prints results.
# =============================================================

def test(data):

    inputs                   = data["inputs"]
    expected_output          = data["expected_output"]
    weights_input_to_hidden  = data["weights_input_to_hidden"]
    bias_hidden              = data["bias_hidden"]
    weights_hidden_to_output = data["weights_hidden_to_output"]
    bias_output              = data["bias_output"]

    print("\nTraining complete!\n")
    print("Input   | Predicted | Rounded | Expected")
    print("-" * 45)

    for i in range(len(inputs)):
        # Forward pass with trained weights (same formula as training)
        # Formula:  hidden = sigmoid(input × weights_input_to_hidden + bias_hidden)
        #           output = sigmoid(hidden × weights_hidden_to_output + bias_output)
        hidden_output    = sigmoid(np.dot(inputs[i], weights_input_to_hidden) + bias_hidden)
        predicted_output = sigmoid(np.dot(hidden_output, weights_hidden_to_output) + bias_output)

        prediction       = predicted_output[0][0]   # raw decimal prediction
        rounded_result   = round(prediction)         # round to 0 or 1
        expected_value   = expected_output[i][0]     # correct answer

        print(f"{inputs[i]}   |   {prediction:.4f}  |    {rounded_result}    |    {expected_value}")


# =============================================================
#  RUN EVERYTHING
# =============================================================

data = initialize()   # PART 1: set up inputs, weights, settings
data = train(data)    # PART 2: train the network
test(data)            # PART 3: test the trained network
