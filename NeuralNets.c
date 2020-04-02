/*
        Starter code: F.J.E. Feb. 16
*/
#include "NeuralNets.h"

int train_1layer_net(double sample[INPUTS], int label,
                     double (*sigmoid)(double input),
                     double weights_io[INPUTS][OUTPUTS]) {
  double activations[OUTPUTS];
  feedforward_1layer(sample, sigmoid, weights_io, activations);
  backprop_1layer(sample, activations, sigmoid, label, weights_io);

  return classify_1layer(sample, label, sigmoid, weights_io);
}

int classify_1layer(double sample[INPUTS], int label,
                    double (*sigmoid)(double input),
                    double weights_io[INPUTS][OUTPUTS]) {
  double activations[OUTPUTS];
  // recaluate weights after backprop update if training
  // caluate weights if just classifying
  feedforward_1layer(sample, sigmoid, weights_io, activations);

  return findMax(activations);
}

void feedforward_1layer(double sample[785], double (*sigmoid)(double input),
                        double weights_io[INPUTS][OUTPUTS],
                        double activations[OUTPUTS]) {
  double weight;
  // for each neuron in the next layer
  for (int i = 0; i < OUTPUTS; i++) {
    weight = 0;
    // sum weights in current layers
    for (int j = 0; j < INPUTS; j++) weight += sample[j] * weights_io[j][i];
    // run weight through activation function and see if its LIT!
    activations[i] = sigmoid(weight * SIGMOID_SCALE);
  }
}

void backprop_1layer(double sample[INPUTS], double activations[OUTPUTS],
                     double (*sigmoid)(double input), int label,
                     double weights_io[INPUTS][OUTPUTS]) {
  double error, target[OUTPUTS];

  for (int i = 0; i < OUTPUTS; i++)
    target[i] =
        isTanh(sigmoid) ? (i == label ? 0.6 : -0.6) : (i == label ? 0.8 : 0.2);

  for (int i = 0; i < OUTPUTS; i++) {
    // calculate the error
    // formula for logistic is: f(x)(1 - f(x)) * (Target - observation) * Weight
    // a to b formula for tanh is: 1 - f(x)^2 * (Target - observation) * Weight
    // a to b activations is the result of the activation function f(x)

    // calcuate the derivative of the activation function
    error = (isTanh(sigmoid) ? 1.0 : activations[i]) - pow(activations[i], 2);
    // multiple by derivative of error squared
    error *= (target[i] - activations[i]);
    // update the weights with error with respectively to input
    for (int j = 0; j < INPUTS; j++)
      weights_io[j][i] += ALPHA * error * sample[j];
  }
}

int train_2layer_net(double sample[INPUTS], int label,
                     double (*sigmoid)(double input), int units,
                     double weights_ih[INPUTS][MAX_HIDDEN],
                     double weights_ho[MAX_HIDDEN][OUTPUTS]) {
  double activations[OUTPUTS], h_activations[units];
  feedforward_2layer(sample, sigmoid, weights_ih, weights_ho, h_activations,
                     activations, units);
  backprop_2layer(sample, h_activations, activations, sigmoid, label,
                  weights_ih, weights_ho, units);

  return classify_2layer(sample, label, sigmoid, units, weights_ih, weights_ho);
}

int classify_2layer(double sample[INPUTS], int label,
                    double (*sigmoid)(double input), int units,
                    double weights_ih[INPUTS][MAX_HIDDEN],
                    double weights_ho[MAX_HIDDEN][OUTPUTS]) {
  int max_idx = 0;
  double activations[OUTPUTS], h_activations[units];

  feedforward_2layer(sample, sigmoid, weights_ih, weights_ho, h_activations,
                     activations, units);
  return findMax(activations);
}

void feedforward_2layer(double sample[INPUTS], double (*sigmoid)(double input),
                        double weights_ih[INPUTS][MAX_HIDDEN],
                        double weights_ho[MAX_HIDDEN][OUTPUTS],
                        double h_activations[MAX_HIDDEN],
                        double activations[OUTPUTS], int units) {
  double sum;

  // Feedforward for sample into hidden layers
  for (int i = 0; i < units; i++) {
    sum = 0;
    // take input and and pass it to the hidden layer
    for (int j = 0; j < INPUTS; j++) sum += sample[j] * weights_ih[j][i];
    h_activations[i] = sigmoid(sum * SIGMOID_SCALE);
  }
  // take hidden layer output and pass it to output
  for (int i = 0; i < OUTPUTS; i++) {
    sum = 0;
    for (int j = 0; j < units; j++) sum += h_activations[j] * weights_ho[j][i];
    activations[i] = sigmoid(sum * SIGMOID_SCALE * (MAX_HIDDEN / units));
  }
}

void backprop_2layer(double sample[INPUTS], double h_activations[MAX_HIDDEN],
                     double activations[OUTPUTS],
                     double (*sigmoid)(double input), int label,
                     double weights_ih[INPUTS][MAX_HIDDEN],
                     double weights_ho[MAX_HIDDEN][OUTPUTS], int units) {
  double error, h_2, new_error, target[OUTPUTS], derivative[OUTPUTS],
      temp_weights_ho[units][OUTPUTS];

  for (int i = 0; i < OUTPUTS; i++)
    target[i] =
        isTanh(sigmoid) ? (i == label ? 0.6 : -0.6) : (i == label ? 0.8 : 0.2);

  for (int i = 0; i < OUTPUTS; i++) {
    derivative[i] =
        ((isTanh(sigmoid) ? 1 : activations[i]) - pow(activations[i], 2)) *
        (target[i] - activations[i]);

    for (int j = 0; j < units; j++) {
      temp_weights_ho[j][i] = weights_ho[j][i];  // store it before change
      weights_ho[j][i] += ALPHA * derivative[i] * h_activations[j];
    }
  }

  for (int i = 0; i < OUTPUTS; i++) {
    error = 0.0;
    // Error is the sum of their outputs
    for (int k = 0; k < OUTPUTS; k++)
      error += derivative[k] * temp_weights_ho[i][k];

    h_2 = pow(h_activations[i], 2);

    error *= isTanh(sigmoid) ? (target[i] - activations[i]) * (1.0 - h_2)
                             : h_activations[i] - h_2;

    for (int j = 0; j < INPUTS; j++)
      weights_ih[j][i] += ALPHA * error * sample[j];
  }
}

double logistic(double input) {
  // logistic value based on input
  return 1.0 / (1.0 + exp(-input));
}

int isTanh(double (*sigmoid)(double input)) { return sigmoid(0) == tanh(0); }

int findMax(double activations[OUTPUTS]) {
  int max_idx = -1;
  double max = -DBL_MAX;
  for (int i = 0; i < OUTPUTS; i++) {
    if (activations[i] > max) {
      max_idx = i;
      max = activations[i];
    }
  }
  return max_idx;
}
