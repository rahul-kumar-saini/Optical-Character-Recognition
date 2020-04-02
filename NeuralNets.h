/*
        Starter by: F.J.E., Feb. 2016
*/

#ifndef __NeuralNets_header

#define __NeuralNets_header

#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INPUTS 785         // Number of inputs 28x28 pixel digits = 784 + 1 bias term
#define OUTPUTS 10         // Output classes
#define MAX_HIDDEN 785     // Maximum number of hidden units
#define ALPHA .01          // Network learning rate
#define SIGMOID_SCALE .01  // Scaling factor for sigmoid function input

int train_1layer_net(double sample[INPUTS], int label,
                     double (*sigmoid)(double input),
                     double weights_io[INPUTS][OUTPUTS]);
int train_2layer_net(double sample[INPUTS], int label,
                     double (*sigmoid)(double input), int units,
                     double weights_ih[INPUTS][MAX_HIDDEN],
                     double weights_ho[MAX_HIDDEN][OUTPUTS]);
void feedforward_1layer(double sample[INPUTS], double (*sigmoid)(double input),
                        double weights_io[INPUTS][OUTPUTS],
                        double activations[OUTPUTS]);
void feedforward_2layer(double sample[INPUTS], double (*sigmoid)(double input),
                        double weights_ih[INPUTS][MAX_HIDDEN],
                        double weights_ho[MAX_HIDDEN][OUTPUTS],
                        double h_activations[MAX_HIDDEN],
                        double activations[OUTPUTS], int units);
void backprop_1layer(double sample[INPUTS], double activations[OUTPUTS],
                     double (*sigmoid)(double input), int label,
                     double weights_io[INPUTS][OUTPUTS]);
void backprop_2layer(double sample[INPUTS], double h_activations[MAX_HIDDEN],
                     double activations[OUTPUTS],
                     double (*sigmoid)(double input), int label,
                     double weights_ih[INPUTS][MAX_HIDDEN],
                     double weights_ho[MAX_HIDDEN][OUTPUTS], int units);
int classify_1layer(double sample[INPUTS], int label,
                    double (*sigmoid)(double input),
                    double weights_io[INPUTS][OUTPUTS]);
int classify_2layer(double sample[INPUTS], int label,
                    double (*sigmoid)(double input), int units,
                    double weights_ih[INPUTS][MAX_HIDDEN],
                    double weights_ho[MAX_HIDDEN][OUTPUTS]);
double logistic(double input);

// If you need to add any function prototypes yourself, you can do so *below*
// this line.

int isTanh(double (*sigmoid)(double input));
int findMax(double activations[OUTPUTS]);
#endif
