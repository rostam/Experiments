#include <utility>

//
// Created by rostam on 19.02.19.
//

#include "nn.h"

nn::nn(int input_nodes, int hidden_nodes, int output_nodes, double learning_rate) {
    this->input = input_nodes;
    this->hidden = hidden_nodes;
    this->output = output_nodes;
    this->lr = learning_rate;
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, pow(hidden, -0.5));
    for(int i=0;i<hidden;i++)
        for(int j=0;j < input;j++)
            weights_input_hidden.push_back(distribution(generator));
}

float nn::sigmoid(float x)
{
    float exp_value;
    float return_value;
    exp_value = exp((double) -x);
    return_value = 1 / (1 + exp_value);
    return return_value;
}
