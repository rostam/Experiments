//
// Created by rostam on 19.02.19.
//

#ifndef NN_NN_H
#define NN_NN_H

#include <vector>
#include <random>

typedef std::vector<double> VD;

class nn {
    int input, hidden, output;
    VD weights_input_hidden, weights_hidden_output;
    double lr;
public:
    nn(int input_nodes, int hidden_nodes, int output_nodes, double learning_rate);

    static double sigmoid(double x);
};




#endif //NN_NN_H
