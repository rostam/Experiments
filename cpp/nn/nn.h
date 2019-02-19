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

    static float sigmoid(float x);
};




#endif //NN_NN_H

//
//self.weights_input_hidden = np.random.normal(0.0, pow(self.hidden, -0.5), (self.hidden, self.input))
//self.weights_hidden_output = np.random.normal(0.0, pow(self.output, -0.5), (self.output, self.hidden))
//
//self.activate_function = lambda x: scs.expit(x)
//pass
//
//        def train(self, inputs_list, targets_list):
//inputs = np.array(inputs_list, ndmin=2).T
//targets = np.array(targets_list, ndmin=2).T
//
//hidden_inputs = np.dot(self.weights_input_hidden, inputs)
//hidden_outputs = self.activate_function(hidden_inputs)
//
//final_inputs = np.dot(self.weights_hidden_output, hidden_outputs)
//final_outputs = self.activate_function(final_inputs)
//
//output_errors = targets - final_outputs
//hidden_errors = np.dot(self.weights_hidden_output.T, output_errors)
//
//self.weights_hidden_output += self.lr * np.dot((output_errors*final_outputs*(1.0 - final_outputs)), np.transpose(hidden_outputs))
//self.weights_input_hidden += self.lr * np.dot((hidden_errors*hidden_outputs*(1.0 - hidden_outputs)), np.transpose(inputs))
//pass
//
//        def query(self, inputs_list):
//inputs = np.array(inputs_list, ndmin=2).T
//hidden_inputs = np.dot(self.weights_input_hidden, inputs)
//hidden_outputs = self.activate_function(hidden_inputs)
//final_inputs = np.dot(self.weights_hidden_output, hidden_outputs)
//final_outputs = self.activate_function(final_inputs)
//return final_outputs
