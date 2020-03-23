#include "Dense.hpp"

Dense::Dense(int num_neurons) {
    numNeurons = num_neurons;
    ouput_shape.append(num_neurons);
}

void Dense::init_layer(const vector<int>& data_shape) {
    for (int i=0;i<numNeurons;i++) {
        weight = Tensor(data_shape);
        weights.push_back(weight);
        biases.push_back(0.0);
    }
    return; 
}

Tensor Dense::evaluate(const Tensor& input) {
    Tensor output(output_shape);
    double result;
    for (int i=0; i<num_neurons;i++) {
        vector<int> loc{i}
        result = input*weights[i]+biases[i];
        //Do activation stuff here
        output.set_value(loc, result);
    }

    return output;
}
