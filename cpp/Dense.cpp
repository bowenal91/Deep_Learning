#include "Dense.hpp"

Dense::Dense(int num_neurons, std::string& activation_name) {
    numNeurons = num_neurons;
    ouput_shape.append(num_neurons);
    act.set_activation(activation_name);
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
        result = act.evaluate(result);
        output.set_value(loc, result);
    }

    return output;
}
