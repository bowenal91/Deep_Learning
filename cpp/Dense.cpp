#include "Dense.hpp"

Dense::Dense(int num_neurons, std::string& activation_name) {
    numNeurons = num_neurons;
    set_activation(activation_name);
}

void Dense::init_layer(const vector<int>& data_shape) {
    input_shape.clear();
    output_shape.clear();
    weights.clear();
    batch_size = data_shape[0];
    output_shape.push_back(batch_size);
    output_shape.push_back(numNeurons);
    for (int i=1;i<data_shape.size();i++) {
        input_shape.push_back(data_shape[i]);
    }
    for (int i=0;i<numNeurons;i++) {
        weight = Tensor(input_shape);
        weights.push_back(weight);
        biases.push_back(0.0);
    }
    return; 
}

void Dense::set_activation(std::string& name) {
    if (name == "ReLU") {
        ReLU a;
        act = &a;
    }

    else if (name == "Logistic") {
        Logistic a;
        act = &a;
    }

    else if (name == "None") {
        Activation a;
        act = &a;
    }
    return;
}

Tensor Dense::evaluate(const Tensor& input) {
    Tensor output(output_shape);
    double result;
    for (int i=0; i<num_neurons;i++) {
        vector<int> loc{i}
        result = input*weights[i]+biases[i];
        output.set_value(loc, result);
    }

    output = act->evaluate(output);

    return output;
}

Tensor Dense::back_propagate(const Tensor& input) {
    Tensor output(input_shape);
    double result;
}
