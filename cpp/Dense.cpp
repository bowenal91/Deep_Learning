#include "Dense.hpp"

Dense::Dense(int num_neurons, std::string activation_name) {
    numNeurons = num_neurons;
    set_activation(activation_name);
}

void Dense::init_layer(const vector<int>& data_shape) {
    input_shape.clear();
    output_shape.clear();
    weight_shape.clear();
    biases.clear();
    batch_size = data_shape[0];
    output_shape.push_back(batch_size);
    output_shape.push_back(numNeurons);
    weight_shape.push_back(numNeurons);
    for (int i=1;i<data_shape.size();i++) {
        input_shape.push_back(data_shape[i]);
        weight_shape.push_back(data_shape[i]);
    }
    weights = Tensor(weight_shape);
    for (int i=0;i<numNeurons;i++) {
        biases.push_back(0.0);
    }
    return; 
}

void Dense::set_activation(std::string& name) {
    if (name == "ReLU") {
        ReLU a(output_shape);
        act = &a;
    }

    else if (name == "Logistic") {
        Logistic a(output_shape);
        act = &a;
    }

    else if (name == "None") {
        Activation a(output_shape);
        act = &a;
    }
    return;
}

Tensor Dense::evaluate(Tensor& input) {
    Tensor output(output_shape);
    double result;
    vector<int> weight_ids, input_ids, loc;
    for (int i=0;i<input_shape.size();i++) {
        weight_ids.push_back(0);
        weight_ids.push_back(input_shape[i]-1);
        input_ids.push_back(0);
        input_ids.push_back(input_shape[i]-1);
    }
    loc.push_back(0);
    loc.push_back(0);
    for (int batch=0;batch<batch_size;batch++) {
        for (int i=0; i<numNeurons;i++) {
            weight_ids[0] = i;
            weight_ids[1] = i;
            input_ids[0] = batch;
            input_ids[1] = batch;
            result = weights.subset_mult(weight_ids, input_ids, input)+biases[i];
            loc[0] = batch;
            loc[1] = i;
            output.set_value(loc, result);
        }
    }
    output = act->evaluate(output);

    return output;
}

Tensor Dense::back_propagate(Tensor& input) {
    Tensor output(input_shape);
    double result;
}
