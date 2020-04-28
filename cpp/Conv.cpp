#include "Conv.hpp"

using namespace std;
Conv::Conv(int num_filters, std::vector<int> kernel_size, std::vector<int> stride_size, bool pad) {

}

Conv::Conv(int num_filters, std::vector<int> kernel_size, std::vector<int> stride_size, bool pad, const vector<int> &data_shape) {
    init_layer(num_neurons,data_shape, NULL, NULL);
}


Conv::Conv(int num_filters, std::vector<int> kernel_size, std::vector<int> stride_size, bool pad, const vector<int> &data_shape, Initializer *initial) {
    init_layer(num_neurons,data_shape, initial, NULL);
}

Conv::Conv(int num_filters, std::vector<int> kernel_size, std::vector<int> stride_size, bool pad, Layer *prev, Initializer *initial) {
    init_layer(num_neurons, prev->get_output_shape(), initial, NULL);
}


Conv::Conv(int num_filters, std::vector<int> kernel_size, std::vector<int> stride_size, bool pad, const vector<int> &data_shape, Initializer *initial, Regularizer *regu) {
    init_layer(num_neurons, data_shape, initial, regu);
}

Conv::Conv(int num_filters, std::vector<int> kernel_size, std::vector<int> stride_size, bool pad, Layer *prev, Initializer *initial, Regularizer *regu) {
    init_layer(num_neurons,prev->get_output_shape(), initial, regu);
}

Conv::Conv(int num_filters, std::vector<int> kernel_size, std::vector<int> stride_size, bool pad, Layer *prev) {
    init_layer(num_neurons, prev->get_output_shape(), NULL, NULL);
}

void Conv::init_layer(int num_filters, std::vector<int> kernel_size, std::vector<int> stride_size,
        bool pad, const vector<int>& data_shape, Initializer *initial, Regularizer *regu) {
    numFilters = num_filters;
    input_shape.clear();
    output_shape.clear();
    weights.clear();
    biases.clear();
    output_shape.push_back(numNeurons);
    reg = regu;
    init = initial;
    if (!init) {
        init = new Glorot_Uniform();
    }
    for (int i=0;i<data_shape.size();i++) {
        input_shape.push_back(data_shape[i]);
    }
    for (int i=0;i<numNeurons;i++) {
        weights.push_back(Tensor(input_shape));
        biases.push_back(0.0);
    }
    init_weights();
    return; 
}

void Conv::init_weights() {
    for (int i = 0;i<numNeurons;i++) {
        weights[i] = init->init_weights(input_shape, weights[i].get_size(), numNeurons);
    }
}

vector<int> Conv::get_output_shape() {
    return output_shape;
}

Tensor Conv::evaluate(Tensor& input) {
    Tensor output(output_shape);
    double result;
    for (int i=0; i<numNeurons;i++) {
        result = weights[i]*input + biases[i];
        output.set_value(i, result); 
    }

    return output;
}

vector<Tensor> Conv::evaluate(vector<Tensor> &input) {
    vector<Tensor> output;
    for (int i=0;i<input.size();i++) {
        output.push_back(evaluate(input[i]));
    }
    return output;
}

Tensor Conv::back_propagate(Tensor &forward, Tensor &backward) {
    Tensor output(input_shape);
    for (int i=0;i<numNeurons;i++) {
        output = output + backward.get_value(i)*weights[i];
    }
    return output;
}

vector<Tensor> Conv::back_propagate(vector<Tensor> &forward, vector<Tensor> &backward) {
    vector<Tensor> output;
    for (int i=0;i<forward.size();i++) {
        output.push_back(back_propagate(forward[i],backward[i]));
    }
    return output;
}

void Conv::update_weights(vector<Tensor> &forward, vector<Tensor> &backward, double rate) {
    vector<Tensor> updates;
    vector<double> update_bias;
    int i;
    for (i=0;i<numNeurons;i++) {
        Tensor t(input_shape);
        updates.push_back(t);
        update_bias.push_back(0.0);
    }

    for (i=0;i<forward.size();i++) {
        for (int j=0;j<numNeurons;j++) {
            updates[j] = updates[j] + backward[i].get_value(j)*forward[i];
            update_bias[j] = update_bias[j] + backward[i].get_value(j);
        }
    }

    if (reg) {
    }

    for (i=0;i<numNeurons;i++) {
        if (reg) {
            weights[i] = weights[i] - reg->calc_deriv(weights[i]);
            biases[i] = biases[i] - reg->calc_deriv(biases[i]);
        }
        weights[i] = weights[i] - rate*updates[i];
        biases[i] = biases[i] - rate*update_bias[i];
    }

}

vector<Tensor> Conv::update_propagate(vector<Tensor> &forward, vector<Tensor> &backward, double rate) {
    vector<Tensor> output = back_propagate(forward,backward);
    update_weights(forward,backward,rate);
    return output;
}
