#include "Conv.hpp"
#include <assert.h>

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
    input_shape.clear();
    output_shape.clear();
    weights.clear();
    biases.clear();
    filter_size.clear();
    strides.clear();
    
    numFilters = num_filters;
    input_shape = data_shape;
    reg = regu;
    init = initial;
    strides = stride_size;
    if (!init) {
        init = new Glorot_Uniform();
    }
   
    filter_size.push_back(input_shape[0]);
    for (int i=0;i<kernel_size.size();i++) {
        filter_size.push_back(kernel_size[i]);
    }

    output_shape.push_back(numFilters);
    for (int i=1;i<input_size.size();i++) {
        int W = input_size[i];
        int K = filter_size[i];
        int S = strides[i-1];
        int P = 0;
        if (padding) {
            P = K-S;
        }
        assert((W-K+2*P)%S == 0);

        int O = (W-K+2*P)/S;
        output_shape.push_back(O);
    }
    
    for (int i=0;i<numFilters;i++) {
        weights.push_back(Tensor(filter_size));
        biases.push_back(0.0);
    }
    init_weights();
    return; 
}

void Conv::init_weights() {
    for (int i = 0;i<numNeurons;i++) {
        weights[i] = init->init_weights(filter_size, weights[i].get_size(), numFilters);
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
