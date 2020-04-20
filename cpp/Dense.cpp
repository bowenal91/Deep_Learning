#include "Dense.hpp"

using namespace std;
Dense::Dense(int num_neurons) {
    numNeurons = num_neurons;
}

void Dense::init_layer(const vector<int>& data_shape) {
    input_shape.clear();
    output_shape.clear();
    weights.clear();
    biases.clear();
    output_shape.push_back(numNeurons);
    for (int i=0;i<data_shape.size();i++) {
        input_shape.push_back(data_shape[i]);
    }
    for (int i=0;i<numNeurons;i++) {
        weights.push_back(Tensor(input_shape));
        biases.push_back(0.0);
    }
    return; 
}

Tensor Dense::evaluate(Tensor& input) {
    Tensor output(output_shape);
    vector<int> id{0};
    double result;
    for (int i=0; i<numNeurons;i++) {
        id[0] = i;
        result = weights[i]*input + biases[i];
        output.set_value(id, result); 
    }

    return output;
}

vector<Tensor> Dense::evaluate(vector<Tensor> &input) {
    vector<Tensor> output;
    for (int i=0;i<input.size();i++) {
        output.push_back(evaluate(input[i]);
    }
    return output;
}

Tensor Dense::back_propagate(Tensor& input) {
    Tensor output(input_shape);
    double result;
}
