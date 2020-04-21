#include "Conv.hpp"

using namespace std;
Conv::Conv(int num_neurons) {
    numNeurons = num_neurons;
}

Conv::Conv(int num_neurons, const vector<int> &data_shape) {
    numNeurons = num_neurons;
    init_layer(data_shape);
}

Conv::Conv(int num_neurons, Layer *prev) {
    numNeurons = num_neurons;
    init_layer(prev->get_output_shape());
}

void Conv::init_layer(const vector<int>& data_shape) {
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

vector<int> Conv::get_output_shape() {
    return output_shape;
}

Tensor Conv::evaluate(Tensor& input) {
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

vector<Tensor> Conv::evaluate(vector<Tensor> &input) {
    vector<Tensor> output;
    for (int i=0;i<input.size();i++) {
        output.push_back(evaluate(input[i]));
    }
    return output;
}

Tensor Conv::back_propagate(Tensor &forward, Tensor &backward) {
    Tensor output(input_shape);
    double sum;
    for (int i=0;i<forward.get_size();i++) {
        sum = 0.0;
        for (int j=0;j<backward.get_size();j++) {
            sum += backward.get_value(j)*weights[j].get_value(i);
        }
        output.set_value(i,sum);
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
    int i;
    for (i=0;i<numNeurons;i++) {
        Tensor t(input_shape);
        updates.push_back(t);
    }

    for (i=0;i<forward.size();i++) {
        for (int j=0;j<numNeurons;j++) {
            updates[j] = updates[j] + backward[i].get_value(j)*forward[i];
        }
    }

    for (i=0;i<numNeurons;i++) {
        weights[i] = weights[i] - rate*updates[i];
    }
}

vector<Tensor> Conv::update_propagate(vector<Tensor> &forward, vector<Tensor> &backward, double rate) {
    vector<Tensor> output = back_propagate(forward,backward);
    update_weights(forward,backward,rate);
    return output;
}
