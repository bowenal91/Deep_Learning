#ifndef DENSE_H
#define DENSE_H

#include <vector>
#include "Tensor.hpp"
#include "Activation.hpp"
#include <string>

class Dense {
    std::vector<int> input_shape;
    std::vector<int> output_shape;
    std::vector<int> weight_shape;
    Tensor weights;
    std::vector<double> biases;
    Activation *act;
    int numNeurons;
    int batch_size;
    void set_activation(std::string& name);
    public:
        Dense(int num_neurons, std::string activation_name="None");
        void init_layer(const std::vector<int>& data_shape);
        void randomize_weights(double max);
        Tensor evaluate(Tensor& input);
        Tensor back_propagate(Tensor& input);
};

#endif
