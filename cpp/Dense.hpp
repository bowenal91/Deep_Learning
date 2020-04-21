#ifndef DENSE_H
#define DENSE_H

#include <vector>
#include "Tensor.hpp"
#include "Layer.hpp"
#include <string>

class Dense : public Layer {
    std::vector<int> input_shape;
    std::vector<int> output_shape;
    std::vector<int> weight_shape;
    vector<Tensor> weights;
    std::vector<double> biases;
    int numNeurons;
    int batch_size;
    public:
        Dense(int num_neurons);
        Dense(int num_neurons, const std::vector<int> &data_shape);
        Dense(int num_neurons, Layer *prev);
        std::vector<int> get_output_shape() override;
        void init_layer(const std::vector<int>& data_shape) override;
        void randomize_weights(double max);
        Tensor evaluate(Tensor& input) override;
        vector<Tensor> evaluate(vector<Tensor>& input) override;
        Tensor back_propagate(Tensor& input) override;
};

#endif
