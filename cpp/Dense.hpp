#ifndef DENSE_H
#define DENSE_H

#include <vector>
#include "Tensor.hpp"
#include "Layer.hpp"
#include "Initializer.hpp"
#include <string>

class Dense : public Layer {
    std::vector<int> input_shape;
    std::vector<int> output_shape;
    std::vector<Tensor> weights;
    std::vector<double> biases;
    Initializer *init;
    int numNeurons;
    public:
        Dense(int num_neurons);
        Dense(int num_neurons, const std::vector<int> &data_shape, std::string &weight_init="glorot uniform");
        Dense(int num_neurons, Layer *prev, std::string *&weight_init="glorot uniform");
        std::vector<int> get_output_shape() override;
        void init_layer(const std::vector<int>& data_shape);
        void init_weights();
        Tensor evaluate(Tensor& input) override;
        std::vector<Tensor> evaluate(std::vector<Tensor>& input) override;
        Tensor back_propagate(Tensor &forward, Tensor &backward) override;
        std::vector<Tensor> back_propagate(std::vector<Tensor> &forward, std::vector<Tensor> &backward) override;
        void update_weights(std::vector<Tensor> &forward, std::vector<Tensor> &backward, double rate) override;
        std::vector<Tensor> update_propagate(std::vector<Tensor> &forward, std::vector<Tensor> &backward, double rate) override;
        
};

#endif
