#ifndef CONV_H
#define CONV_H

#include <vector>
#include "Tensor.hpp"
#include "Layer.hpp"
#include "Activation.hpp"

class Conv : public Layer {
    std::vector<int> input_shape;
    std::vector<int> output_shape;
    std::vector<int> weight_shape;
    int filter_size;
    int filter_dimension;
    Tensor weights;
    std::vector<double> biases;
    Activation *act;
    int num_filters;
    int batch_size;
    void set_activation(std::string& name);
    public:
        Conv(int numFilters, int filterSize, std::string activation_name="None");
        void init_layer(const std::vector<int>& data_shape) override;
        void randomize_weights(double max);
        Tensor evaluate(Tensor& input) override;
        Tensor back_propagate(Tensor& input) override;
};

#endif
