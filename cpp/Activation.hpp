#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <vector>
#include <string>
#include "Tensor.hpp"
#include "Layer.hpp"
#include "ActivationFunc.hpp"

class Activation : public Layer {
    ActivationFunc *activation;
    std::vector<int> input_size;
    public:
        Activation();
        Activation(std::string& name, std::vector<int>& shape, int axis=0);
        Activation(std::string& name, Layer *prev, int axis=0);
        std::vector<int> get_output_shape() override;
        void init_layer(std::string &name, int axis, std::vector<int> shape);
        Tensor evaluate(Tensor& x) override;
        std::vector<Tensor> evaluate(std::vector<Tensor>& x) override;
        Tensor back_propagate(Tensor& forward, Tensor &backward) override;
        std::vector<Tensor> back_propagate(std::vector<Tensor>& forward, std::vector<Tensor> &backward) override;
        std::vector<Tensor> update_propagate(std::vector<Tensor>& forward, std::vector<Tensor> &backward, double rate=0.0) override;
};

#endif
