#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include "Tensor.hpp"
#include <string>

class Layer {
    public:
        Layer() {};
        virtual std::vector<int> get_output_shape() {};
        virtual void init_layer(const std::vector<int> &shape) {};
        virtual Tensor evaluate(Tensor& input) {};
        virtual std::vector<Tensor> evaluate(std::vector<Tensor> &input) {};
        virtual Tensor back_propagate(Tensor &forward, Tensor &backward) {};
        virtual std::vector<Tensor> back_propagate(std::vector<Tensor> &forward, std::vector<Tensor> &backward) {};
        virtual void update_weights(std::vector<Tensor> &forward, std::vector<Tensor> &backward, int rate) {};
        virtual std::vector<Tensor> update_propagate(std::vector<Tensor> &forward, std::vector<Tensor> &backward, int rate) {}; 
};

#endif
