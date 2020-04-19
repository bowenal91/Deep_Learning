#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include "Tensor.hpp"
#include <string>

class Layer {
    public:
        Layer() {};
        virtual void init_layer(const std::vector<int> &shape) {};
        virtual Tensor evaluate(Tensor& input) {};
        virtual Tensor back_propagate(Tensor& input) {};
};

#endif
