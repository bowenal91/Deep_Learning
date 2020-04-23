#ifndef REGULARIZER_H
#define REGULARIZER_H

#include <vector>
#include <string>
#include <random>
#include "Tensor.hpp"

class Regularizer {
    public:
        std::mt19937 mt;
        Initializer();
        virtual Tensor init_weights(std::vector<int> &shape, int numInputs, int numOutputs) {};
};

class Ridge : public Regularizer {

};

class Lasso : public Regularizer {

};

#endif
