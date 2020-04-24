#ifndef INITIALIZER_H
#define INITIALIZER_H

#include <vector>
#include <string>
#include <random>
#include "Tensor.hpp"

class Initializer {
    public:
        std::mt19937 *mt;
        Initializer();
        virtual Tensor init_weights(std::vector<int> &shape, int numInputs, int numOutputs) {};
};

class Glorot_Uniform : public Initializer {
    public:
        Glorot_Uniform() : Initializer() {};
        Tensor init_weights(std::vector<int> &shape, int numInputs, int numOutputs) override;
};

class InitializerFactory {
    public:
        Initializer *create(std::string &name) {
            if (name == "glorot uniform") {
                return new Glorot_Uniform();
            }
        }
};

#endif
