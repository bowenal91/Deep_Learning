#ifndef INITIALIZER_H
#define INITIALIZER_H

#include <vector>
#include <string>
#include <random>
#include "Tensor.hpp"

class Initializer {
    std::mt19937 *mt; 
    public: 
        Initializer();
        virtual std::vector<Tensor> init_weights(std::vector<int> &shape, int numInputs, int numOutputs) {};
};

class Glorot_Uniform : public Initializer {
    public:
        Glorot_Uniform() : Initializer() {};
        std::vector<Tensor> init_weights(std::vector<int> &shape, int numInputs, int numOutputs) override;
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
