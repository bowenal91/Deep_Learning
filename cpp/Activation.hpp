#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <vector>
#include "Tensor.hpp"
#include "Layer.hpp"

class Activation : public Layer {
    std::vector<int> input_size;
    public:
        Activation();
        Activation(const std::vector<int>& shape);
        void init_layer(const std::vector<int> &shape) override;
        Tensor evaluate(Tensor& x) override;
        virtual double point_wise_function(double x);
        virtual void extra_function(Tensor& x);
        Tensor back_propagate(Tensor& x) override;
        virtual double deriv(double x);
};

class ReLU : public Activation {
    public: 
        ReLU(const std::vector<int>& shape) : Activation(shape) {}
        double point_wise_function(double x) override;
        double deriv(double x) override;
};

class Logistic : public Activation {
    public:
        Logistic(const std::vector<int>& shape) : Activation(shape) {}
        double point_wise_function(double x) override;
        double deriv(double x) override;
};

class SoftMax : public Activation {
    public:
        SoftMax(const std::vector<int>& shape) : Activation(shape) {}
        double point_wise_function(double x) override;
        double deriv(double x) override;
        void extra_function(Tensor& x) override;
};

#endif
