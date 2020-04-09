#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <vector>
#include <functional>
#include "Tensor.hpp"
#include <math.h>

class Activation {
    vector<int> input_size;
    public:
        Activation(const vector<int>& shape);
        Tensor evaluate(Tensor& x);
        virtual double point_wise_function(double x);
        virtual void extra_function(Tensor& x);
        Tensor back_propagate(Tensor& x);
        virtual double deriv(double x);
};

class ReLU : public Activation {
    public: 
        ReLU(const vector<int>& shape) : Activation(shape) {}
        double point_wise_function(double x) override;
        double deriv(double x) override;
};

class Logistic : public Activation {
    public:
        Logistic(const vector<int>& shape) : Activation(shape) {}
        double point_wise_function(double x) override;
        double deriv(double x) override;
};

class SoftMax : public Activation {
    public:
        SoftMax(const vector<int>& shape) : Activation(shape) {}
        double point_wise_function(double x) override;
        double deriv(double x) override;
        void extra_function(Tensor& x) override;
};

#endif
