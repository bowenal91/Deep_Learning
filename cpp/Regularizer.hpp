#ifndef REGULARIZER_H
#define REGULARIZER_H

#include <vector>
#include <string>
#include <random>
#include "Tensor.hpp"

class Regularizer {
    double lambda;
    public:
        Regularizer(double L) {lambda{L}}; 
        virtual double calc_loss(Tensor &input) {};
        virtual Tensor calc_deriv(Tensor &input) {};
};

class Ridge : public Regularizer {
    public:
        Ridge(double L) : Regularizer(L) {};
        double calc_loss(Tensor &input) override;
        Tensor calc_deriv(Tensor &input) override;
};

class Lasso : public Regularizer {
    public:
        Lasso(double L) : Regularizer(L) {};
        double calc_loss(Tensor &input) override;
        Tensor calc_deriv(Tensor &input) override;
};

#endif
