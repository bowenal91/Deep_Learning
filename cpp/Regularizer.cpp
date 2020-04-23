#include "Regularizer.hpp"
#include "Tensor.hpp"


double Ridge::calc_loss(Tensor &input) {
    return lambda*input*input;
}

double Ridge::calc_deriv(Tensor &input) {
    return lambda*input;
}

double Lasso::calc_loss(Tensor &input) {
    return lambda*input.sum();
}

double Lasso::calc_deriv(Tensor &input) {
    return lambda*(1+(0.0*input));
}
