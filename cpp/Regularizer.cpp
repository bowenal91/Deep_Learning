#include "Regularizer.hpp"
#include "Tensor.hpp"


double Ridge::calc_loss(Tensor &input) {
    return lambda*input*input;
}

Tensor Ridge::calc_deriv(Tensor &input) {
    return lambda*input;
}

double Ridge::calc_deriv(double input) {
    return lambda*input;
}

double Lasso::calc_loss(Tensor &input) {
    return lambda*input.sum();
}

Tensor Lasso::calc_deriv(Tensor &input) {
    Tensor output(input.get_shape());
    for (int i=0;i<input.get_size();i++) {
        if (input.get_value(i) > 0.0) {
            output.set_value(i,lambda);
        } else {
            output.set_value(i,-lambda);
        }
    }
    return output;
}

double Lasso::calc_deriv(double input) {
    if (input > 0) {
        return lambda;
    } else {
        return -lambda;
    }
}
