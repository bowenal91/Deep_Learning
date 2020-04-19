#include "Loss.hpp"
#include <math.h>
#include <assert.h>

using namespace std;

double Loss::calculate_loss(Tensor &y_pred, Tensor &y_label) {
    assert(y_pred.size == y_label.size);
    assert(y_pred.rank == y_label.rank);
    for (int i=0;i<y_pred.rank;i++) {
        assert(y_pred.shape[i] == y_label.shape[i]);
    }

    int batch_size = y_pred.shape[0];
    double prefactor = 1.0/double(batch_size);
    double error = 0.0;
    error_deriv.resize(y_pred.shape);

    for (int i=0;i<y_pred.size;i++) {
        error += point_wise_func(y_pred.get_value(i), y_label.get_value(i));
        error_deriv.set_value(i, prefactor*deriv(i, y_pred.get_value(i), y_label.get_value(i)));
    }

    return prefactor*error;
}

Tensor Loss::back_propagate() {
    return error_deriv.collapse();
}

double MSE::point_wise_func(double x_pred, double x_label) {
    return (x_pred-x_label)*(x_pred-x_label);
}

double MSE::deriv(double x_pred, double x_label) {
    return 2*(x_pred-x_label);
}   

double MAE::point_wise_func(double x_pred, double x_label) {
    return abs(x_pred-x_label);
}

double MAE::deriv(double x_pred, double x_label) {
    if (x_pred > x_label) {
        return 1.0;
    } else {
        return -1.0;
    }
}

double Binary_Cross_Entropy::point_wise_func(double x_pred, double x_label) {
    if (x_label > 0.5) {
        return log(x_pred);
    } else {
        return log(1.0-x_pred);
    }
}

double Binary_Cross_Entropy::deriv(double x_pred, double x_label) {
    if (x_label > 0.5) {
        return 1.0/x_pred;
    } else {
        return -1.0/(1.0-x_pred);
    }
}

double Categorical_Cross_Entropy::point_wise_func(double x_pred, double x_label) {
    if (x_label > 0.5) {
        return log(x_pred);
    } else {
        return 0.0;
    }
}

double Categorical_Cross_Entropy::deriv(double x_pred, double x_label) {
    if (x_label > 0.5) {
        return 1.0/x_pred;
    } else {
        return 0.0;
    }
}
