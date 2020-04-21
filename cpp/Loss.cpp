#include "Loss.hpp"
#include <math.h>
#include <assert.h>

using namespace std;

double Loss::calculate_loss(Tensor &y_pred, Tensor &y_label) {
    double loss = 0.0;
    assert(y_pred.get_size() == y_label.get_size());
    for (int i=0; i<y_pred.get_size(); i++) {
        loss += point_wise_func(y_pred.get_value(i), y_label.get_value(i));
    }

    return loss;
}

Tensor Loss::calc_deriv(Tensor &y_pred, Tensor &y_label, double prefactor) {
    int i;
    assert(y_pred.get_rank() == y_label.get_rank());
    for (i=0;i<y_pred.get_rank();i++) {
        assert(y_pred.get_shape(i) == y_label.get_shape(i));
    }
    Tensor output(y_pred.get_shape());
    
    for (i=0;i<y_pred.get_size();i++) {
        output.set_value(i, prefactor*deriv(y_pred.get_value(i), y_label.get_value(i)));
    }
    return output;
}

double Loss::calculate_loss(vector<Tensor> &y_pred, vector<Tensor> &y_label) {
    assert(y_pred.size() == y_label.size());
    assert(y_pred.size() > 0);
    

    int batch_size = y_pred.size();
    double prefactor = 1.0/double(batch_size);
    double error = 0.0;
    error_out.clear();
    for (int i=0; i<batch_size; i++) {
        error += calculate_loss(y_pred[i],y_label[i]);
        error_out.push_back(calc_deriv(y_pred[i],y_label[i],prefactor));
    }

    return prefactor*error;
}

vector<Tensor> Loss::back_propagate() {
    return error_out;
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
