#include "Activation.hpp"
#include <math.h>

using namespace std;

Activation::Activation() {
    return;
}

Activation::Activation(const vector<int> &shape) {
    input_size = shape;
    return;
}

void Activation::init_layer(const vector<int> &shape) {
    input_size = shape;
    return;
}

Tensor Activation::evaluate(Tensor& x) {
    Tensor output(input_size);
    double value;
    for (int i=0;i<x.size;i++) {
        value = x.get_value(i);
        output.set_value(i, point_wise_function(value));
    }
    extra_function(output);
    return output;
}

Tensor Activation::back_propagate(Tensor& x) {

}

double Activation::point_wise_function(double x) {
    return x;
}

void Activation::extra_function(Tensor& x) {
    return;
}

double Activation::deriv(double x) {
    return 1.0;
}

double ReLU::point_wise_function(double x) {
    return max(0.0,x);
}

double ReLU::deriv(double x) {
    if (x <= 0.0) {return 0.0;}
    else {return 1.0;}
}

double Logistic::point_wise_function(double x) {
    double out = exp(x);
    return 1.0/(1.0+out);

}

double Logistic::deriv(double x) {
    double out = point_wise_function(x);
    return out*(1.0-out);
}

double SoftMax::point_wise_function(double x) {
    return exp(x);
}

void SoftMax::extra_function(Tensor& x) {
    return;
}

double SoftMax::deriv(double x) {
    return 0.0;
}
