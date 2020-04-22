#include "ActivationFuncFunc.hpp"
#include <math.h>

using namespace std;

ActivationFunc::ActivationFunc() {
    return;
}

ActivationFunc::ActivationFunc(const vector<int> &shape) {
    init_layer(shape);
    return;
}

ActivationFunc::ActivationFunc(Layer *prev) {
    init_layer(prev->get_output_shape());
}

vector<int> ActivationFunc::get_output_shape() {
    return input_size;
}

void ActivationFunc::init_layer(const vector<int> &shape) {
    input_size = shape;
    return;
}

Tensor ActivationFunc::evaluate(Tensor& x) {
    Tensor output(x.get_shape());
    double value;
    for (int i=0;i<x.get_size();i++) {
        value = x.get_value(i);
        output.set_value(i, point_wise_function(value));
    }
    extra_function(output);
    return output;
}

vector<Tensor> ActivationFunc::evaluate(vector<Tensor> &x) {
    vector<Tensor> output;
    for (int i=0;i<x.size();i++) {
        output.push_back(evaluate(x[i]));
    }
    return output;
}

Tensor ActivationFunc::back_propagate(Tensor &forward, Tensor &backward) {
    vector<Tensor> output, forward2;
    forward2 = forward;
    double value;
    
    for (int i=0;i<forward.get_size();i++) {
        value = backward.get_value(i);
        value *= deriv(forward2.get_value(i));
        output.set_value(i,value);
    }
    return output;
   
}

vector<Tensor> ActivationFunc::back_propagate(vector<Tensor> &forward, vector<Tensor> &backward) {
    vector<Tensor> output;
    for (int i=0;i<forward.size();i++) {
        output.push_back(back_propagate(forward[i],backward[i]));
    }
    return output;
}

vector<Tensor> ActivationFunc::update_propagate(vector<Tensor> &forward, vector<Tensor> &backward, double rate=0.0) {
    return back_propagate(forward,backward); 
}

double ActivationFunc::point_wise_function(double x) {
    return x;
}

void ActivationFunc::extra_function(Tensor& x) {
    
    return;
}

double ActivationFunc::deriv(double x) {
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
    x = x.normalize(axis);
    return;
}

double SoftMax::deriv(double x) {
    //Note this is not the derivative - it's the derivative in terms of y 
    return x*(1-x);
}

void SoftMax::extra_function_deriv(Tensor& x) {
    x = evaluate(x);
}
