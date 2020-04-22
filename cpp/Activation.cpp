#include "Activation.hpp"
#include <math.h>

using namespace std;

Activation::Activation() {
    return;
}

Activation::Activation(string &name, int axis, vector<int> &shape) {
    init_layer(name, axis, shape);
    return;
}

Activation::Activation(string &name, int axis, Layer *prev) {
    init_layer(name,axis, prev->get_output_shape());
}

vector<int> Activation::get_output_shape() {
    return activation->get_output_shape();
}

void Activation::init_layer(string &name, int axis, vector<int> &shape) {
    ActivationFactory f;
    activation = f.create(name, shape, axis);
    return;
}

Tensor Activation::evaluate(Tensor& x) {
    return activation->evaluate(x);
}

vector<Tensor> Activation::evaluate(vector<Tensor> &x) {
    return activation->evaluate(x);
}

Tensor Activation::back_propagate(Tensor &forward, Tensor &backward) {
    return activation->back_propagate(forward,backward);
   
}

vector<Tensor> Activation::back_propagate(vector<Tensor> &forward, vector<Tensor> &backward) {
    return activation->back_propagate(forward,backward);
}

vector<Tensor> Activation::update_propagate(vector<Tensor> &forward, vector<Tensor> &backward, double rate) {
    return activation->update_propagate(forward,backward,rate); 
}

