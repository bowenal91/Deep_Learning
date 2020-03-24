#include "Activation.hpp"

Activation::Activation() {
    return;
}

void Activation::set_activation(std::string& name) {
    if (name == "ReLU") {
        evaluate = ReLU;
        derivative = d_ReLU;
    }

    if (name == "Logistic") {
        evaluate = Logistic;
        derivative = d_ReLU;
    }

    return;
}

double Activation::ReLU(double input) {
    return min(0.0,input);
}

double Activation::d_ReLU(double input) {
    if (input <= 0.0) {return 0.0;}
    else {return 1.0;}
}

double Activation::Logistic(double input) {

}

double Activation::d_Logistic(double input) {

}
