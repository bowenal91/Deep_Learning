#include "Activation.hpp"

Activation::Activation() {
    return;
}

virtual double Activation::evaluate(double x) {
    return 0.0;
}

virtual double Activation::deriv(double x) {
    return 0.0;
}

ReLU::ReLU() {
    return;
}

double ReLU::evaluate(double x) {
    return max(0.0,x);
}

double ReLU::deriv(double x) {
    if (x <= 0.0) {return 0.0;}
    else {return 1.0;}
}

Logistic::Logistic() {
    return;
}

double Logistic::evaluate(double x) {
    return 0.0;
}

double Logistic::deriv(double x) {
    return 0.0;
}
