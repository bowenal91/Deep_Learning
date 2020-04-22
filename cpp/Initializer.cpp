#include "Initializer.hpp"
#include <chrono>
#include <vecitor>
#include <math.h>

using namespace std

Initializer::Initializer() {
    mt = new mt19937(chrono::high_resolution_clock::now().time_since_epoch().count());
}

Glorot_Uniform::init_weights(vector<int> shape, int numInputs, int numOutputs) {
    Tensor output(shape);
    double limit = sqrt(6.0/(double(numInputs+numOutputs)));
    uniform_real_distribution<double> dist(-limit,limit);
    int size = output.get_size();
    for (int i=0;i<size;i++) {
        output.set_value(i,dist(mt()));
    }

    return output;
}
