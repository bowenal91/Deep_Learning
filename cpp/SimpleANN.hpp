#ifndef SIMPLEANN_H
#define SIMPLEANN_H

#include "Tensor.hpp"
#include "Layer.hpp"
#include "Activation.hpp"
#include "Dense.hpp"
#include "Loss.hpp"

using namespace std;

class SimpleANN {
    vector<Layer*> layers;
    int numLayers;
    Loss *loss;
    public:
        SimpleANN(vector<int> &input_size, vector<int> &layerSize, string &loss_function);
        Tensor predict(Tensor &input);
        void train(vector<Tensor> &x, vector<Tensor> &y, int batch_size, int num_epochs, double rate);
        double test(vector<Tensor> &x, vector<Tensor> &y);
};


#endif
