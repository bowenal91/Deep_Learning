#include "SimpleANN.hpp"
#include <math.h>
#include <iostream>

using namespace std;

SimpleANN::SimpleANN(vector<int> &input_size, vector<int> &layerSize, string &loss_function) {
    numLayers = layerSize.size();
    LossFactory f;
    loss = f.create(loss_function);
    Activation *myAct;
    Dense *myDense;
    myDense = new Dense(layerSize[0],input_size);
    layers.push_back(myDense);
    for (int i=1;i<numLayers;i++) {
        myAct = new Activation("ReLU",myDense);
        layers.push_back(myAct);
        myDense = new Dense(layerSize[i],myAct);
        layers.push_back(myDense);
    }
    numLayers = layers.size();
}

Tensor SimpleANN::predict(Tensor &input) {
    Tensor output = input;
    for (int i=0;i<layers.size();i++) {
        output = layers[i]->evaluate(output);
    }
    return output;
}

void SimpleANN::train(vector<Tensor> &x, vector<Tensor> &y, int batch_size, int num_epochs, double rate) {
    vector<vector<Tensor> > tensors;
    vector<vector<Tensor> > derivs;
    vector<Tensor> labels;
    int num_samples = x.size();
    int num_batch = ceil(double(x.size())/double(batch_size));
    
    int i;
    for (i=0;i<numLayers+1;i++) {
        vector<Tensor> dummy,dummy2;
        tensors.push_back(dummy);
        derivs.push_back(dummy2);
    }
    int epoch,batch; 
    for (epoch=0;epoch<num_epochs;epoch++) {
        for (batch=0;batch<num_batch;batch++) {
            tensors[0].clear();
            labels.clear();
            //Assign batch
            int start = batch*batch_size;
            int end = min((batch+1)*batch_size, num_samples);
            for (i=start;i<end;i++) {
                tensors[0].push_back(x[i]);
                labels.push_back(y[i]);
            }

            for (i=0;i<numLayers;i++) {
                tensors[i+1] = layers[i]->evaluate(tensors[i]);
            }

            double batch_loss = loss->calculate_loss(tensors[numLayers-1], labels);
            derivs[numLayers-1] = loss->back_propagate();

            for (i=numLayers-1;i>=0;i--) {
                derivs[i] = layers[i]->update_propagate(tensors[i], derivs[i], rate); 
            }
            
            cout << "Epoch: " << epoch << ", Batch: " << batch << ", Loss: " << batch_loss << endl;

        }


    }
}



