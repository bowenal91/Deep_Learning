#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <vector>
#include "Tensor.hpp"
#include "Layer.hpp"

class Activation : public Layer {
    std::vector<int> input_size;
    public:
        Activation();
        Activation(const std::vector<int>& shape);
        Activation(Layer *prev);
        std::vector<int> get_output_shape() override;
        void init_layer(const std::vector<int> &shape) override;
        Tensor evaluate(Tensor& x) override;
        vector<Tensor> evaluate(vector<Tensor>& x) override;
        virtual double point_wise_function(double x);
        virtual void extra_function(Tensor& x);
        virtual void extra_function_deriv(Tensor& x) {};
        Tensor back_propagate(Tensor& forward, Tensor &backward) override;
        vector<Tensor> back_propagate(vector<Tensor>& forward, vector<Tensor> &backward) override;
        vector<Tensor> update_propagate(vector<Tensor>& forward, vector<Tensor> &backward, double rate=0.0) override;
        virtual double deriv(double x);
};

class ReLU : public Activation {
    public: 
        ReLU(const std::vector<int>& shape) : Activation(shape) {}
        double point_wise_function(double x) override;
        double deriv(double x) override;
};

class Logistic : public Activation {
    public:
        Logistic(const std::vector<int>& shape) : Activation(shape) {}
        double point_wise_function(double x) override;
        double deriv(double x) override;
};

class SoftMax : public Activation {
    Tensor normalizations;
    int axis;
    public:
        SoftMax(const std::vector<int>& shape, int axis) : Activation(shape) {axis = axis;}
        double point_wise_function(double x) override;
        double deriv(double x) override;
        void extra_function(Tensor& x) override;
        void extra_function_deriv(Tensor &x) override;
};

#endif
