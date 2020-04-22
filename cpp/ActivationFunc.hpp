#ifndef ACTIVATIONFUNC_H
#define ACTIVATIONFUNC_H

#include <vector>
#include <string>
#include "Tensor.hpp"
#include "Layer.hpp"

class ActivationFunc : public Layer {
    std::vector<int> input_size;
    public:
        ActivationFunc();
        ActivationFunc(const std::vector<int>& shape);
        ActivationFunc(Layer *prev);
        std::vector<int> get_output_shape() override;
        void init_layer(const std::vector<int> &shape);
        Tensor evaluate(Tensor& x) override;
        std::vector<Tensor> evaluate(std::vector<Tensor>& x) override;
        virtual double point_wise_function(double x);
        virtual void extra_function(Tensor& x);
        virtual void extra_function_deriv(Tensor& x) {};
        Tensor back_propagate(Tensor& forward, Tensor &backward) override;
        std::vector<Tensor> back_propagate(std::vector<Tensor>& forward, std::vector<Tensor> &backward) override;
        std::vector<Tensor> update_propagate(std::vector<Tensor>& forward, std::vector<Tensor> &backward, double rate=0.0) override;
        virtual double deriv(double x);
};

class ReLU : public ActivationFunc {
    public: 
        ReLU(const std::vector<int>& shape) : ActivationFunc(shape) {}
        double point_wise_function(double x) override;
        double deriv(double x) override;
};

class Logistic : public ActivationFunc {
    public:
        Logistic(const std::vector<int>& shape) : ActivationFunc(shape) {}
        double point_wise_function(double x) override;
        double deriv(double x) override;
};

class SoftMax : public ActivationFunc {
    Tensor normalizations;
    int axis;
    public:
        SoftMax(const std::vector<int>& shape, int axis) : ActivationFunc(shape) {axis = axis;}
        double point_wise_function(double x) override;
        double deriv(double x) override;
        void extra_function(Tensor& x) override;
        void extra_function_deriv(Tensor &x) override;
};

class ActivationFactory {
    public:
        ActivationFunc *create(std::string &name, std::vector<int> &shape, int axis=0) {
            if (name == "ReLU") {
                return new ReLU(shape);
            }
            if (name == "Logistic") {
                return new Logistic(shape);
            }
            if (name == "Softmax") {
                return new SoftMax(shape, axis);
            }
        }
};

#endif
