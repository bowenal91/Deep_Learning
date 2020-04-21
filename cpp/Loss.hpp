#ifndef LOSS_H
#define LOSS_H

#include "Tensor.hpp"
#include <vector>

class Loss {
    std::vector<Tensor> error_out;
    public:
        Loss() {};
        double calculate_loss(std::vector<Tensor> &y_pred, std::vector<Tensor> &y_label);
        double calculate_loss(Tensor &y_pred, Tensor &y_label);
        virtual double point_wise_func(double x_pred, double x_label) {};
        virtual double deriv(double x_pred, double x_label) {};
        std::vector<Tensor> back_propagate();
        Tensor calc_deriv(Tensor &y_pred, Tensor &y_label, double prefactor);
};

class MSE : public Loss {
    public:
        MSE() : Loss() {};
        double point_wise_func(double x_pred, double x_label) override;
        double deriv(double x_pred, double x_label) override;

};

class MAE : public Loss {
    public:
        MAE() : Loss() {};
        double point_wise_func(double x_pred, double x_label) override;
        double deriv(double x_pred, double x_label) override;
};

class Binary_Cross_Entropy : public Loss {
    public:
        Binary_Cross_Entropy() : Loss() {};
        double point_wise_func(double x_pred, double x_label) override;
        double deriv(double x_pred, double x_label) override;

};

class Categorical_Cross_Entropy : public Loss {
     public:
        Categorical_Cross_Entropy() : Loss() {};
        double point_wise_func(double x_pred, double x_label) override;
        double deriv(double x_pred, double x_label) override;
   
};

#endif
