#ifndef LOSS_H
#define LOSS_H

#include <vector>
#include "Tensor.cpp"

class Loss {
    Tensor error_deriv;
    public:
        Loss() {};
        double calculate_loss(Tensor &y_pred, &Tensor y_label);
        virtual double point_wise_func(double x_pred, double x_label);
        virtual double deriv(double x_pred, double x_label);
        Tensor back_propagate();
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
}

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
   
}

#endif
