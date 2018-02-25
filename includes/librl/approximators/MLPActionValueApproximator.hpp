//
// Created by xetql on 23.02.18.
//

#ifndef LIBLJ_MLPACTIONVALUEAPPROXIMATOR_HPP
#define LIBLJ_MLPACTIONVALUEAPPROXIMATOR_HPP
#include <mlpack/core.hpp>

#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>
#include <mlpack/core/optimizers/sgd/update_policies/vanilla_update.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>

#include <armadillo>
#include <type_traits>
#include <vector>

#include "FunctionApproximator.hpp"

namespace librl { namespace approximator {

template<class ModelType,
         class OptimizerType = mlpack::optimization::RMSProp,
         class ApproximatorState = arma::mat,
         typename std::enable_if<
             std::is_same<ApproximatorState, std::vector<double>>::value ||
             std::is_same<ApproximatorState, arma::mat>::value
         >::type* = nullptr
>
class MLPActionValueApproximator : public ActionValueApproximator<arma::mat, int> {
    ModelType* model;
    OptimizerType opt;
    int max_actions;
public:

    MLPActionValueApproximator(
            ModelType* model,
            OptimizerType& optimizer,
            double alpha,
            int max_actions) :
            model(model),
            opt(optimizer),
            max_actions(max_actions),
            ActionValueApproximator<arma::mat, int>(alpha) {}

    void reset() {model->ResetParameters();};

    void set_learning_parameter(double alpha) {}

    int argmax(arma::mat state, std::vector<int> available_actions = std::vector<int>() ) const {
        arma::mat responses = arma::zeros(max_actions);
        const_cast<ModelType*>(model)->Predict(state, responses);//I thank a lot mlpack devs. for the non cost function.
        return responses.index_max();
    }

    void Q(arma::mat state, int action, double value) {
        arma::mat responses = arma::zeros(max_actions);
        model->Predict(state, responses);
        responses(action) = value;
        model->Train(state, responses, opt);
    }

    double max(arma::mat state) const {
        arma::mat responses = arma::zeros(max_actions);
        const_cast<ModelType*>(model)->Predict(state, responses);//I thank a lot mlpack devs. for the non cost function.
        return arma::max(responses).max();
    }

    double Q(arma::mat state, int action) const {
        arma::mat responses = arma::zeros(max_actions);
        const_cast<ModelType*>(model)->Predict(state, responses);//I thank a lot mlpack devs. for the non cost function.
        return responses(action);
    }
};

template<class ModelType,
         class OptimizerType
>
class MLPActionValueApproximator<ModelType, OptimizerType, std::vector<double>> : public ActionValueApproximator<std::vector<double>, int> {
    ModelType* model;
    OptimizerType opt;
    int max_actions;
public:

    MLPActionValueApproximator(
            ModelType* model,
            OptimizerType& optimizer,
            double alpha,
            int max_actions) :
            model(model),
            opt(optimizer),
            max_actions(max_actions),
            ActionValueApproximator<std::vector<double>, int>(alpha) {}

    void reset() {model->ResetParameters();};

    void set_learning_parameter(double alpha) {}

    int argmax(std::vector<double> state, std::vector<int> available_actions = std::vector<int>() ) const {
        arma::mat responses = arma::zeros(max_actions);
        arma::mat mstate(state);
        const_cast<ModelType*>(model)->Predict(mstate, responses);//I thank a lot mlpack devs. for the non cost function.
        return responses.index_max();
    }

    void Q(std::vector<double> state, int action, double value) {
        arma::mat responses = arma::zeros(max_actions);
        arma::mat mstate(state);
        model->Predict(mstate, responses);
        responses(action) = value;
        model->Train(mstate, responses, opt);
    }

    double max(std::vector<double> state) const {
        arma::mat responses = arma::zeros(max_actions);
        arma::mat mstate(state);
        const_cast<ModelType*>(model)->Predict(mstate, responses);//I thank a lot mlpack devs. for the non cost function.
        return arma::max(responses).max();
    }

    double Q(std::vector<double> state, int action) const {
        arma::mat responses = arma::zeros(max_actions);
        arma::mat mstate(state);
        const_cast<ModelType*>(model)->Predict(mstate, responses);//I thank a lot mlpack devs. for the non cost function.
        return responses(action);
    }
};


}}
#endif //LIBLJ_MLPACTIONVALUEAPPROXIMATOR_HPP
