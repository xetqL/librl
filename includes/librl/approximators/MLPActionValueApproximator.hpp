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

#include "FunctionApproximator.hpp"

namespace librl { namespace approximator {

template<class ModelType, class OptimizerType = mlpack::optimization::RMSProp>
class MLPActionValueApproximator : public ActionValueApproximator<arma::vec, int> {
    ModelType* model;
    OptimizerType opt;
    int max_actions;
public:

    MLPActionValueApproximator(
            ModelType* model,
            OptimizerType& optimizer,
            int max_actions,
            double alpha) :
            model(model),
            opt(optimizer),
            max_actions(max_actions),
            ActionValueApproximator<arma::vec, int>(alpha) {}

    void reset() {model.ResetParameters();};

    /**
     * There is no learning parameter when using neural network as a function
     * approximation
     * @param alpha
     */
    void set_learning_parameter(double alpha) {}

    /**
     * Get argmax_a Q(s,a), it may contains several indices
     * Query NN
     * @param state
     * @return argmax_a Q(s,a)
     */
    int argmax(arma::vec state, std::vector<int> available_actions = std::vector<int>() ) const {
        arma::vec responses = arma::zeros(max_actions);
        const_cast<ModelType*>(model)->Predict(state, responses);
        return arma::index_max(responses);
    }

    /**
     * Setter of the Q function
     * @param state
     * @param action
     * @param value
     */
    void Q(arma::vec state, int action, double value) {
        arma::vec responses = arma::zeros(max_actions);
        responses(action) = value;
        model->Train(state, responses, opt);
    }

    /**
     * @brief Get max Q(s,a)
     */
    double max(arma::vec state) const {
        arma::vec responses = arma::zeros(max_actions);
        const_cast<ModelType*>(model)->Predict(state, responses);//I thank a lot mlpack devs. for the non cost function.
        return arma::max(responses);
    }
    /**
    * @brief Get Q(s,a)
    */
    double Q(arma::vec state, int action) const {
        arma::vec responses = arma::zeros(max_actions);
        const_cast<ModelType*>(model)->Predict(state, responses);//I thank a lot mlpack devs. for the non cost function.
        return responses(action);
    }
};

}}
#endif //LIBLJ_MLPACTIONVALUEAPPROXIMATOR_HPP
