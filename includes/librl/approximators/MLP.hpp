//
// Created by xetql on 09.03.18.
//

#ifndef LIBLJ_FITTEDMLPAPPROXIMATOR_HPP
#define LIBLJ_FITTEDMLPAPPROXIMATOR_HPP

#include <mlpack/core.hpp>

#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>
#include <mlpack/core/optimizers/sgd/update_policies/vanilla_update.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>

#include <armadillo>

#include <type_traits>
#include <vector>

#include "FunctionApproximator.hpp"

namespace librl {
namespace approximator {
namespace action_value {
/**
 * Default implementation of Q Function Approximation through NeuralNet. Actions are modeled as list with indexes
 * @tparam OptimizerType Type of the optimizer used for training the NeuralNet
 * @tparam ParametersType MLPACK Inner Neural Net types, see mlpack.
 */
template<typename OptimizerType, class... ParametersType>
class MLP : public ActionValueApproximator<arma::mat, int> {
protected:
    std::unique_ptr<mlpack::ann::FFN<ParametersType... >> model;
    OptimizerType opt;
public:
    MLP(OptimizerType opt, ParametersType... params) : opt(opt), ActionValueApproximator<arma::mat, int>(opt.Alpha()) {
        model = std::move(std::make_unique<mlpack::ann::FFN<ParametersType...>>(params...));
    }
    MLP(OptimizerType opt) : opt(opt), ActionValueApproximator<arma::mat, int>(opt.Alpha()) {
        model = std::move(std::make_unique<mlpack::ann::FFN<ParametersType...>>());
    }

    template<class LayerType, class... Args>
    void add_layer(Args... args) {
        model->template Add<LayerType>(args...);
    }

    virtual void reset() { this->reset_model(); }

    virtual void reset_model(ParametersType... params) noexcept {
        this->model.reset(new mlpack::ann::FFN<ParametersType...>(params...));
    }
    virtual void reset_model() noexcept {
        this->model.reset(new mlpack::ann::FFN<ParametersType...>());
    }
    virtual void set_model(mlpack::ann::FFN<ParametersType...> *model) noexcept {
        this->model.reset(model);
    }

    /**
     * Return the output neuron that returns the highest response value
     * @param state The entry state (input)
     * @param available_actions The available actions, "output neurons"
     * @return action that returns the highest response value
     */
    virtual int argmax(const arma::mat& state, const std::vector<int>& available_actions = std::vector<int>() ) const {
        arma::mat responses;
        const_cast<mlpack::ann::FFN<ParametersType...>*>(model.get())->Predict(state, responses);//I thank a lot mlpack devs. for the non cost function.
        return responses.index_max();
    }

    /**
     * Train the neural net to match the output response (value) for a given action (output neuron)
     * @param state The entry state (input)
     * @param action The id of the output neuron
     * @param value The response value
     */
    virtual void Q(const arma::mat& state, const int& action, double value) {
        arma::mat responses;
        model->Predict(state, responses);
        responses(action) = value;
        model->Train(state, responses, opt);
    }

    /**
     * Return the highest response value.
     * @param state The entry state (input)
     * @return The highest output response
     */
    virtual double max(const arma::mat& state) const {
        arma::mat responses;
        const_cast<mlpack::ann::FFN<ParametersType...>*>(model.get())->Predict(state, responses);//I thank a lot mlpack devs. for the non const function.
        return arma::max(responses).max();
    }

    /**
     * Get the response value of a given action.
     * @param state The entry state (input)
     * @param action The output neuron of interest
     * @return The output response
     */
    virtual double Q(const arma::mat& state, const int& action) const {
        arma::mat responses;
        const_cast<mlpack::ann::FFN<ParametersType...>*>(model.get())->Predict(state, responses);//I thank a lot mlpack devs. for the non const function.
        return responses(action);
    }
};

}}}

#endif //LIBLJ_FITTEDMLPAPPROXIMATOR_HPP
