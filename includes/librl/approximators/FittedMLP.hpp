//
// Created by xetql on 12.03.18.
//

#ifndef LIBLJ_FITTEDMLP_HPP
#define LIBLJ_FITTEDMLP_HPP

#include <mlpack/core.hpp>

#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>
#include <mlpack/core/optimizers/sgd/update_policies/vanilla_update.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>

#include <armadillo>

#include <type_traits>
#include <vector>

#include "FunctionApproximator.hpp"
#include "MLP.hpp"

namespace librl {
namespace approximator {
namespace action_value {
/**
 * NeuralNet approximator
 * @tparam OptimizerType Optimizer Type
 * @tparam ParametersType Parameters type of the neural net. See mlpack.
 */
template<typename OptimizerType, class... ParametersType>
class FittedMLP : public MLP<OptimizerType, ParametersType...> {
    const std::vector<int> actions;
public:
    FittedMLP(const std::vector<int>& action_space, OptimizerType opt, ParametersType... params) : actions(action_space), MLP<OptimizerType, ParametersType...>(opt, params...) {}
    FittedMLP(const std::vector<int>& action_space, OptimizerType opt) : actions(action_space), MLP<OptimizerType, ParametersType...>(opt)  {}

    virtual int argmax(const arma::mat &state, const std::vector<int> &available_actions) const {
        arma::mat response;
        int argmax = -1;
        double max=std::numeric_limits<double>::lowest();
        for(const int& action : actions){
            arma::mat in(state);
            in.insert_rows(in.n_rows, 1);
            in(in.n_rows - 1) = action;
            this->model->Predict(in, response);
            double tmp = response.at(0);
            if(tmp > max){
                max = tmp;
                argmax = action;
            }
        }
        assert(argmax >= 0);
        return argmax;
    }

    virtual void Q(const arma::mat &state, const int &action, double value) {
        arma::mat out = {0};
        out(0) = value;
        arma::mat in(state);
        in.insert_rows(in.n_rows, 1);
        in(in.n_rows - 1) = action;
        this->model->Train(in, out);
    }

    virtual double max(const arma::mat &state) const {
        arma::mat response;
        double max=std::numeric_limits<double>::lowest();
        for(const int& action : actions){
            arma::mat in(state);
            in.insert_rows(in.n_rows, 1);
            in(in.n_rows - 1) = action;
            this->model->Predict(in, response);
            double tmp = response.at(0);
            if(tmp > max){
                max = tmp;
            }
        }
        return max;
    }

    virtual double Q(const arma::mat &state, const int &action) const {
        arma::mat response;
        arma::mat in(state);
        in.insert_rows(in.n_rows, 1);
        in(in.n_rows - 1) = action;
        this->model->Predict(in, response);
        return response.at(0);
    }
};

}}}


#endif //LIBLJ_FITTEDMLP_HPP
