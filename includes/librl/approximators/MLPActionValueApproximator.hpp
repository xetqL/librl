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

    void reset() {};

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


/*
        fann_type in[net.get_num_input()];
        fann_type *run_out;

        for (size_t i = 0; i < net.get_num_input(); ++i) in[i] = state_as_vect[i];

        //predict value for state
        run_out = net.run(in);

        //find argmax in prediction value which is available in that state
*/
        return 1;
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
/*
        //verify sizes of input and topology
        assert(state_as_vect.size() == net.get_num_input());

        fann_type in[net.get_num_input()];
        fann_type *run_out, desired_out[net.get_num_output()];
        for (size_t i = 0; i < net.get_num_input(); ++i) {
            in[i] = state_as_vect[i];
        }

        //predict value for state
        run_out = net.run(in);

        for (size_t i = 0; i < net.get_num_output(); ++i) {
            if (i == action) desired_out[action] = value;
            else desired_out[i] = run_out[i];
        }

        net.train(in, desired_out);

        //std::cout <<"MSE " <<net.get_MSE() << std::endl;
        */
    }

    /**
     * @brief Get max Q(s,a)
     */
    double max(arma::vec state) const {
        arma::vec responses = arma::zeros(max_actions);
        const_cast<ModelType*>(model)->Predict(state, responses);
        return arma::max(responses);
        /*
        //verify sizes of input and topology
        assert(state_as_vect.size() == net.get_num_input());

        fann_type in[net.get_num_input()];
        fann_type *run_out;
        for (size_t i = 0; i < net.get_num_input(); ++i) in[i] = state_as_vect[i];
        //predict value for state
        run_out = net.run(in);
        //find argmax in prediction value which is available in that state
        return filtered_max(run_out, state, this->available_actions);*/
        return 0.0;
    }
    /**
    * @brief Get Q(s,a)
    */
    double Q(arma::vec state, int action) const {
        arma::vec responses = arma::zeros(max_actions);
        const_cast<ModelType*>(model)->Predict(state, responses);
        return responses(action);
        /*
        //verify sizes of input and topology
        assert(state_as_vect.size() == net.get_num_input());
        fann_type in[net.get_num_input()];
        fann_type *out;
        for (size_t i = 0; i < net.get_num_input(); ++i) in[i] = state_as_vect[i];
        //predict value for state
        out = net.run(in);
        //find argmax in prediction value which is available in that state
        return out[action];*/
        return 0.0;
    }
};

}}
#endif //LIBLJ_MLPACTIONVALUEAPPROXIMATOR_HPP
