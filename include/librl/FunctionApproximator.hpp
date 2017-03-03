/* 
 * File:   FunctionApproximator.hpp
 * Author: xetql
 *
 * Created on February 20, 2017, 5:05 PM
 */

#ifndef FUNCTIONAPPROXIMATOR_HPP
#define FUNCTIONAPPROXIMATOR_HPP
#include <vector>
#include <set>
#include <map>
#include <limits>
#include <iostream>
#include <utility>

#include "doublefann.h"
#include "fann_cpp.h"

#include "util.hpp"

template<typename TState, typename TAction>
class ActionValueApproximator {
public:

    ActionValueApproximator(double _alpha) : alpha(_alpha) {
    }

    ActionValueApproximator() : alpha(1.0) {
    }
    /**
     * Get argmax_a Q(s,a), it may contains several indices
     * @param state
     * @return argmax_a Q(s,a)
     */
    virtual TAction argmax(TState state) = 0;
    /**
     * Setter of the Q function
     * @param state
     * @param action
     * @param value
     */
    virtual void Q(TState state, TAction action, double value) = 0;
    /**
     * @brief Get max Q(s,a)
     */
    virtual double max(TState state) = 0;
    /**
     * @brief Get Q(s,a)
     */
    virtual double Q(TState state, TAction action) = 0;
    /**
     * @brief reset the function approximation
     */
    virtual void reset() = 0;

    void set_learning_parameter(double alpha) {
        this->alpha = alpha;
    }
protected:
    double alpha;
};

template<typename TState, typename TAction>
class DoubleApproximator : public ActionValueApproximator<TState, TAction> {
public:

    DoubleApproximator() {
    };

    TAction argmaxQa(TState state) {
        return this->argmax(this->Qa, state);
    }

    TAction argmaxQb(TState state) {
        return this->argmax(this->Qb, state);
    }

    /**
     * Get argmax_a Q(s,a), it may contains several indices
     * @param state
     * @return argmax_a Q(s,a)
     */
    TAction argmax(TState state) {
        double max = this->max(state);
        std::set<TAction> all;
        for (auto const &actionValue : this->Qa[state]) {
            if (max == this->Q(state, actionValue.first)) {
                all.insert(actionValue.first);
            }
        }
        for (auto const &actionValue : this->Qb[state]) {
            // TODO: reduce complexity
            if (max == actionValue.second) {
                all.insert(actionValue.first);
            }
        }
        return *select_randomly(all.begin(), all.end());
    }

    /**
     * Setter of the Q function
     * @param state
     * @param action
     * @param value
     */
    void Q(TState state, TAction action, double value) {
        ;
    }

    /**
     * @brief Get max Q(s,a)
     */
    double max(TState state) {
        double max = std::numeric_limits<double>::lowest();
        std::set<TAction> all_actions;
        for (auto const &actionValue : this->Qa[state]) all_actions.insert(actionValue.first);
        for (auto const &actionValue : this->Qb[state]) all_actions.insert(actionValue.first);
        for (auto const &action : all_actions) {
            if (max <= this->Q(state, action))
                max = this->Q(state, action);
        }
        return max;
    }

    double merge(double QaV, double QbV) {
        return (QaV + QbV) / 2.0;
    }

    double Q(TState state, TAction action) {
        if (this->Qa[state].find(action) != this->Qa[state].end()) {
            if (this->Qb[state].find(action) != this->Qb[state].end())
                return merge(this->Qa[state][action], this->Qb[state][action]);
            else
                return this->Qa[state][action];
        } else if (this->Qb[state].find(action) != this->Qb[state].end()) {
            return this->Qb[state][action];
        } else
            return 0;
    }

    /**
     * @brief reset the function approximation
     */
    void reset() {
        this->Qa.clear();
        this->Qb.clear();
    }

    TAction argmax(std::map<TState, std::map<TAction, double>> selectedQ, TState state) {
        double max = std::numeric_limits<double>::lowest();
        TAction id;
        for (auto const &actionValue : selectedQ[state]) {
            if (max < actionValue.second) max = actionValue.second;
            id = actionValue.first;
        }
        return id;
    }
    std::map<TState, std::map<TAction, double>> Qa, Qb;
};

template<typename TState, typename TAction>
class AfterstateValueApproximator : public ActionValueApproximator<TState, TAction> {
public:

    AfterstateValueApproximator(double _alpha, std::function< std::vector<TAction>(TState) > _available_actions, std::function<TState(TState, TAction) > _state_transition) : ActionValueApproximator<TState, TAction>(_alpha), available_actions(_available_actions), state_transition(_state_transition) {
    };

    AfterstateValueApproximator(std::function< std::vector<TAction>(TState) > _available_actions, std::function<TState(TState, TAction) > _state_transition) : ActionValueApproximator<TState, TAction>(0.1), available_actions(_available_actions), state_transition(_state_transition) {
    };

    char piece_to_char(int p) {
        switch (p) {
            case 1:
                return 'X';
            case 2:
                return 'O';
            default:
                return ' ';
        }
    }

    void print_tictactoe(TState m) {
        for (auto const &row : m) {
            for (auto const &cell : row) {
                std::cout << '|' << piece_to_char(cell);
            }
            std::cout << '|' << std::endl;
        }
    }

    /**
     * Get argmax_a Q(s,a), it may contains several indices
     * @param state
     * @return argmax_a Q(s,a)
     */
    TAction argmax(TState state) {
        double max = this->max(state);

        std::vector<TAction> all_actions;

        for (auto const &action : available_actions(state)) {
            if (max == this->Q(state, action)) {
                all_actions.push_back(action);
            }
        }
        //std::cout << "Random between : " << all_actions.size() << std::endl;
        //if(all_actions.size() == 9) { std::cout<<max;print_tictactoe(state); }
        //std::cout << (all_actions.begin() == all_actions.end()) << std::endl;
        return *select_randomly(all_actions.begin(), all_actions.end());
    }

    /**
     * Setter of the Q function with value from RL agent
     * @param state
     * @param action
     * @param value
     */
    void Q(TState state, TAction action, double value) {
        //std::cout << (this->_Q.find(state) != this->_Q.end()) << std::endl;
        /*std::map<TAction, double> action_value = this->_Q[state];
        std::cout << action_value[action] << std::endl;
        action_value[action] = value;
        std::cout << action_value[action] << std::endl;*/
        /*action_value.erase(action);
        action_value.insert(std::make_pair(action, value));
        this->_Q.erase(state);
        this->_Q.insert(std::make_pair(state, action_value));
         */
        this->_afterstate[state_transition(state, action)] = this->Q(state, action) + this->alpha * (value - this->Q(state, action));
        this->_Q[state][action] = this->_afterstate[state_transition(state, action)];
        //std::cout << "Q value is updated and is now equal to : " << this->_Q[state][action] << std::endl;
    }

    /**
     * @brief Get max_a Q(s,a)
     */
    double max(TState state) {
        double max = available_actions(state).size() > 0 ? std::numeric_limits<double>::lowest() : 1.0;
        //std::cout << "==========================" <<std::endl;
        for (auto const &action : available_actions(state)) {
            double qv = this->Q(state, action);
            if (max <= qv) max = qv;
        }
        //std::cout << "MAX = "<< max << std::endl <<"==========================" <<std::endl;
        return max;
    }

    /**
     * Get the Expected Reward for a in s.
     * @param state
     * @param action
     * @return 
     */
    double Q(TState state, TAction action) {

        if (_afterstate.find(state_transition(state, action)) == _afterstate.end())
            _afterstate[state_transition(state, action)] = 1.0;

        if (this->_Q.find(state) == this->_Q.end())
            this->_Q[state] = std::map<TAction, double>();

        if (this->_Q[state].find(action) == this->_Q[state].end())
            this->_Q[state][action] = _afterstate[state_transition(state, action)];

        return this->_Q[state][action];
    }

    /**
     * @brief reset the function approximation
     */
    void reset() {
        this->_Q.clear();
    }

    bool has_seen_state(TState s) {
        return (this->_Q.find(s) != this->_Q.end());
    }

    int number_of_explored_states() {
        return this->_Q.size();
    }

    std::map<TState, std::map<TAction, double> > _Q;

    std::map<TState, double> _afterstate;

    std::function<std::vector<TAction>(TState) > available_actions;
    std::function<TState(TState, TAction) > state_transition;
};

template<typename TState, typename TAction>
class ArrayActionValueApproximator : public ActionValueApproximator<TState, TAction> {
public:

    ArrayActionValueApproximator(double _alpha, std::function< std::vector<TAction>(TState) > _available_actions) : ActionValueApproximator<TState, TAction>(_alpha), available_actions(_available_actions) {
    };

    ArrayActionValueApproximator(std::function< std::vector<TAction>(TState) > _available_actions) : ActionValueApproximator<TState, TAction>(0.1), available_actions(_available_actions) {
    };

    /**
     * Get argmax_a Q(s,a), it may contains several indices
     * @param state
     * @return argmax_a Q(s,a)
     */
    TAction argmax(TState state) {
        double max = this->max(state);

        std::vector<TAction> all_actions;

        for (auto const &action : available_actions(state)) {
            if (max == this->Q(state, action)) {
                all_actions.push_back(action);
            }
        }
        //std::cout << "Random between : " << all_actions.size() << std::endl;
        //if(all_actions.size() == 9) { std::cout<<max;print_tictactoe(state); }
        //std::cout << (all_actions.begin() == all_actions.end()) << std::endl;
        return *select_randomly(all_actions.begin(), all_actions.end());
    }

    /**
     * Setter of the Q function with value from RL agent
     * @param state
     * @param action
     * @param value
     */
    void Q(TState state, TAction action, double value) {
        this->_Q[state][action] = this->Q(state, action) + this->alpha * (value - this->Q(state, action));
    }

    /**
     * @brief Get max_a Q(s,a)
     */
    double max(TState state) {
        double max = available_actions(state).size() > 0 ? std::numeric_limits<double>::lowest() : 1.0;
        for (auto const &action : available_actions(state)) {
            double qv = this->Q(state, action);
            if (max <= qv) max = qv;
        }
        return max;
    }

    /**
     * Get the Expected Reward for a in s.
     * @param state
     * @param action
     * @return 
     */
    double Q(TState state, TAction action) {
        if (this->_Q.find(state) == this->_Q.end())
            this->_Q[state] = std::map<TAction, double>();
        if (this->_Q[state].find(action) == this->_Q[state].end())
            this->_Q[state][action] = 0.0;
        return this->_Q[state][action];
    }

    /**
     * @brief reset the function approximation
     */
    void reset() {
        this->_Q.clear();
    }

    bool has_seen_state(TState s) {
        return (this->_Q.find(s) != this->_Q.end());
    }

    int number_of_explored_states() {
        return this->_Q.size();
    }

    std::map<TState, std::map<TAction, double> > _Q;
    std::function<std::vector<TAction>(TState) > available_actions;
};

template<typename TState>
class MLPActionValueApproximator : public ActionValueApproximator<TState, int> {
public:

    MLPActionValueApproximator(int _input_neurons,
            int _hidden_neurons,
            int _output_neurons,
            std::function< std::vector<int>(TState) > _available_actions)
    : available_actions(_available_actions) {
        if (net.create_standard(9, _input_neurons,
                _hidden_neurons,
                2 * _hidden_neurons,
                _hidden_neurons,
                _hidden_neurons,
                2 * _hidden_neurons,
                _hidden_neurons,
                _hidden_neurons,
                _output_neurons)) {
            std::cout << "Network created" << std::endl;
            net.set_activation_function_hidden(FANN::LINEAR);
            net.set_activation_function_output(FANN::LINEAR);
            net.randomize_weights(-1, 1);
            net.print_parameters();
        } else {
            std::cout << "Network not created" << std::endl;
        }
    }

    MLPActionValueApproximator(int _input_neurons,
            int _hidden_neurons,
            int _output_neurons,
            std::function< std::vector<double>(TState) > state_to_vector,
            std::function< std::vector<int>(TState) > _available_actions)
    : available_actions(_available_actions), stov(state_to_vector) {
        if (net.create_standard(9, _input_neurons,
                _hidden_neurons,
                2 * _hidden_neurons,
                _hidden_neurons,
                _hidden_neurons,
                2 * _hidden_neurons,
                _hidden_neurons,
                _hidden_neurons,
                _output_neurons)) {
            std::cout << "Network created" << std::endl;
            net.set_activation_function_hidden(FANN::SIGMOID);
            net.set_activation_function_output(FANN::LINEAR);
            net.randomize_weights(-1, 1);
            net.print_parameters();
        } else {
            std::cout << "Network not created" << std::endl;
        }
    }

    /**
     * TODO: Randomize the weights of the network
     */
    void reset() {
        net.randomize_weights(-1, 1);
    };

    /**
     * There is no learning parameter when using neural network as a function 
     * approximation
     * @param alpha
     */
    void set_learning_parameter(double alpha) {
        net.set_learning_rate(alpha);
    }

    /**
     * Get argmax_a Q(s,a), it may contains several indices
     * Query NN 
     * @param state
     * @return argmax_a Q(s,a)
     */
    int argmax(TState state) {
        std::vector<double> state_as_vect = stov(state);

        //verify sizes of input and topology
        assert(state_as_vect.size() == net.get_num_input());

        fann_type in[net.get_num_input()];
        fann_type *run_out;

        for (size_t i = 0; i < net.get_num_input(); ++i) in[i] = state_as_vect[i];

        //predict value for state
        run_out = net.run(in);

        //find argmax in prediction value which is available in that state
        return filtered_argmax(run_out, state, available_actions);
    }

    /**
     * Setter of the Q function
     * @param state
     * @param action
     * @param value
     */
    void Q(TState state, int action, double value) {
        std::vector<double> state_as_vect = stov(state);

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
    }

    /**
     * @brief Get max Q(s,a)
     */
    double max(TState state) {
        std::vector<double> state_as_vect = stov(state);

        //verify sizes of input and topology
        assert(state_as_vect.size() == net.get_num_input());

        fann_type in[net.get_num_input()];
        fann_type *run_out;
        for (size_t i = 0; i < net.get_num_input(); ++i) in[i] = state_as_vect[i];
        //predict value for state
        run_out = net.run(in);
        //find argmax in prediction value which is available in that state
        return filtered_max(run_out, state, available_actions);
    }

    int filtered_argmax(fann_type* matrix, TState state, std::function<std::vector<int>(TState) > A) {
        double max = filtered_max(matrix, state, A);
        std::vector<int> actions = A(state), argsmax;
        for (auto const &action : actions) {
            if (matrix[action] == max) argsmax.push_back(action);
        }
        return *select_randomly(argsmax.begin(), argsmax.end());
    }

    double filtered_max(fann_type* matrix, TState state, std::function<std::vector<int>(TState) > A) {
        std::vector<int> actions = A(state);
        double max = std::numeric_limits<double>::lowest();
        for (auto const &action : actions) {
            if (matrix[action] > max) max = matrix[action];
        }
        return max;
    }

    /**
     * @brief Get Q(s,a)
     */
    double Q(TState state, int action) {
        std::vector<double> state_as_vect = stov(state);

        //verify sizes of input and topology
        assert(state_as_vect.size() == net.get_num_input());

        fann_type in[net.get_num_input()];
        fann_type *out;

        for (size_t i = 0; i < net.get_num_input(); ++i) in[i] = state_as_vect[i];

        //predict value for state
        out = net.run(in);

        //find argmax in prediction value which is available in that state
        return out[action];
    }

    std::function<std::vector<int>(TState) > available_actions;

    /**
     * By default we assume that the state is a vectorial representation
     * @param state
     * @return the state without doing anything
     */
    std::vector<double> it_is_a_vector(TState state) {
        return state;
    }

    std::function< std::vector<double>(TState) > stov = it_is_a_vector;

    FANN::neural_net net;
};

template<typename TState>
class StateValueApproximator {
public:
    /**
     * Get argmax_a Q(s,a), it may contains several indices
     * @param state
     * @return argmax_a Q(s,a)
     */
    virtual std::vector<TState> argmaxV() = 0;
    /**
     * Setter of the Q function
     * @param state
     * @param action
     * @param value
     */
    virtual void V(TState state, double value) = 0;
    /**
     * @brief Get max Q(s,a)
     */
    virtual double max() = 0;
    /**
     * @brief Get Q(s,a)
     */
    virtual double V(TState state) = 0;
    /**
     * @brief reset the function approximation
     */
    virtual double reset() = 0;
};

template<typename TState>
class ArrayStateValueApproximator : public StateValueApproximator<TState> {
public:

    ArrayStateValueApproximator() {
    };

    /**
     * Get argmax_a Q(s,a), it may contains several indices
     * @param state
     * @return argmax_a Q(s,a)
     */
    std::vector<TState> argmax() {
        double max = this->max();
        std::vector<TState> idxV;
        for (auto const &actionValue : this->_V) {
            if (max == actionValue.second) {
                idxV.push_back(actionValue.first);
            }
        }
        return idxV;
    }

    /**
     * Setter of the Q function
     * @param state
     * @param action
     * @param value
     */
    void V(TState state, double value) {
        this->_V[state] = value;
    }

    /**
     * @brief Get max Q(s,a)
     */
    double max(TState state) {
        double max = std::numeric_limits<double>::lowest();
        for (auto const &actionValue : this->_V) {
            if (max <= actionValue.second)
                max = actionValue.second;
        }
        return max;
    }

    /**
     * Get the Expected Reward for a in s.
     * @param state
     * @param action
     * @return 
     */
    double V(TState state) {
        return this->_V[state];
    }

    /**
     * @brief reset the function approximation
     */
    void reset() {
        this->_V.clear();
    }

    std::map<TState, double> _V;
};

#endif /* FUNCTIONAPPROXIMATOR_HPP */