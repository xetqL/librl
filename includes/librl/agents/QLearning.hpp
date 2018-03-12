#ifndef QLEARNING_AGENT
#define QLEARNING_AGENT

#include "RLAgent.hpp"
#include "../approximators/FunctionApproximator.hpp"
#include <iostream>
#include <limits>
#include <map>
namespace librl { namespace agent {

template<typename TState, typename TAction>
class QLearning : public RLAgent<TState, TAction> {
public:
    const std::string name = "QLearning";

    QLearning(librl::policy::Policy<TState, TAction> *pi,
              librl::approximator::action_value::ActionValueApproximator<TState, TAction> *ava,
              double discount_factor) :
              RLAgent<TState, TAction>(pi, ava, discount_factor) {
    }

    TAction choose_action(const TState& state, const std::vector<TAction>& actions) {
        TAction action = this->pi->choose_action(this->q, actions, state);
        return action;
    }

    void reset() {
        this->pi->reset();
        this->q->reset();
    }

    void set_learning_parameters(std::vector<double> parameters) {
        this->gamma = parameters[1];
    }

    void learn(TState prev_state, TAction action, TState next_state, double reward) {
        double value = this->get_reinforcement(prev_state, action, next_state, reward);
        this->q->Q(prev_state, action, value);
    }

protected:
    double get_reinforcement(TState prev_state, TAction action, TState next_state, double reward) const {
        return reward + this->gamma * this->q->max(next_state);
    }
};
}}
#endif
