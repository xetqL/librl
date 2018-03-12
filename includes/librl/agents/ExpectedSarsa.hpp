#ifndef EXPECTEDSARSAAGENT_HPP
#define EXPECTEDSARSAAGENT_HPP

#include "RLAgent.hpp" // Base class: RLAgent
#include "../utils/array.hpp"

namespace librl { namespace agent {

template<typename TState, typename TAction>
class ExpectedSarsa : public RLAgent<TState, TAction> {
public:
    const std::string name = "Expected Sarsa";

    ExpectedSarsa(
            librl::policy::Policy<TState, TAction> *pi,
            librl::approximator::action_value::ActionValueApproximator<TState, TAction> *ava,
            double discount_factor)
            : RLAgent<TState, TAction>(pi, ava, discount_factor) {
    }

    TAction choose_action(const TState& state, const std::vector<TAction>& actions) {
        this->current_state = state;
        this->current_actions = actions;
        return this->pi->choose_action(this->q, actions, state);
    }

    void learn(TState prev_state, TAction action, TState next_state, double reward) {
        double value = this->get_reinforcement(prev_state, action, next_state, reward);
        this->q->Q(prev_state, action, value);
    }

    void reset() {
        this->pi->reset();
        this->q->reset();
    }

    void set_learning_parameters(std::vector<double> parameters) {
        this->gamma = parameters[1];
    }

protected:
    double get_reinforcement(TState prev_state, TAction action, TState next_state, double reward) const {
        double Vs = 0;
        auto probabilities = this->pi->get_probabilities(this->q, current_actions, next_state);
        for (auto const &actionProbabilities : probabilities) { //Weighted E(a | s)
            Vs += actionProbabilities.second * this->q->Q(next_state, actionProbabilities.first);
        }
        return reward + this->gamma * Vs;
    }
private:
    TState current_state;
    std::vector<TAction> current_actions;
};
}}
#endif // EXPECTEDSARSAAGENT_HPP
