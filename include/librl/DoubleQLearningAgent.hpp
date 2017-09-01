#ifndef DOUBLEQLEARNING_HPP
#define DOUBLEQLEARNING_HPP

#include "RLAgent.hpp" // Base class: RLAgent

template<typename TState, typename TAction>
class DoubleQLearningAgent : public RLAgent<TState, TAction> {
public:

    DoubleQLearningAgent(
            Policy<TState, TAction>* pi,
            MDP<TState, TAction>* mdp,
            DoubleApproximator<TState, TAction>* dba,
            double discount_factor)
    : RLAgent<TState, TAction>(pi, mdp, dba, discount_factor) {
    }

    /**
     * @brief Method for telling the agent that the performed action is terminated
     * @param action the id of the performed action
     * @param reward the reward associated with the performed action
     */
    void learn(TState prev_state, TAction action, TState next_state, double reward) {
        double value = this->get_reinforcement(prev_state, action, next_state, reward);
        this->q->Q(prev_state, action, value);
    }

    std::string getName() const {
        return "Double Q-Learning";
    }

    void set_learning_parameters(std::vector<double> parameters) {
        this->gamma = parameters[1];
        this->alpha = parameters[0];
    }

    void reset() {
        this->stats = std::make_shared<AgentStatistics>();
        this->pi->reset();
        this->q->reset();
    }

    TAction choose_action() const {
        TAction action = this->pi->choose_action(this);
        return action;
    }

protected:

    double get_reinforcement(TState prev_state, TAction action, TState next_state, double reward) const {
        auto function_approximators = std::dynamic_pointer_cast<DoubleApproximator<TState, TAction>>(this->q)->get_FA_update_pair();
        auto fstar = function_approximators.first->argmax(next_state);
        double q_value = function_approximators.first->Q(prev_state, action);
        return reward + gamma * function_approximators.second->Q(next_state, fstar);
    }

};

#endif // DOUBLEQLEARNING_HPP
