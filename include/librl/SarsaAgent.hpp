#ifndef SARSAAGENT_HPP
#define SARSAAGENT_HPP

#include "RLAgent.hpp" // Base class: RLAgent

template<typename TState, typename TAction>
class SarsaAgent : public RLAgent<TState, TAction> {
public:

    SarsaAgent(
        Policy<TState, TAction>* pi,
        MDP<TState, TAction>* mdp,
        ActionValueApproximator<TState, TAction>* ava,
        double discount_factor)
    : RLAgent<TState, TAction>(pi, mdp, ava, discount_factor) {
    }

    std::string getName() const {
        return "Sarsa";
    }

    /**
     * @brief Method for asking the RL agent to starting to perform an action
     * @return Gives the id of the action to perform
     */
    TAction choose_action() const {
        TAction action = this->pi->choose_action(this);
        return action;
    }

    /**
     * @brief Method for telling the agent that the performed action is terminated
     * @param action the id of the performed action
     * @param reward the reward associated with the performed action
     */
    void learn(TState prev_state, TAction action, TState next_state, double reward) {
        double value = this->get_reinforcement(prev_state, action, next_state, reward); //side effect : change the current state due to sarsa and set the next action
        this->q->Q(prev_state, action, value);
    }

    void reset() {
        this->stats = std::make_shared<AgentStatistics>();
        this->pi->reset();
        this->q->reset();
    }

    void set_learning_parameters(std::vector<double> parameters) {
        this->gamma = parameters[1];
    }
protected:
    double get_reinforcement(TState prev_state, TAction action, TState next_state, double reward) const {
        TAction futureAction = this->pi->choose_action(this);
        return reward + this->gamma * this->q->Q(next_state, futureAction);
    }

};

#endif // SARSAAGENT_HPP
