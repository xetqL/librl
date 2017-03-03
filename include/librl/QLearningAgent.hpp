#ifndef QLEARNING_AGENT
#define QLEARNING_AGENT

#include "RLAgent.hpp"
#include "FunctionApproximator.hpp"
#include <iostream>
#include <limits>
#include <map>

template<typename TState, typename TAction>
class QLearningAgent : public RLAgent<TState, TAction> {
public:

    QLearningAgent(std::vector<double> params, std::shared_ptr<Policy<TState, TAction>> pi, std::shared_ptr<MDP<TState, TAction>> mdp, std::shared_ptr<ActionValueApproximator<TState, TAction>> ava)
    : RLAgent<TState, TAction>(pi, mdp, ava), alpha(params[0]), gamma(params[1]) {
    }

    /**
     * @brief Method for asking the RL agent to starting to perform an action
     * @return Gives the id of the action to perform
     */
    TAction choose_action() {
        TAction action = this->pi->choose_action(this);
        return action;
    }

    std::string getName() {
        return "QLearning";
    }

    void reset() {
        this->stats = std::make_shared<AgentStatistics>();
        this->pi->reset();
        this->q.reset();
    }

    void set_learning_parameters(std::vector<double> parameters) {
        this->gamma = parameters[1];
        this->alpha = parameters[0];
    }

    double __get_reinforcement(TState prev_state, TAction action, TState next_state, double reward) {
        //std::cout << this->q->max(next_state) << std::endl; 
        return reward + this->gamma * this->q->max(next_state);
    }

    /**
     * DEPRECATED; DO NOT USE
     * @param prev_state
     * @param action
     * @param next_state
     * @param reward
     * @return 
     */
    double get_reinforcement(TState prev_state, TAction action, TState next_state, double reward) {
        return this->q->Q(prev_state, action) + this->alpha * (reward + this->gamma * this->q->max(next_state) - this->q->Q(prev_state, action));
    }

    /**
     * @brief Method for telling the agent that the performed action is terminated
     * @param action the id of the performed action
     * @param reward the reward associated with the performed action
     */
    void learn(TState prev_state, TAction action, TState next_state, double reward) {
        double value = this->__get_reinforcement(prev_state, action, next_state, reward);
        this->q->Q(prev_state, action, value);
        //this->stats->update(this->current_state(), action, reward);
    }
    double alpha = 0.0;
    double gamma = 0.0;
};
#endif
