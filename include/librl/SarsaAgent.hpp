#ifndef SARSAAGENT_HPP
#define SARSAAGENT_HPP

#include "RLAgent.hpp" // Base class: RLAgent

template<typename TState, typename TAction>
class SarsaAgent : public RLAgent<TState, TAction> {
public:
    
    SarsaAgent(
            std::vector<double> params, 
            std::shared_ptr< Policy<TState, TAction> > pi, 
            std::shared_ptr<MDP<TState, TAction>> mdp, 
            std::shared_ptr<ActionValueApproximator<TState, TAction>> ava) 
    : RLAgent<TState, TAction>(pi, mdp, ava) {
        this->gamma = params[1];
        this->alpha = params[0];
    }

    std::string getName() {
        return "Sarsa";
    }

    /**
     * @brief Method for asking the RL agent to starting to perform an action
     * @return Gives the id of the action to perform
     */
    TAction choose_action() {
        TAction action = this->pi->choose_action(this);
        return action;
    }

    double __get_reinforcement(TState prev_state, TAction action, TState next_state, double reward) {
        TAction futureAction = this->pi->choose_action(this);
        return reward + this->gamma * this->q->Q(next_state, futureAction);
    }
    
    double get_reinforcement(TState prev_state, TAction action, TState next_state, double reward) {
        TAction futureAction = this->pi->choose_action(this);
        return this->q->Q(prev_state, action) + this->alpha * (reward + this->gamma * this->q->Q(next_state, futureAction) - this->q->Q(prev_state, action));
    }

    /**
     * @brief Method for telling the agent that the performed action is terminated
     * @param action the id of the performed action
     * @param reward the reward associated with the performed action
     */
    void learn(TState prev_state, TAction action, TState next_state, double reward) {
        double value = this->__get_reinforcement(prev_state, action, next_state, reward); //side effect : change the current state due to sarsa and set the next action
        this->q->Q(prev_state, action, value);
    }

    void reset() {
        this->stats = std::make_shared<AgentStatistics>();
        this->pi->reset();
        this->q->reset();
    }

    void set_learning_parameters(std::vector<double> parameters) {
        this->alpha = parameters[0];
        this->gamma = parameters[1];
    }
    double gamma;
    double alpha;
    int plannedAction = 0;
};

#endif // SARSAAGENT_HPP
