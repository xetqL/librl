#ifndef DOUBLEQLEARNING_HPP
#define DOUBLEQLEARNING_HPP

#include "RLAgent.hpp" // Base class: RLAgent

template<typename TState, typename TAction>
class DoubleQLearningAgent : public RLAgent<TState, TAction> {
public:

    DoubleQLearningAgent(
            std::vector<double> params,
            std::shared_ptr<Policy<TState, TAction>> pi,
            std::shared_ptr<MDP<TState, TAction>> mdp,
            std::shared_ptr<DoubleApproximator<TState, TAction>> dba)
    : RLAgent<TState, TAction>(pi, mdp, dba) {
        this->gamma = params[1];
        this->alpha = params[0];
    }

    /**
     * @brief Method for telling the agent that the performed action is terminated
     * @param action the id of the performed action
     * @param reward the reward associated with the performed action
     */
    void learn(TState prev_state, TAction action, TState next_state, double reward) {
        std::static_pointer_cast<DoubleApproximator<TState, TAction>>(this->q)->approximate(prev_state, action, next_state, reward, this->gamma);
        //this->get_reinforcement(prev_state, action, next_state, reward); //Update Qa|b as a side effect
    }   

    std::string getName() {
        return "Double Q-Learning";
    }

    double get_reinforcement(TState prev_state, TAction action, TState next_state, double reward) {
        return 0.0;
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

    TAction choose_action() {
        TAction action = this->pi->choose_action(this);
        return action;
    }

private:

    double gamma;
    double alpha;
};

#endif // DOUBLEQLEARNING_HPP
