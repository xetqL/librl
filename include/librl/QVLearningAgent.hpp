#ifndef QVLEARNINGAGENT_HPP
#define QVLEARNINGAGENT_HPP

#include "FunctionApproximator.hpp"

#include "RLAgent.hpp" // Base class: RLAgent

#include <map>

template<typename TState, typename TAction>
class QVLearningAgent : public RLAgent<TState, TAction> {
public:

    QVLearningAgent(
        Policy<TState, TAction>* pi,
        MDP<TState, TAction>* mdp,
        ActionValueApproximator<TState, TAction>* ava,
        StateValueApproximator<TState>* sva,
        double discount_factor)
    : RLAgent<TState, TAction>(pi, mdp, ava, discount_factor), v(sva) {
    }

    std::string getName() const {
        return "QVLearning";
    }

    /**
     * @brief Method for asking the RL agent to starting to perform an action
     * @return Gives the id of the action to perform
     */
    TAction choose_action() const {
        return this->pi->choose_action(this);
    }


    /**
     * @brief Method for telling the agent that the performed action is terminated
     * @param action the id of the performed action
     * @param reward the reward associated with the performed action
     */
    void learn(TState prev_state, TAction action, TState next_state, double reward) {
        //learn the previous state value via V
        this->v->V(prev_state, this->get_reinforcement(prev_state, action, next_state, reward));
        //learn the previous action value via Q
        this->q->Q(prev_state, action, this->get_reinforcement(prev_state, action, next_state, reward));
    }

    void reset() {
        this->stats = std::make_shared<AgentStatistics>();
        this->pi->reset();
    }

    void set_learning_parameters(std::vector<double> parameters) {
        this->gamma = parameters[1];
        this->q->set_learning_parameter(parameters[0]);
        this->v->set_learning_parameter(parameters[2]);
    }
protected:
    double get_reinforcement(TState prev_state, TAction action, TState next_state, double reward) const {
        return reward + this->gamma * this->v->V(next_state);
    }
    StateValueApproximator<TState>* v;
};
#endif // QVLEARNINGAGENT_HPP
