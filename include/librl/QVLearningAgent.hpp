#ifndef QVLEARNINGAGENT_HPP
#define QVLEARNINGAGENT_HPP

#include "FunctionApproximator.hpp"

#include "RLAgent.hpp" // Base class: RLAgent

#include <map>

template<typename TState, typename TAction>
class QVLearningAgent : public RLAgent<TState, TAction> {
public:

    QVLearningAgent(
            std::vector<double> params,
            std::shared_ptr<Policy<TState, TAction>> pi,
            std::shared_ptr<MDP<TState, TAction>> mdp,
            std::shared_ptr<ActionValueApproximator<TState, TAction>> ava,
            std::shared_ptr<StateValueApproximator<TState>> sva)
    : RLAgent<TState, TAction>(pi, mdp, ava), alpha(params[0]), gamma(params[1]), beta(params[2]), q(ava), v(sva) {
    }

    std::string getName() {
        return "QVLearning";
    }

    /**
     * @brief Method for asking the RL agent to starting to perform an action
     * @return Gives the id of the action to perform
     */
    TAction choose_action() {
        return this->pi->choose_action(this);
    }

    double __get_reinforcement(TState prev_state, TAction action, TState next_state, double reward) {
        return reward + this->gamma * this->v->V(next_state);
        //return this->Q(prev_state, action) + this->alpha * (reward + this->gamma * this->_V[next_state] - this->Q(prev_state, action));
    }

    double get_reinforcement(TState prev_state, TAction action, TState next_state, double reward) {
        //this->_V[prev_state] += beta * (reward + this->gamma * this->_V[next_state] - this->_V[prev_state]);
        return 0.0; //this->Q(prev_state, action) + this->alpha * (reward + this->gamma * this->_V[next_state] - this->Q(prev_state, action));
    }

    /**
     * @brief Method for telling the agent that the performed action is terminated
     * @param action the id of the performed action
     * @param reward the reward associated with the performed action
     */
    void learn(TState prev_state, TAction action, TState next_state, double reward) {
        //learn the previous state value via V
        this->v->V(prev_state, this->__get_reinforcement(prev_state, action, next_state, reward));
        //learn the previous action value via Q
        this->q->Q(prev_state, action, this->__get_reinforcement(prev_state, action, next_state, reward));
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
    std::shared_ptr<StateValueApproximator<TState> > v;

    std::shared_ptr<ActionValueApproximator<TState, TAction> > q;

    double gamma;

};
#endif // QVLEARNINGAGENT_HPP
