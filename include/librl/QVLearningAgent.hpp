#ifndef QVLEARNINGAGENT_HPP
#define QVLEARNINGAGENT_HPP

#include "RLAgent.hpp" // Base class: RLAgent

#include <map>

template<typename TState, typename TAction>
class QVLearningAgent : public RLAgent<TState, TAction> {
public:

    QVLearningAgent(std::vector<double> params, std::shared_ptr<Policy<TState, TAction>> pi, std::shared_ptr<MDP<TState, TAction>> mdp, std::shared_ptr<ActionValueApproximator<TState, TAction>> ava, StateValueApproximator<TState> sva)
    : RLAgent<TState, TAction>(pi, mdp, ava), alpha(params[0]), gamma(params[1]), beta(params[2], V(sva)) {
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

    double Q(TState state, TAction action) {
        return this->_Q[state][action];
    }

    double maxQ(TState state) {
        double max = std::numeric_limits<double>::lowest();
        for(auto const &it : this->_Q[state]){
            if (max <= it.second)
                max = it.second;
        }
        return max;
    }

    std::vector<TAction> argmaxQ(TState state) {
        double max = this->maxQ(state);
        std::vector<TAction> idxV;
        for(auto const &it : this->_Q[state]){
            if (max == it.second){
                idxV.push_back(it.first);
            }
        }
        return idxV;
    }

    void Q(TState state, TAction action, double value) {
        this->_Q[state][action] = value;
    }
    
    double get_reinforcement_for_V(TState prev_state, TAction action, TState next_state, double reward){
        return reward + this->gamma * this->_V[next_state];
    }
    
    double get_reinforcement_for_Q(TState prev_state, TAction action, TState next_state, double reward){
        return reward + this->gamma * this->_V[next_state];
    }
    
    double __get_reinforcement(TState prev_state, TAction action, TState next_state, double reward) {
        return reward + this->gamma * this->_V[next_state];
        //return this->Q(prev_state, action) + this->alpha * (reward + this->gamma * this->_V[next_state] - this->Q(prev_state, action));
    }
    
    double get_reinforcement(TState prev_state, TAction action, TState next_state, double reward) {
        this->_V[prev_state] += beta * (reward + this->gamma * this->_V[next_state] - this->_V[prev_state]);
        return this->Q(prev_state, action) + this->alpha * (reward + this->gamma * this->_V[next_state] - this->Q(prev_state, action));
    }
    
    /**
     * @brief Method for telling the agent that the performed action is terminated
     * @param action the id of the performed action
     * @param reward the reward associated with the performed action
     */
    void learn(TState prev_state, TAction action, TState next_state, double reward){
        double value = this->__get_reinforcement(prev_state, action, next_state, reward);
        this->_V[prev_state] += beta * (value - this->_V[prev_state]); // TODO: replace by FunctionApproximator
        value = this->__get_reinforcement(prev_state, action, next_state, reward); 
        this->Q(prev_state, action, this->Q(prev_state, action) + this->alpha * (value - this->Q(prev_state, action)));
        //double value = this->get_reinforcement(prev_state, action, next_state, reward);
        //this->stats->update(this->current_state(), action, reward);
    }
    void reset() {
        this->stats = std::make_shared<AgentStatistics>();
        this->pi->reset();
        this->_Q.clear();
        this->_V.clear();
    }

    void set_learning_parameters(std::vector<double> parameters) {
        this->beta  = parameters[2];
        this->gamma = parameters[1];
        this->alpha = parameters[0];
    }
    std::map<TState, std::map<TAction, double>> _Q;
    std::map<TState, double> _V;
    double gamma;
    double beta;
    double alpha;
};
#endif // QVLEARNINGAGENT_HPP
