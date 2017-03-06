#ifndef EXPECTEDSARSAAGENT_HPP
#define EXPECTEDSARSAAGENT_HPP

#include "RLAgent.hpp" // Base class: RLAgent
#include "array.hpp"

template<typename TState, typename TAction>
class ExpectedSarsaAgent : public RLAgent<TState, TAction> {
public:

    ExpectedSarsaAgent(std::vector<double> params, std::shared_ptr<Policy<TState, TAction>> pi, std::shared_ptr<MDP<TState, TAction>> mdp, std::shared_ptr<ActionValueApproximator<TState, TAction>> ava)
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

    /**
     * @brief Method for telling the agent that the performed action is terminated
     * @param action the id of the performed action
     * @param reward the reward associated with the performed action
     */
    void learn(TState prev_state, TAction action, TState next_state, double reward) {
#if LOGGING_STATE_ENABLED
        printf("Agent %s has performed action %d in state %d and gain a reward of %e\n", this->uuid.c_str(), action, currentState, r);
#endif
        double value = this->__get_reinforcement(prev_state, action, next_state, reward);
        this->q->Q(prev_state, action, value);
        //this->stats->update(this->current_state(), action, reward); //TODO AgentStatistics UPDATE
    }

    std::string getName() {
        return "Expected Sarsa";
    }

    double __get_reinforcement(TState prev_state, TAction action, TState next_state, double reward) {
        double Vs = 0;
        std::map<TAction, double> probabilities = this->pi->get_probabilities(this, next_state);
        for (auto const &actionProbabilities : probabilities) { //Weighted E(a | s)
            Vs += actionProbabilities.second * this->q->Q(next_state, actionProbabilities.first);
        }
        return reward + this->gamma * Vs;
    }

    double get_reinforcement(TState prev_state, TAction action, TState next_state, double reward) {
        double Vs = 0;
        std::map<TAction, double> probabilities = this->pi->get_probabilities(this, next_state);
        for (auto const &actionProbabilities : probabilities) { //Weighted E(a | s)
            Vs += actionProbabilities.second * this->q->Q(next_state, actionProbabilities.first);
        }
        return reward + this->gamma * Vs;
    }

    void reset() {
        this->stats = std::make_shared<AgentStatistics>();
        this->pi->reset();
        this->q->reset();
    }

    void set_learning_parameters(std::vector<double> parameters) {
        this->gamma = parameters[1];
        this->alpha = parameters[0];
    }
    double alpha;
    double gamma;

};

#endif // EXPECTEDSARSAAGENT_HPP
