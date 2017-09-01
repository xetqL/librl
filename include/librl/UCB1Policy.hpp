#ifndef UCB1POLICY_HPP
#define UCB1POLICY_HPP
#include "Policy.hpp"
#include <cassert>

template<typename TState, typename TAction>
class UCB1Policy : public Policy<TState, TAction> {
public:

    TAction choose_action(const RLAgent<TState, TAction>* agent) const {
        return ucb1Exploration(agent);
    }

    std::string getName() const {
        return "UCB1 Exploration";
    }

    void reset() {
    }

    std::vector<double> get_probabilities(const RLAgent<TState, TAction>* agent, TState state) const {
        const int nb_actions = agent->get_available_actions().size();
        std::vector<double> probabilities(nb_actions, 0);
        double ucbValue, max = std::numeric_limits<double>::lowest();
        int idx;
        for (size_t i = 0; i < nb_actions; i++) {
            ucbValue = agent->q->Q(agent->current_state(), i)
                    + std::sqrt((2.0 * std::log(agent->current_state())) / agent->stats->numberOfTimeAction[agent->current_state()][i]);
            if (ucbValue > max) {
                max = ucbValue;
                idx = i;
            }
        }
        probabilities[idx] = 1.0;
        return probabilities;
    }

    TAction ucb1Exploration(const RLAgent<TState, TAction>* agent) const {
        double max = std::numeric_limits<double>::lowest(), ucbValue;
        const int nb_actions = agent->get_available_actions().size(),
                current_state = agent->current_state(),
                number_of_steps = agent->stats->numberOfStepsUntilNow;
        int idx = -1;
        for (int i = 0; i < nb_actions; i++) {
            ucbValue = agent->q->Q(current_state, i) + std::sqrt((2.0 * std::log(number_of_steps)) / agent->stats->numberOfTimeAction[current_state][i]);
            if (ucbValue > max) {
                max = ucbValue;
                idx = i;
            }
        }
        assert(0 <= idx && idx < nb_actions);
        return idx;
    }
};

#endif // UCB1POLICY_HPP
