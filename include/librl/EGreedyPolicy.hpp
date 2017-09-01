#ifndef EGREEDYPOLICY_HPP
#define EGREEDYPOLICY_HPP
#include "Policy.hpp"
#include "GreedyPolicy.hpp"
#include "util.hpp"
#include <vector>

template<typename TState, typename TAction>
class EGreedyPolicy : public Policy<TState, TAction> {
public:

    EGreedyPolicy(double E) {
        this->E = E;
    }

    TAction choose_action(const RLAgent<TState, TAction>* agent) const {
        return eGreedyExploration(agent, this->E);
    }

    std::string getName() const {
        return "E-Greedy Exploration";
    }

    void reset() {
    }

    std::map<TAction, double> get_probabilities(const RLAgent<TState, TAction>* agent, TState state) const {
        std::vector<TAction> actions = agent->get_available_actions();
        double max = agent->q->max(state);
        int number_of_optimal_action = 0;

        for (auto const &action : actions)
            if (agent->q->Q(state, action) == max) number_of_optimal_action++;

        std::map<TAction, double> probabilities;
        for (auto const &action : actions) {
            probabilities[action] = agent->q->Q(state, action) == max ?
                    (1 - E) / ((double) number_of_optimal_action) : E / ((double) actions.size() - number_of_optimal_action);
        }

        return probabilities;
    }

    /**
     * @brief As a probability (E) of selecting the best otherwise it is random.
     * @param agent
     * @return
     */
    TAction eGreedyExploration(const RLAgent<TState, TAction>* agent, double E) const {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        if (dis(gen) > E) {
            GreedyPolicy<TState, TAction> g;
            return g.choose_action(agent);
        } else {
            std::vector<TAction> actions = agent->get_available_actions();
            return *select_randomly(actions.begin(), actions.end());
        }
    }

protected:
    double E;
};
#endif // EGREEDYPOLICY_HPP
