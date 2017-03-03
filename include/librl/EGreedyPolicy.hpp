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

    TAction choose_action(RLAgent<TState, TAction>* agent) {
        return eGreedyExploration(agent, this->E);
    }

    std::string getName() {
        return "E-Greedy Exploration";
    }

    void reset() {
    }

    std::map<TAction, double> get_probabilities(RLAgent<TState, TAction>* agent, TState state) {
        TAction bestAction = agent->q->argmax(state);
        std::map<TAction, double> probabilities;
        int nb_actions = agent->get_available_actions().size();
        for (auto const &action : agent->get_available_actions())
            probabilities[action] = (1 - E) / (double) (nb_actions) - 1;
        probabilities[bestAction] = E;
        return probabilities;
    }

    /**
     * @brief As a probability (E) of selecting the best otherwise it is random.
     * @param agent
     * @return 
     */
    TAction eGreedyExploration(RLAgent<TState, TAction>* agent, double E) {
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