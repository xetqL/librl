#ifndef GREEDYPOLICY_HPP
#define GREEDYPOLICY_HPP
#include "Policy.hpp"
#include "util.hpp"

template<typename TState, typename TAction>
class GreedyPolicy : public Policy<TState, TAction> {
public:
    TAction choose_action(const RLAgent<TState, TAction>* agent) const {
        return greedyExploration(agent);
    }

    std::map<TAction, double> get_probabilities(const RLAgent<TState, TAction>* agent, TState state) const {
        double max = agent->q->max(state);
        std::map<TAction, double> probabilities;
        std::vector<TAction> actions = agent->get_available_actions();
        int number_of_optimal_action = 0;
        for (auto const &action : actions){
            if(agent->q->Q(state, action) == max) number_of_optimal_action++;
        }
        for (auto const &action : actions)
            probabilities[action] = agent->q->Q(state,action) == max ? 1.0 / number_of_optimal_action : 0.0;

        return probabilities;
    }


    std::string getName() const {
        return "Greedy Exploration";
    }

    void reset() {}

    /**
     * @brief Always take the best estimated reward
     * @param agent
     * @return best ID
     */
    TAction greedyExploration(const RLAgent<TState, TAction>* agent) const {
        //get either the best indice of a random one among the action space
        return agent->q->argmax(agent->current_state());
    }
};

#endif // GREEDYPOLICY_HPP
