#ifndef GREEDYPOLICY_HPP
#define GREEDYPOLICY_HPP
#include "Policy.hpp"
#include "util.hpp"
template<typename TState, typename TAction>
class GreedyPolicy : public Policy<TState, TAction> {
public:
    TAction choose_action(RLAgent<TState, TAction>* agent) {
        return greedyExploration(agent);
    }
    
    std::map<TAction, double> get_probabilities(RLAgent<TState, TAction>* agent, TState state) {
        TAction bestAction = agent->q->argmax(state);
        std::map<TAction, double> probabilities;
        for (auto const &action : agent->get_available_actions())
            probabilities[action] = 0;
        probabilities[bestAction] = 1.0;
        return probabilities;
    }


    std::string getName() {
        return "Greedy Exploration";
    }

    void reset() {}

    /**
     * @brief Always take the best estimated reward
     * @param agent
     * @return best ID
     */
    TAction greedyExploration(RLAgent<TState, TAction>* agent) {
        //get either the best indice of a random one among the action space
        return agent->q->argmax(agent->current_state());
    }
};

#endif // GREEDYPOLICY_HPP
