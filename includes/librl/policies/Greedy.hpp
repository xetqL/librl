#ifndef GREEDYPOLICY_HPP
#define GREEDYPOLICY_HPP
#include "Policy.hpp"
#include "../utils/util.hpp"
#include <algorithm>

namespace librl { namespace policy {

template<typename TState, typename TAction>
class Greedy : public Policy<TState, TAction> {
public:
    virtual TAction choose_action(const librl::approximator::ActionValueApproximator<TState, TAction>* f,
                                  const std::vector<TAction> &available_actions,
                                  const TState &at_state) {
        return greedyExploration(f, available_actions, at_state);
    }

    virtual TAction predict_action(const librl::approximator::ActionValueApproximator<TState, TAction>* f,
                                  const std::vector<TAction> &available_actions,
                                  const TState &at_state) const {
        return greedyExploration(f, available_actions, at_state);
    }

    virtual std::unordered_map<TAction, double> get_probabilities(const librl::approximator::ActionValueApproximator<TState, TAction>* f,
                                                        const std::vector<TAction> &available_actions,
                                                        const TState &at_state) const {
        double max = f->max(at_state);
        std::unordered_map<TAction, double> probabilities;
        int number_of_optimal_action = 0;
        for (auto const &action : available_actions)
            if (f->Q(at_state, action) == max) number_of_optimal_action++;
        for (auto const &action : available_actions)
            probabilities[action] = f->Q(at_state, action) == max ? 1.0 / number_of_optimal_action : 0.0;
        return probabilities;
    }

    void reset() {}

    TAction greedyExploration(const librl::approximator::ActionValueApproximator<TState, TAction>* f,
                              const std::vector<TAction> &available_actions,
                              const TState &at_state) const {
        //get either the best indice of a random one among the action space
        auto action = f->argmax(at_state, available_actions);
        return action;
    }
};
}}
#endif // GREEDYPOLICY_HPP
