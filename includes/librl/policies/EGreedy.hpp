#ifndef EGREEDYPOLICY_HPP
#define EGREEDYPOLICY_HPP
#include "Policy.hpp"
#include "Greedy.hpp"
#include "../utils/util.hpp"
#include <vector>
#include "../agents/RLAgent.hpp"
namespace librl { namespace policy {

template<typename TState, typename TAction>
class EGreedy : public Policy<TState, TAction> {
public:

    EGreedy(double E) {
        this->E = E;
    }

    virtual TAction choose_action(const librl::approximator::action_value::ActionValueApproximator<TState, TAction>* f,
                                  const std::vector<TAction> &available_actions,
                                  const TState &at_state) {
        return eGreedyExploration(f, available_actions, at_state);
    }

    virtual TAction predict_action(const librl::approximator::action_value::ActionValueApproximator<TState, TAction>* f,
                                  const std::vector<TAction> &available_actions,
                                  const TState &at_state) const {
        auto v = ((double) std::rand() / (RAND_MAX));
        if (v > E) {
            return g.predict_action(f, available_actions, at_state);
        } else {
            return *select_randomly(available_actions.begin(), available_actions.end());
        }
    }

    void reset() {}

    virtual std::unordered_map<TAction, double> get_probabilities(const librl::approximator::action_value::ActionValueApproximator<TState, TAction>* f,
                                                        const std::vector<TAction> &available_actions,
                                                        const TState &at_state) const {
        double max = f->max(at_state);
        int number_of_optimal_action = 0;

        for (auto const &action : available_actions)
            if (f->Q(at_state, action) == max) number_of_optimal_action++;

        std::unordered_map<TAction, double> probabilities;
        for (auto const &action : available_actions) {
            probabilities[action] = f->Q(at_state, action) == max ? (1 - E) / ((double) number_of_optimal_action) : E / ((double) available_actions.size() - number_of_optimal_action);
        }

        return probabilities;
    }

    TAction eGreedyExploration(const librl::approximator::action_value::ActionValueApproximator<TState, TAction>* f,
                               const std::vector<TAction> &available_actions,
                               const TState &at_state) {
        auto v = ((double) std::rand() / (RAND_MAX));
        if (v > E) {
            return g.choose_action(f, available_actions, at_state);
        } else {
            return *select_randomly(available_actions.begin(), available_actions.end());
        }
    }

protected:
    Greedy<TState, TAction> g;
    double E;
};
}}
#endif // EGREEDYPOLICY_HPP
