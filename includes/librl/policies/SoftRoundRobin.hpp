//
// Created by xetql on 22.02.18.
//

#ifndef PROJECT_SOFTROUNDROBIN_HPP
#define PROJECT_SOFTROUNDROBIN_HPP
#include "Policy.hpp"
namespace librl{ namespace policy{

template<typename TState, typename TAction>
class SoftRoundRobin : public Policy <TState, TAction>{
    unsigned int action_index = 0;
public:
    virtual std::unordered_map<TAction, double>
    get_probabilities(const librl::approximator::ActionValueApproximator<TState, TAction>* f,
                      const std::vector<TAction> &available_actions,
                      const TState &at_state) const {
        std::unordered_map<TAction, double> probabilities;
        for(auto const& a: available_actions) probabilities[a] = 0.0;
        TAction action = available_actions.at(action_index % available_actions.size());
        probabilities[action] = 1.0;
        return probabilities;
    }

    virtual TAction choose_action(const librl::approximator::ActionValueApproximator<TState, TAction>* f,
                                  const std::vector<TAction> &available_actions,
                                  const TState &at_state) {
        TAction action = available_actions.at(action_index % available_actions.size());
        action_index++;
        return action;
    }

    virtual std::string getName() const {return "Soft Round Robin";};

    virtual void reset() {action_index = 0;};

    virtual std::string to_string() { return ""; };
};

}}
#endif //PROJECT_SOFTROUNDROBIN_HPP
