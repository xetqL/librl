//
// Created by xetql on 22.02.18.
//

#ifndef PROJECT_ROUNDROBIN_HPP
#define PROJECT_ROUNDROBIN_HPP

#include <map>

#include "Policy.hpp"
namespace librl{ namespace policy{

template<typename TState, typename TAction>
class RoundRobin : public Policy <TState, TAction>{
    //count how many actions have been tried in a given state
    std::unordered_map<TState, unsigned int> action_indices;
public:

    virtual std::unordered_map<TAction, double>
    get_probabilities(const librl::approximator::ActionValueApproximator<TState, TAction>* f,
                      const std::vector<TAction> &available_actions,
                      const TState &at_state) const {
        std::unordered_map<TAction, double> probabilities;
        for(auto const& a: available_actions) probabilities[a] = 0.0;
        TAction action = available_actions.at(action_indices.at(at_state) % available_actions.size());
        probabilities[action] = 1.0;
        return probabilities;
    }

    virtual TAction choose_action(const librl::approximator::ActionValueApproximator<TState, TAction>* f,
                                  const std::vector<TAction> &available_actions,
                                  const TState &at_state) {
        if(action_indices.find(at_state) == action_indices.end()) action_indices[at_state] = 0;
        TAction action = available_actions.at(action_indices.at(at_state) % available_actions.size());
        action_indices[at_state] = action_indices.at(at_state) + 1;
        return action;
    }

    virtual std::string getName() const {return "Round Robin";};

    virtual void reset() {action_indices.clear();};

    virtual void update() {};

    virtual std::string to_string() { return ""; };
};

}}
#endif //PROJECT_ROUNDROBIN_HPP
