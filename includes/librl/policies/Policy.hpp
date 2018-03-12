#ifndef POLICY_HPP
#define POLICY_HPP
#include <unordered_map>
#include <string>
#include <limits>
#include <queue>
#include "../approximators/FunctionApproximator.hpp"

namespace librl { namespace policy {

/** Action Selection Policy **/
template<typename TState, typename TAction>
class Policy {
public:
    virtual std::unordered_map<TAction, double>
    get_probabilities(const librl::approximator::action_value::ActionValueApproximator<TState, TAction>* f,
                      const std::vector<TAction> &available_actions,
                      const TState &at_state) const = 0;
    /**
     * Choose the next action given the env. state and the learned function. It does change the internal
     * state of the policy making it goes to its next step.
     * @param f the learned action value function
     * @param available_actions the available actions
     * @param at_state the state of the env.
     * @return The selected action
     */
    virtual TAction choose_action(const librl::approximator::action_value::ActionValueApproximator<TState, TAction>* f,
                                  const std::vector<TAction> &available_actions,
                                  const TState &at_state) = 0;
    /**
     * Predict which action will be selected by the policy without changing any internal state.
     * @param f the learned action value function
     * @param available_actions the available actions
     * @param at_state the state of the env.
     * @return The selected action
     */
    virtual TAction predict_action(const librl::approximator::action_value::ActionValueApproximator<TState, TAction>* f,
                                   const std::vector<TAction> &available_actions,
                                   const TState &at_state) const = 0;

    virtual void reset() = 0;

    virtual void update() {};

    virtual std::string to_string() { return ""; };
};
}}
#endif // POLICY_HPP
