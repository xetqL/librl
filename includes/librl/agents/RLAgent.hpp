/**
 * TODO: Agent should not be aware of the state-transition function
 * */

#ifndef RLAGENT
#define RLAGENT
#include "../approximators/FunctionApproximator.hpp"
#include "../utils/array.hpp"
#include "../utils/util.hpp"
#include "../env/MDP.hpp"
#include "../policies/Policy.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <memory>

namespace librl { namespace agent {
template<typename TState, typename TAction>
class RLAgent {
public:

    /**
     * @brief Reinforcement Learning agent abstract class, constructor with default alpha gamma
     * @param nbStates Number of state, one state is added as a starting state.
     * @param explTechnique Specifiy the function that selects the next Action following a given policy
     */
     RLAgent(librl::policy::Policy<TState, TAction>* pi,
             librl::approximator::ActionValueApproximator<TState, TAction>* fa,
             double discount_factor) :
             q(fa),
             pi(pi),
             gamma(discount_factor) {}

    /**
     * Set the behavioral policy followed by the agent.
     * @param pi A pointer to the behavioral policy
     */
    void set_behavioral_policy(librl::policy::Policy<TState, TAction>* pi) { this->pi = pi; }

    /**
     * Choose an action in a given state from the possible actions given in parameters.
     * This function should be used before this->learn.
     * @param state The current state of the env.
     * @param actions The available actions
     * @return The selected action
     */
    virtual TAction choose_action(const TState& state, const std::vector<TAction>& actions) = 0;

    /**
     * Reward the agent for the action A taken in state S leading to state S'. Should only be used
     * after a call to choose_action(...). Only experienced user may call learn() without calling
     * choose_action.
     * @param prev_state The previous state S
     * @param action The action taken in state S
     * @param next_state The result of the state transition S'
     * @param reward The scalar reward
     */
    virtual void learn(TState prev_state, TAction action, TState next_state, double reward) = 0;

    /**
     * Set the learning parameters for the agent.
     * @param parameters A vector containing the parameters
     */
    virtual void set_learning_parameters(std::vector<double> parameters) = 0;

    /**
     * Notify the object managed by the agent.
     */
    virtual void notify() const { pi->update(); }

    /**
     * Get the behavioral policy
     * @return The behavioral policy
     */
    const librl::policy::Policy<TState, TAction>* get_policy() { return this->pi; };

protected:
    /**
     * Compute the reinforcement used to update the function approximator.
     * @param prev_state The state where the action has been taken.
     * @param action The action selected by the agent.
     * @param next_state The destination state.
     * @param reward The reward from taking action in state prev_state
     * @return The reinforcement
     */
    virtual double get_reinforcement(TState prev_state, TAction action, TState next_state, double reward) const = 0;

    librl::approximator::ActionValueApproximator<TState, TAction>* q;
    librl::policy::Policy<TState, TAction>* pi;
    double gamma;
};
}}
#include "../policies/Policy.hpp"

#endif
