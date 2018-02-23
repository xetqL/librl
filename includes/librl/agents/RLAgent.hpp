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
             librl::environment::MDP<TState, TAction>* mdp,
             librl::approximator::ActionValueApproximator<TState, TAction>* fa,
             double discount_factor) :
             q(fa),
             pi(pi),
             mdp(mdp),
             gamma(discount_factor) {}

    void set_behavioral_policy(librl::policy::Policy<TState, TAction>* pi) { this->pi = pi; }

    virtual TAction choose_action(const TState& state, const std::vector<TAction>& actions) = 0;
    virtual TAction choose_action(const std::vector<TAction>& actions) { return this->choose_action(this->mdp->current_state, actions); }
    virtual TAction choose_action() { return this->choose_action(this->mdp->get_available_actions()); }

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
     * @brief Setter for the learning parameters (alpha, beta, gamma etc.)
     */
    virtual void set_learning_parameters(std::vector<double> parameters) = 0;

    virtual void notify() const { pi->update(); }

    std::vector<TAction> get_available_actions() const { return this->mdp->get_available_actions(); }

    const librl::policy::Policy<TState, TAction>* get_policy() { return this->pi; };

protected:
    /**
    * Compute the reinforcement
    * @param action
    * @param reward
    * @param nextState
    * @return
    */
    virtual double get_reinforcement(TState prev_state, TAction action, TState next_state, double reward) const = 0;

    librl::approximator::ActionValueApproximator<TState, TAction>* q;
    librl::policy::Policy<TState, TAction>* pi;
    librl::environment::MDP<TState, TAction>* mdp;

    double gamma;
};
}}
#include "../policies/Policy.hpp"

#endif
