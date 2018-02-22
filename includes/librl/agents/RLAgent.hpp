/**
 * TODO: Agent should not be aware of the state-transition function
 * */

#ifndef RLAGENT
#define RLAGENT
#include "../approximators/FunctionApproximator.hpp"
#include "AgentStatistics.hpp"
#include "../utils/array.hpp"
#include "../utils/util.hpp"
#include "../env/MDP.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <memory>
namespace librl { namespace agent {
    template<typename TState, typename TAction> class RLAgent;
}}

#include "../policies/Policy.hpp"

#define LOGGING_STATE_ENABLED 0
#define GET_LOGGING_STATE_ENABLED 0
#define START_ACTION 0
namespace librl { namespace policy {
        template<typename TState, typename TAction>
        class Policy; //bypass cross import problem
    }}
namespace librl { namespace agent {
template<typename TState, typename TAction>
class RLAgent {
public:

    /**
     * @brief Reinforcement Learning agent abstract class, constructor with default alpha gamma
     * @param nbStates Number of state, one state is added as a starting state.
     * @param explTechnique Specifiy the function that selects the next Action following a given policy
     */
     RLAgent(
             librl::policy::Policy<TState, TAction>* pi,
             librl::environment::MDP<TState, TAction>* mdp,
             librl::approximator::ActionValueApproximator<TState, TAction>* fa,
             double discount_factor)
     : q(fa), pi(pi), mdp(mdp), gamma(discount_factor) {
     }

    void set_behavioral_policy(librl::policy::Policy<TState, TAction>* pi) {
        this->pi = pi;
    }

    void print_report() const {
        std::cout << "***************************************" << std::endl;
        std::cout << " RL Agent stats report" << std::endl;
        std::cout << "***************************************" << std::endl;
        std::cout << "Agent uuid    : " << this->uuid << std::endl;
        std::cout << "Reward mean     : " << this->stats->meanReward << std::endl;
        std::cout << "Reward variance : " << this->stats->varianceReward << std::endl;
        std::cout << "Reward standard deviation : " << std::sqrt(this->stats->varianceReward) << std::endl;
        std::cout << "***************************************" << std::endl;
        print2DArray(this->stats->numberOfTimeAction);
    }

    /**
     * @brief Get the agent's algorithm name
     */
    virtual std::string getName() const = 0;

    /**
     * @brief Select the action to perform given the current policy pi and the state of the environment
     */
    virtual TAction choose_action() const = 0;

    /**
     * @brief Reward the agent with the signal for performing action A in state S
     */
    virtual void learn(TState prev_state, TAction action, TState next_state, double reward) = 0;

    /**
     * @brief Setter for the learning parameters (alpha, beta, gamma etc.)
     */
    virtual void set_learning_parameters(std::vector<double> parameters) = 0;

    virtual void notify() const {
        pi->update();
    }

    TState current_state() const {
        return this->mdp->current_state;
    }

    std::vector<TAction> get_available_actions() const {
        return this->mdp->get_available_actions();
    }
    std::shared_ptr<librl::stats::AgentStatistics> stats;
    librl::approximator::ActionValueApproximator<TState, TAction>* q;
protected:
    /**
     * Compute the reinforcement
     * @param action
     * @param reward
     * @param nextState
     * @return
     */
    virtual double get_reinforcement(TState prev_state, TAction action, TState next_state, double reward) const = 0;

    std::string uuid;
    librl::policy::Policy<TState, TAction>* pi;
    librl::environment::MDP<TState, TAction>* mdp;
    double gamma;

};
}}
#include "../policies/Policy.hpp"

#endif
