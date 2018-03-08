#ifndef RLALGORITHMS_HPP
#define RLALGORITHMS_HPP

#include "DoubleQLearning.hpp"
#include "QLearning.hpp"
#include "Sarsa.hpp"
#include "ExpectedSarsa.hpp"
#include "QVLearning.hpp"

#include <memory>
#include <vector>
#include <exception>
namespace librl { namespace agent {
#if __cplusplus>201103L
template<typename State, typename Action>
class RLAgentFactory
{
public:
    static std::unique_ptr<RLAgent<State, Action>>
    get_instance(std::string algorithm_name,
                 double discount_factor,
                 librl::policy::Policy<State, Action>* policy,
                 librl::approximator::ActionValueApproximator<State, Action>* fa) {
        if (!algorithm_name.compare("qlearning")) {
            return std::move(std::make_unique<QLearning<State, Action>>(policy, fa, discount_factor));
        } else if (!algorithm_name.compare("sarsa")) {
            return std::move(std::make_unique<Sarsa<State, Action>>(policy, fa, discount_factor));
        } else if (!algorithm_name.compare("expected-sarsa")) {
            return std::move(std::make_unique<ExpectedSarsa<State, Action>>(policy, fa, discount_factor));
        } else {
            std::cerr << "This RL algorithm does not work with action value function approximator" << std::endl;
        }
        throw new std::invalid_argument("Not a valid algorithm/parameters combination.");
    }

    static std::unique_ptr<RLAgent<State, Action>>
    get_instance(std::string algorithm_name,
                 double discount_factor,
                 librl::policy::Policy<State, Action>* policy,
                 librl::approximator::DoubleApproximator<State, Action>* fa){
        if (!algorithm_name.compare("double-qlearning")) {
            return std::move(std::make_unique<DoubleQLearning<State, Action>>(policy, fa, discount_factor));
        } else {
            librl::approximator::ActionValueApproximator<State, Action>* generic_fa = fa;
            return RLAgentFactory::get_instance(algorithm_name, discount_factor, policy, generic_fa);
        }
    }

    static std::unique_ptr<RLAgent<State, Action>>
    get_instance(std::string algorithm_name,
                 double discount_factor,
                 librl::policy::Policy<State, Action>* policy,
                 librl::approximator::ActionValueApproximator<State,Action>* fa,
                 librl::approximator::StateValueApproximator<State>* sfa){
        if (!algorithm_name.compare("qvlearning")) {
            return std::move(std::make_unique<QVLearning<State, Action>>(policy, fa, sfa, discount_factor));
        } else {
            std::cerr << "This RL algorithm does not work with state value function approximator" << std::endl;
        }
        throw new std::invalid_argument("Not a valid algorithm/parameters combination.");
    }
};
#else
template<typename State, typename Action>
class RLAgentFactory
{
public:
    static std::shared_ptr<RLAgent<State, Action>>
    get_instance(std::string algorithm_name,
                 double discount_factor,
                 librl::policy::Policy<State, Action>* policy,
                 librl::approximator::ActionValueApproximator<State, Action>* fa) {
        if (!algorithm_name.compare("qlearning")) {
            return std::make_shared<QLearning<State, Action>>(policy, fa, discount_factor);
        } else if (!algorithm_name.compare("sarsa")) {
            return std::make_shared<Sarsa<State, Action>>(policy, fa, discount_factor);
        } else if (!algorithm_name.compare("expected-sarsa")) {
            return std::make_shared<ExpectedSarsa<State, Action>>(policy, fa, discount_factor);
        } else {
            std::cerr << "This RL algorithm does not work with action value function approximator" << std::endl;
        }
        throw new std::invalid_argument("Not a valid algorithm/parameters combination.");
    }

    static std::shared_ptr<RLAgent<State, Action>>
    get_instance(std::string algorithm_name,
                 double discount_factor,
                 librl::policy::Policy<State, Action>* policy,
                 librl::approximator::DoubleApproximator<State, Action>* fa){
        if (!algorithm_name.compare("double-qlearning")) {
            return std::make_shared<DoubleQLearning<State, Action>>(policy, fa, discount_factor);
        } else {
            librl::approximator::ActionValueApproximator<State, Action>* generic_fa = fa;
            return RLAgentFactory::get_instance(algorithm_name, discount_factor, policy, generic_fa);
        }
    }

    static std::shared_ptr<RLAgent<State, Action>>
    get_instance(std::string algorithm_name,
                 double discount_factor,
                 librl::policy::Policy<State, Action>* policy,
                 librl::approximator::ActionValueApproximator<State,Action>* fa,
                 librl::approximator::StateValueApproximator<State>* sfa){
        if (!algorithm_name.compare("qvlearning")) {
            return std::make_shared<QVLearning<State, Action>>(policy, fa, sfa, discount_factor);
        } else {
            std::cerr << "This RL algorithm does not work with state value function approximator" << std::endl;
        }
        throw new std::invalid_argument("Not a valid algorithm/parameters combination.");
    }
};
#endif
}}
#endif
