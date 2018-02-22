#ifndef RLALGORITHMS_HPP
#define RLALGORITHMS_HPP

#include "DoubleQLearningAgent.hpp"
#include "QLearningAgent.hpp"
#include "SarsaAgent.hpp"
#include "ExpectedSarsaAgent.hpp"
#include "QVLearningAgent.hpp"

#include <memory>
#include <vector>
#include <exception>
namespace librl { namespace agent {
    template<typename State, typename Action>
    class ReinforcementLearningAgentFactory
    {
    public:
        static std::unique_ptr<RLAgent<State, Action>>
        get_instance(std::string algorithm_name,
                     double discount_factor,
                     librl::policy::Policy<State, Action>* policy,
                     librl::environment::MDP<State, Action>* mdp,
                     librl::approximator::ActionValueApproximator<State,Action>* fa) {
            if (!algorithm_name.compare("qlearning")) {
                return std::move(std::make_unique<QLearningAgent<State, Action>>(policy, mdp, fa, discount_factor));
            } else if (!algorithm_name.compare("sarsa")) {
                return std::move(std::make_unique<SarsaAgent<State, Action>>(policy, mdp, fa, discount_factor));
            } else if (!algorithm_name.compare("expected-sarsa")) {
                return std::move(std::make_unique<ExpectedSarsaAgent<State, Action>>(policy, mdp, fa, discount_factor));
            } else {
                std::cerr << "This RL algorithm does not work with action value function approximator" << std::endl;
            }
            throw new std::invalid_argument("Not a valid algorithm/parameters combination.");
        }

        static std::unique_ptr<RLAgent<State, Action>>
        get_instance(std::string algorithm_name,
                     double discount_factor,
                     librl::policy::Policy<State, Action>* policy,
                     librl::environment::MDP<State, Action>* mdp,
                     librl::approximator::DoubleApproximator<State, Action>* fa){
            if (!algorithm_name.compare("double-qlearning")) {
                return std::move(std::make_unique<DoubleQLearningAgent<State, Action>>(policy, mdp, fa, discount_factor));
            } else {
                librl::approximator::ActionValueApproximator<State, Action>* generic_fa = fa;
                return ReinforcementLearningAgentFactory::get_instance(algorithm_name, discount_factor, policy, mdp, generic_fa);
            }
        }

        static std::unique_ptr<RLAgent<State, Action>>
        get_instance(std::string algorithm_name,
                     double discount_factor,
                     librl::policy::Policy<State, Action>* policy,
                     librl::environment::MDP<State, Action>* mdp,
                     librl::approximator::ActionValueApproximator<State,Action>* fa,
                     librl::approximator::StateValueApproximator<State>* sfa){
            if (!algorithm_name.compare("qvlearning")) {
                return std::move(std::make_unique<QVLearningAgent<State, Action>>(policy, mdp, fa, sfa, discount_factor));
            } else {
                std::cerr << "This RL algorithm does not work with state value function approximator" << std::endl;
            }
            throw new std::invalid_argument("Not a valid algorithm/parameters combination.");
        }
    };
    }}
#endif
