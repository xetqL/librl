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

template<typename State, typename Action>
class ReinforcementLearningAgentFactory
{
public:
    static RLAgent<State, Action>*
    get_instance(std::string algorithm_name,
        double discount_factor,
        Policy<State, Action>* policy,
        MDP<State, Action>* mdp,
        ActionValueApproximator<State,Action>* fa){
        if (!algorithm_name.compare("qlearning")) {
            return new QLearningAgent<State, Action>(policy, mdp, fa, discount_factor);
        } else if (!algorithm_name.compare("sarsa")) {
            return new SarsaAgent<State, Action>(policy, mdp, fa, discount_factor);
        } else if (!algorithm_name.compare("expected-sarsa")) {
            return new ExpectedSarsaAgent<State, Action>(policy, mdp, fa, discount_factor);
        } else {
            std::cerr << "This RL algorithm does not work with action value function approximator" << std::endl;
        }
        throw new std::invalid_argument("Not a valid algorithm/parameters combination.");
    }

    static RLAgent<State, Action>*
    get_instance(std::string algorithm_name,
        double discount_factor,
        Policy<State, Action>* policy,
        MDP<State, Action>* mdp,
        DoubleApproximator<State, Action>* fa){
        if (!algorithm_name.compare("double-qlearning")) {
            return new DoubleQLearningAgent<State, Action>(policy, mdp, fa, discount_factor);
        } else {
            ActionValueApproximator<State, Action>* generic_fa = fa;
            return ReinforcementLearningAgentFactory::get_instance(algorithm_name, discount_factor, policy, mdp, generic_fa);
        }
        throw new std::invalid_argument("Not a valid algorithm/parameters combination.");
    }

    static RLAgent<State, Action>*
    get_instance(std::string algorithm_name,
        double discount_factor,
        Policy<State, Action>* policy,
        MDP<State, Action>* mdp,
        ActionValueApproximator<State,Action>* fa,
        StateValueApproximator<State>* sfa){
        if (!algorithm_name.compare("qvlearning")) {
            std::cout << "return qvlearning" << std::endl;
            return new QVLearningAgent<State, Action>(policy, mdp, fa, sfa, discount_factor);
        } else {
            std::cerr << "This RL algorithm does not work with state value function approximator" << std::endl;
        }
        throw new std::invalid_argument("Not a valid algorithm/parameters combination.");
    }
};

#endif
