//
// Created by xetql on 15.02.18.
//
#include "../../includes/librl/agents/RLAgentFactory.hpp"
#include "../../includes/librl/approximators/FunctionApproximator.hpp"
#include "../../includes/librl/approximators/MLPActionValueApproximator.hpp"
#include "../../includes/librl/policies/Policies.hpp"

#include <mlpack/methods/ann/init_rules/const_init.hpp>

#include <vector>

#include <armadillo>
/**
 * Multi armed bandit example
 */

std::vector<int> ACTION(arma::mat S) {
    std::vector<int> x = {0,1,2,3};
    return x;
}

double REWARD(arma::mat S, int A) {
    switch(A){
        case 1: return  1;
        case 2: return  1;
        case 3: return   80;
        default: return 1;
    }
}

arma::mat TRANSITION(arma::mat S, int A) {
    arma::mat r = arma::zeros(1);
    return r;
}

int main(int argc, char** argv){
    librl::environment::MDP<arma::mat, int> mdp(ACTION, REWARD, TRANSITION, arma::zeros(1));
    librl::policy::Layered<arma::mat, int> layeredPolicy;

    mlpack::ann::FFN<mlpack::ann::MeanSquaredError<>, mlpack::ann::ConstInitialization > model(mlpack::ann::MeanSquaredError<>(), mlpack::ann::ConstInitialization(0.0));
    mlpack::optimization::RMSProp opt(0.8, 32, 0.9, 1e-8, 10, -1);

    model.Add<mlpack::ann::Linear<> >(1, 6);
    model.Add<mlpack::ann::LeakyReLU<> >();
    model.Add<mlpack::ann::Linear<> >(6, 4);
    model.Add<mlpack::ann::LeakyReLU<> >();

    librl::approximator::MLPActionValueApproximator<
        mlpack::ann::FFN<mlpack::ann::MeanSquaredError<>, mlpack::ann::ConstInitialization  >
    > learner(&model, opt, opt.Alpha(), 4);

    layeredPolicy.add<librl::policy::EGreedy, 10>(1.0);
    layeredPolicy.add<librl::policy::Greedy>();

    auto player = librl::agent::RLAgentFactory<arma::mat, int>::get_instance("qlearning", 0.9, &layeredPolicy, &learner);
    std::vector<int> cnt(4);

    for(int i = 0; i < 20; ++i){
        auto s = player->choose_action(mdp.current_state, mdp.get_available_actions());
        auto reward = mdp.get_reward(s);
        std::cout << "Action : "<< s << " Reward : "<<reward << std::endl;

        auto transition = mdp.perform_state_transition(s);  
        player->learn(transition.first, s, transition.second, reward);
        cnt[s]++;
    }

    for(int i = 0; i < 4; i++){
        std::cout << "Arm " << i << " used " << cnt[i] << " times" << std::endl;
    }

    return 0;
}