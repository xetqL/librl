//
// Created by xetql on 15.02.18.
//
#include "../../includes/librl/agents/RLAgentFactory.hpp"
#include "../../includes/librl/approximators/MLP.hpp"
#include "../../includes/librl/policies/Policies.hpp"
#include "../../includes/librl/approximators/FittedMLP.hpp"

#include <mlpack/methods/ann/init_rules/const_init.hpp>

#include <vector>

std::vector<int> ACTION(arma::mat S) {
    std::vector<int> x = {10, 20, 30, 40};
    return x;
}

double REWARD(arma::mat S, int A) {
    switch(A){
        case 40: return  80;
        default: return 1;
    }
}

arma::mat TRANSITION(arma::mat S, int A) {
    arma::mat r = arma::zeros(2);
    return r;
}

int main(int argc, char** argv){

    const int STATE_SIZE = 2;
    arma::mat starting_point = arma::zeros(STATE_SIZE);
    librl::environment::MDP<arma::mat, int> mdp(ACTION, REWARD, TRANSITION, starting_point);
    librl::policy::Layered<arma::mat, int> layeredPolicy;

    mlpack::optimization::RMSProp opt2(0.8, 32, 0.9, 1e-8, 10, -1);
    librl::approximator::action_value::FittedMLP<
            mlpack::optimization::RMSProp,
            mlpack::ann::MeanSquaredError<>> learner({10, 20, 30, 40}, opt2);

    learner.add_layer<mlpack::ann::Linear<> >(starting_point.n_rows+1, 6);
    learner.add_layer<mlpack::ann::LeakyReLU<> >();
    learner.add_layer<mlpack::ann::Linear<> >(6, 1);
    learner.add_layer<mlpack::ann::LeakyReLU<> >();

    layeredPolicy.add<librl::policy::EGreedy, 10>(1.0);
    layeredPolicy.add<librl::policy::Greedy >();

    auto player = librl::agent::RLAgentFactory<arma::mat, int>::get_instance("qlearning", 0.9, &layeredPolicy, &learner);
    std::unordered_map<int, int> cnt;

    for(int i = 0; i < 20; ++i){
        auto s = player->choose_action(mdp.current_state, mdp.get_available_actions());
        auto reward = mdp.get_reward(s);
        std::cout << "Action : "<< s << " Reward : "<<reward << std::endl;
        auto transition = mdp.perform_state_transition(s);  
        player->learn(transition.first, s, transition.second, reward);
        if (cnt.find(s) == cnt.end()) cnt[s] = 0;
        cnt[s]++;
    }

    for(auto i : mdp.get_available_actions()){
        std::cout << "Arm " << i << " used " << cnt[i] << " times" << std::endl;
    }

    return 0;
}