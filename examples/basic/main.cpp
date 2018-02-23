//
// Created by xetql on 15.02.18.
//
#include <librl/agents/RLAgentFactory.hpp>
#include <librl/approximators/FunctionApproximator.hpp>
#include <librl/approximators/MLPActionValueApproximator.hpp>
#include <mlpack/methods/ann/init_rules/const_init.hpp>
#include <librl/policies/Policies.hpp>

#include <vector>

/**
 * Multi armed bandit example
 */

std::vector<int> ACTION(arma::mat S) {
    std::vector<int> x = {1, 2, 3, 4};
    return x;
}

std::vector<arma::mat> STATE() {
    arma::mat r = arma::randu(1);
    return {r};
}

double REWARD(arma::mat S, int A) {
    switch(A){
        case 1: return -10;
        case 2: return -10;
        case 3: return 80;
        default: return -10;
    }
}

arma::mat TRANSITION(arma::mat S, int A) {
    arma::mat r = arma::zeros(1);
    return r;
}

int main(int argc, char** argv){
    //librl::approximator::ArrayActionValueApproximator<int, int> afa(1.0);

    librl::environment::MDP<arma::mat, int> mdp(ACTION, REWARD, TRANSITION, arma::zeros(1));
    librl::policy::Layered<arma::mat, int> layeredPolicy;

    mlpack::ann::FFN<mlpack::ann::MeanSquaredError<>, mlpack::ann::ConstInitialization > model(mlpack::ann::MeanSquaredError<>(), mlpack::ann::ConstInitialization(0.0));
    mlpack::optimization::RMSProp opt(0.01, 32, 0.88, 1e-8, 10, -1);

    model.Add<mlpack::ann::Linear<> >(1, 6);
    model.Add<mlpack::ann::LeakyReLU<> >();
    model.Add<mlpack::ann::Linear<> >(6, 4);
    model.Add<mlpack::ann::LeakyReLU<> >();

    librl::approximator::MLPActionValueApproximator<mlpack::ann::FFN<mlpack::ann::MeanSquaredError<>, mlpack::ann::ConstInitialization  >> learner(&model, opt, opt.Alpha(), 4);

    layeredPolicy.add<librl::policy::EGreedy, 5>(1.0);
    layeredPolicy.add<librl::policy::Greedy>();

    auto player = librl::agent::RLAgentFactory<arma::mat, int>::get_instance("qlearning", 0.9, &layeredPolicy, &learner);
    std::vector<int> cnt(4);

    for(int i = 0; i < 15; ++i){
        auto s = player->choose_action(mdp.current_state, mdp.get_available_actions());
        std::cout << "Action : "<< s << std::endl;
        auto reward = mdp.get_reward(s);
        auto transition = mdp.perform_state_transition(s);
        player->learn(transition.first, s, transition.second, reward);
        cnt[s-1]++;
    }

    for(int i = 0; i < 4; i++){
        std::cout << "Arm " << i << " used " << cnt[i] << " times" << std::endl;
    }

    return 0;
}