//
// Created by xetql on 15.02.18.
//
#include <librl/agents/RLAgentFactory.hpp>
#include <librl/approximators/FunctionApproximator.hpp>
#include <librl/approximators/MLPActionValueApproximator.hpp>
#include <librl/policies/Policies.hpp>

#include <vector>

/**
 * Multi armed bandit example
 */

std::vector<int> ACTION(int S) {
    std::vector<int> x = {1, 2, 3, 4};
    return x;
}

std::vector<int> STATE() {
    std::vector<int> x = {0, 1};
    return x;
}

double REWARD(int S, int A) {
    switch(A){
        case 1: return 1;
        case 2: return 1;
        case 3: return 80;
        default: return -1;
    }
}

int TRANSITION(int S, int A) {
    return 0;
}

int main(int argc, char** argv){
    librl::approximator::ArrayActionValueApproximator<int, int> afa(1.0);

    librl::environment::MDP<int,int> mdp(ACTION, REWARD, TRANSITION, 0);
    librl::policy::Layered<int, int> layeredPolicy;

    mlpack::ann::FFN<mlpack::ann::MeanSquaredError<> > model;
    mlpack::optimization::RMSProp opt(0.01, 32, 0.88, 1e-8, 10, -1);

    model.Add<mlpack::ann::Linear<> >(1, 6);
    model.Add<mlpack::ann::LeakyReLU<> >();
    model.Add<mlpack::ann::Linear<> >(6, 4);
    model.Add<mlpack::ann::LeakyReLU<> >();

    librl::approximator::MLPActionValueApproximator<mlpack::ann::FFN<mlpack::ann::MeanSquaredError<> >> learner(&model, opt, opt.Alpha(), 4);

    layeredPolicy.add<librl::policy::EGreedy, 10>(1.0);
    layeredPolicy.add<librl::policy::Greedy>();

    auto player = librl::agent::RLAgentFactory<int, int>::get_instance("qlearning", 0.9, &layeredPolicy, &afa);
    std::vector<int> cnt(4);

    for(int i = 0; i < 15; ++i){
        auto s = player->choose_action(mdp.current_state, mdp.get_available_actions());
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