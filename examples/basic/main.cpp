//
// Created by xetql on 15.02.18.
//
#include <librl/agents/RLAlgorithms.hpp>
#include <librl/approximators/FunctionApproximator.hpp>
#include <librl/policies/Policies.hpp>

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
        case 1: return -1;
        case 2: return -1;
        case 3: return 80;
        case 4: return -1;
    }
    return 1;
}

int TRANSITION(int S, int A) {
    if(S==0 && A==4) return 1;
    if(S==1 && A==4) return 0;
    return S;
}

int main(int argc, char** argv){
    librl::approximator::ArrayActionValueApproximator<int, int> afa(0.8);
    librl::environment::MDP<int,int> mdp(STATE, ACTION, REWARD, TRANSITION, 0);
    librl::policy::EGreedyPolicy<int, int> egreedy(0.1);
    librl::policy::GreedyPolicy<int, int, librl::policy::FullRoundRobin, 1> greedy;
    auto player = librl::agent::ReinforcementLearningAgentFactory<int, int>::get_instance("qlearning", 0.9, &greedy, &mdp, &afa);
    std::vector<int> cnt(4);

    for(int i = 0; i < 80; ++i){
        if(i == 450) player->set_behavioral_policy(&greedy);
        auto s = player->choose_action();
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