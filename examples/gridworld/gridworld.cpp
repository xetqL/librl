/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   gridworld.cpp
 * Author: xetql
 *
 * Created on February 27, 2017, 11:10 AM
 */


#include "librl/RLAlgorithms.hpp"
#include "librl/Policies.hpp"
#include "librl/util.hpp"
#include "librl/MDP.hpp"
#include "librl/FunctionApproximator.hpp"
#include "gridworld.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <thread>
#include <vector>
using namespace std;

/*
 * 
 */
int main(int argc, char** argv) {

    Maze entry_state = {
        {EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY},
        {EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY},
        {EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY},
        {EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY},
        {EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY},
        {PLAYER, HOLE, HOLE, HOLE, HOLE, HOLE, HOLE, STOP},
    };


    std::shared_ptr<MDP<Maze, Move>> maze_mdp = make_shared<MDP<Maze, Move >> (S, A, R, T, entry_state);
    std::shared_ptr<Policy<Maze, Move>> policy_egreedy = dynamic_pointer_cast<Policy<Maze, Move >> (std::make_shared<EGreedyPolicy<Maze, Move >> (0.2));
    std::shared_ptr<Policy<Maze, Move>> policy_greedy = dynamic_pointer_cast<Policy<Maze, Move >> (std::make_shared<GreedyPolicy<Maze, Move >> ());

    std::shared_ptr<ArrayActionValueApproximator<Maze, Move>> fa = make_shared<ArrayActionValueApproximator<Maze, Move >> (0.3, A);

    std::shared_ptr<MDP<Maze, Move>> tictactoe_mdp = make_shared<MDP<Maze, Move >> (S, A, R, T, entry_state);

    std::shared_ptr<SarsaAgent<Maze, Move >> player(
            new SarsaAgent<Maze, Move>({0.3, 0.9}, policy_egreedy, maze_mdp, fa)
            );
    const int START_SHOW = 9800;
    const int RESTART_AFTER = 1000;
    const int TEST_BATCH = 10;
    const int LEARNING_ITERATIONS = 10000;
    double r = 0.0;
    for (size_t i = 0; i < LEARNING_ITERATIONS; ++i) {

        if (i >= START_SHOW) {
            print_maze(maze_mdp->current_state);
            player->set_behavioral_policy(policy_greedy);
        }

        if (i > START_SHOW) std::this_thread::sleep_for(std::chrono::milliseconds(250));

        int trial = 0;

        while (trial < RESTART_AFTER) {
            if (i >= START_SHOW) clear();
            auto sel_a = player->choose_action();
            double reward = maze_mdp->get_reward(sel_a);
            pair<Maze, Maze> prev_next_states = maze_mdp->perform_state_transition(sel_a);
            if (i >= START_SHOW) print_maze(prev_next_states.second);
            player->learn(prev_next_states.first, sel_a, prev_next_states.second, reward);
            r += reward;
            if (reward != -1.0) break;
            if (i >= START_SHOW) std::this_thread::sleep_for(std::chrono::milliseconds(250));
            trial++;
            //cout << "NN MSE: " << fa->net.get_MSE() << endl;
        }

        cout << "EPOCH [" << i << "]" << " average reward : " << (r / i) << endl;
        //std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        maze_mdp->reset();
    }
    return 0;
}

