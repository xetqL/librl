/*
 * File:   gridworld.cpp
 * Author: xetql
 *
 * Created on February 27, 2017, 11:10 AM
 */
#include "../../include/librl/RLAlgorithms.hpp"

#include "gridworld.hpp"

using namespace std;

Maze T(Maze maze, Move m) {
    Position player_location = locate_player(maze);
    assert(is_valid(maze, m, player_location));
    maze[player_location.first][player_location.second] = EMPTY;
    player_location = apply_move(player_location, m);
    maze[player_location.first][player_location.second] = PLAYER;
    return maze;
}

std::vector<Maze> S() {
    Maze m = {
        {EMPTY, EMPTY, EMPTY, STOP},
        {EMPTY, EMPTY, EMPTY, EMPTY},
        {EMPTY, HOLE, EMPTY, EMPTY},
        {EMPTY, EMPTY, EMPTY, EMPTY}
    }, tmp = m;
    std::vector<Maze> states;
    for (size_t i = 0; i < m.size(); i++) {
        for (size_t j = 0; j < m[i].size(); j++) {
            if (m[i][j] == EMPTY) {
                tmp[i][j] = PLAYER;
                states.push_back(tmp);
                tmp = m;
            }
        }
    }
    return states;
}

double R(Maze maze, Move m) {
    Position player_location = locate_player(maze);
    assert(is_valid(maze, m, player_location));
    Position after_move = apply_move(player_location, m);
    if (maze[after_move.first][after_move.second] == STOP) {
        return 0.0;
    }
    if (maze[after_move.first][after_move.second] == HOLE) {
        return -100.0;
    }
    return -1.0;
}

std::vector<Move> A(Maze maze) {
    Position player_location = locate_player(maze);
    std::vector<Move> restricted_moves;
    if (is_valid(maze, LEFT, player_location)) restricted_moves.insert(restricted_moves.begin(), LEFT);
    if (is_valid(maze, RIGHT, player_location))restricted_moves.insert(restricted_moves.begin(), RIGHT);
    if (is_valid(maze, UP, player_location)) restricted_moves.insert(restricted_moves.begin(), UP);
    if (is_valid(maze, DOWN, player_location)) restricted_moves.insert(restricted_moves.begin(), DOWN);

    return restricted_moves;
}

int main(int argc, char** argv) {

    Maze entry_state = {
        {EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY},
        {EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY},
        {PLAYER, HOLE, HOLE, HOLE, HOLE, HOLE, HOLE, STOP},
    };


    MDP<Maze, Move> maze_mdp = MDP<Maze, Move >(S, A, R, T, entry_state);
    // e-Greedy exploration policy
    EGreedyPolicy<Maze, Move> policy_egreedy = EGreedyPolicy<Maze, Move > (0.1);
    //  Greedy exploration policy
    GreedyPolicy<Maze, Move> policy_greedy = GreedyPolicy<Maze, Move> ();
    // Action value approximator that uses an array
    ArrayActionValueApproximator<Maze, Move> afa = ArrayActionValueApproximator<Maze, Move > (0.3, A);
    // Action value approximator that uses an array
    DoubleApproximator<Maze, Move> dblfa = DoubleApproximator<Maze, Move > (0.3, A);
    // Action value approximator that uses an array
    ArrayStateValueApproximator <Maze> sfa =  ArrayStateValueApproximator<Maze> (0.3);

    // Sarsa RL agent
    std::shared_ptr<RLAgent<Maze, Move >> player(
        ReinforcementLearningAgentFactory<Maze, Move>::get_instance("qlearning", 0.9, &policy_egreedy, &maze_mdp, &afa, &sfa)
    );
    
    const int START_SHOW = 750;
    const int RESTART_AFTER = 100000;
    int LEARNING_ITERATIONS = 1100;
    double r = 0.0;
    for (size_t i = 0; i < LEARNING_ITERATIONS; ++i) {

        if (i >= START_SHOW) {
            print_maze(maze_mdp.current_state);
            player->set_behavioral_policy(&policy_greedy);
        }

        if (i > START_SHOW) std::this_thread::sleep_for(std::chrono::milliseconds(250));

        int trial = 0;

        while (trial < RESTART_AFTER) {
            if (i >= START_SHOW) clear();
            auto sel_a = player->choose_action();
            double reward = maze_mdp.get_reward(sel_a);
            pair<Maze, Maze> prev_next_states = maze_mdp.perform_state_transition(sel_a);
            if (i >= START_SHOW) print_maze(prev_next_states.second);
            player->learn(prev_next_states.first, sel_a, prev_next_states.second, reward);
            r += reward;
            if (reward != -1.0) break;
            if (i >= START_SHOW) std::this_thread::sleep_for(std::chrono::milliseconds(250));
            trial++;
            //cout << "NN MSE: " << fa->net.get_MSE() << endl;
        }

        cout << "EPOCH [" << i << "]" << " average reward : " << (r / i) << endl;
        maze_mdp.reset();
    }
    return 0;
}
