/*
 * File:   tictactoe.cpp
 * Author: xetql
 *
 * Created on February 27, 2017, 11:10 AM
 */

#include "tictactoe.hpp"

using namespace std;

int main(int argc, char** argv) {

    tictactoe entry_state = {
        {EMPTY, EMPTY, EMPTY},
        {EMPTY, EMPTY, EMPTY},
        {EMPTY, EMPTY, EMPTY}
    };

    EGreedyPolicy<tictactoe, Move> policy_egreedy = EGreedyPolicy<tictactoe, Move> (0.2);
    ArrayActionValueApproximator<tictactoe, Move> afa = ArrayActionValueApproximator<tictactoe, Move > (0.3, A);
    MDP<tictactoe, Move> mdp = MDP<tictactoe, Move > (S, A, R, T, entry_state);

    std::shared_ptr<RLAgent<tictactoe, Move >> player(
        ReinforcementLearningAgentFactory<tictactoe, Move>::get_instance("qlearning", 0.9, &policy_egreedy, &mdp, &afa)
    );

    unsigned int LEARNING_GAMES = atoi(argv[0]), START_SHOWING = LEARNING_GAMES - 10;
    bool show = false, finished, human_player = false;

    double tot_reward_p = 0, tot_reward_ai = 0;

    int turns_ai = 0, turns_p = 0;
    int move_index = -1;
    tictactoe old_prev, old_next;
    Move old_action;

    for (size_t game = 0; game < 1000000; ++game) {
        if(!(game%5000)) cout << "GAME [" << game << "]" << " average reward AI: " << (tot_reward_ai / (double) game) << " average reward p: " << (tot_reward_p / (double) game) << endl;
        if (game == START_SHOWING) {
            show = true;
            human_player = true;
        }

        finished = false;
        while (!finished && !isfinal(mdp.current_state)) {

            auto action = AIturn ? player->choose_action() : player->choose_action();

            if (AIturn && human_player) {
                std::cout << "=====================" << std::endl;
                print_tictactoe(mdp.current_state);
                std::cout << "Your move : ";
                std::cin >> move_index;
                action = make_pair(move_index / 3, move_index % 3);
            }

            auto state = mdp.perform_state_transition(action);
            piece winner = iswin(state.second);

            switch (winner) {
                case PLAYER: // O win
                case AI: // X win
                    player->learn( (old_prev), (old_action), state.second, -1);
                    tot_reward_p += AIturn ? -1 : 1;
                    player->learn( (state.first), (action), state.second, 1);
                    tot_reward_ai += AIturn ? 1 : -1;
                    if (show) {
                        std::cout << (AIturn ? " X win " : "O win") << std::endl;
                        print_tictactoe(state.second);
                    }
                    finished = true;
                    break;
                case TIE:
                    player->learn((old_prev), (old_action), state.second, 0.0);
                    player->learn((state.first), (action), state.second, 0.0);
                    tot_reward_ai += 0.0;
                    tot_reward_p  += 0.0;
                    if (show) {
                        std::cout << " Draw " << std::endl;
                        print_tictactoe(state.second);
                    }
                    finished = true;
                    break;
                case EMPTY:
                    if (old_prev.size() > 0) {
                        player->learn(old_prev, old_action, state.second, 0.0);
                    }
                    player->learn(state.first, action, state.second, 0.0);
                    old_prev = state.first;
                    old_next = state.second;
                    old_action = action;
                    break;
            }
            AIturn = !AIturn;
        }
        mdp.reset();
    }
    return 0;
}
