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
    
    std::shared_ptr<Policy<tictactoe, Move>> policy_egreedy = dynamic_pointer_cast<Policy<tictactoe, Move >> (std::make_shared<EGreedyPolicy<tictactoe, Move >> (0.2));

    std::shared_ptr<ArrayActionValueApproximator<tictactoe, Move>> fap = std::make_shared<ArrayActionValueApproximator<tictactoe, Move >> (0.3, A);
    std::shared_ptr<ArrayActionValueApproximator<tictactoe, Move>> faai = std::make_shared<ArrayActionValueApproximator<tictactoe, Move >> (0.3, A);

    std::shared_ptr<MDP<tictactoe, Move>> tictactoe_mdp = make_shared<MDP<tictactoe, Move >> (S, A, R, T, entry_state);

    std::shared_ptr<QLearningAgent<tictactoe, Move >> qlp(
            new QLearningAgent<tictactoe, Move>({0.3, 0.9}, policy_egreedy, tictactoe_mdp, fap)
            );
    std::shared_ptr<QLearningAgent<tictactoe, Move >> qlai(
            new QLearningAgent<tictactoe, Move>({0.3, 0.9}, policy_egreedy, tictactoe_mdp, faai)
            );
            
    const int LEARNING_GAMES = atoi(argv[0]), START_SHOWING = LEARNING_GAMES - 10;

    bool show = false, finished, human_player = false;

    double tot_reward_p = 0, tot_reward_ai = 0;

    int turns_ai = 0, turns_p = 0;
    int move_index = -1;
    tictactoe old_prev, old_next;
    Move old_action;

    for (size_t game = 0; game < LEARNING_GAMES; ++game) {

        if(!(game % 5000)) 
            cout << "GAME [" << game << "]" << " average reward AI: " << tot_reward_ai / (double) turns_ai << " average reward p: " << tot_reward_ai / (double) turns_p << endl;

        if (game == START_SHOWING) {
            show = true;
            human_player = true;
        }

        finished = false;
        while (!finished && !isfinal(tictactoe_mdp->current_state)) {

            auto action = AIturn ? qlai->choose_action() : qlp->choose_action();
            
            turns_ai += AIturn ? 1 : 0;
            turns_p  += AIturn ? 0 : 1;
            
            if (AIturn && human_player) {
                std::cout << "=====================" << std::endl;
                print_tictactoe(tictactoe_mdp->current_state);
                std::cout << "Your move : ";
                std::cin >> move_index;
                action = make_pair(move_index / 3, move_index % 3);
            }

            auto state = tictactoe_mdp->perform_state_transition(action);

            piece winner = iswin(state.second);
            switch (winner) {
                case PLAYER: // O win
                case AI: // X win
                    qlp->learn(
                            (AIturn ? old_prev : state.first),
                            (AIturn ? old_action : action),
                            state.second,
                            (AIturn ? -1 : 1));
                    tot_reward_p += AIturn ? -1 : 1;
                    qlai->learn(
                            (AIturn ? state.first : old_prev),
                            (AIturn ? action : old_action),
                            state.second,
                            (AIturn ? 1 : -1));
                    tot_reward_ai += AIturn ? 1 : -1;

                    if (show) {
                        std::cout << (AIturn ? " X win " : "O win") << std::endl;
                        print_tictactoe(state.second);
                    }
                    finished = true;
                    break;
                case TIE:
                    qlp->learn(
                            (AIturn ? old_prev : state.first),
                            (AIturn ? old_action : action),
                            state.second,
                            0.5);
                    qlai->learn(
                            (AIturn ? state.first : old_prev),
                            (AIturn ? action : old_action),
                            state.second,
                            0.5);
                    tot_reward_ai += 0.5;
                    tot_reward_p  += 0.5;
                    if (show) {
                        std::cout << " Draw " << std::endl;
                        print_tictactoe(state.second);
                    }
                    finished = true;
                    break;
                case EMPTY:
                    if (old_prev.size() > 0) {
                        if (AIturn)  qlp->learn(old_prev, old_action, state.second, 0.0);
                        else        qlai->learn(old_prev, old_action, state.second, 0.0);
                    }
                    old_prev = state.first;
                    old_next = state.second;
                    old_action = action;
                    break;
            }
            AIturn = !AIturn;
        }
        tictactoe_mdp->reset();
    }
    return 0;
}
