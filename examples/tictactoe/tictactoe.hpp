/* 
 * File:   tictactoe.hpp
 * Author: xetql
 *
 * Created on February 28, 2017, 11:06 AM
 */

#ifndef TICTACTOE_HPP
#define TICTACTOE_HPP

#include <librl/RLAlgorithms.hpp>
#include <librl/Policies.hpp>
#include <librl/util.hpp>
#include <librl/MDP.hpp>
#include <librl/FunctionApproximator.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

#define EMPTY  0
#define AI     1
#define PLAYER 2
#define TIE    3

typedef int piece;
typedef std::vector<std::vector<piece>> tictactoe;
typedef std::pair<int, int> Move;

bool AIturn = false;

bool isfinal(tictactoe arr) {
    int space_left = 0;
    for (auto const &row : arr) {
        for (auto const &cell : row) {
            space_left += cell == EMPTY ? 1 : 0;
        }
    }
    return space_left == 0;
}

std::vector<Move> A(tictactoe state) {
    std::vector<Move> moves;
    for (size_t i = 0; i < state.size(); ++i) {
        for (size_t j = 0; j < state[i].size(); ++j) {
            if (state[i][j] == EMPTY) moves.push_back(std::make_pair(i, j));
        }
    }
    return moves;
}

piece iswin(tictactoe arr) {
    piece ret = isfinal(arr) ? TIE : EMPTY;

    // Checks for horizontal win
    for (int i = 0; i < 3; ++i)
        if (arr[i][0] == arr[i][1] && arr[i][1] == arr[i][2])
            if ((ret = arr[i][0]) != EMPTY)
                return arr[i][0];

    // Checks for vertical win
    for (int i = 0; i < 3; ++i)
        if (arr[0][i] == arr[1][i] && arr[1][i] == arr[2][i])
            if ((ret = arr[0][i]) != EMPTY)
                return arr[0][i];

    // Check for diagonal win (upper left to bottom right)
    if (arr[0][0] == arr[1][1] && arr[1][1] == arr[2][2])
        if ((ret = arr[0][0]) != EMPTY)
            return arr[0][0];

    // Check for diagonal win (upper right to bottom left)
    if (arr[0][2] == arr[1][1] && arr[1][1] == arr[2][0])
        if ((ret = arr[0][2]) != EMPTY)
            return arr[0][2];

    return ret;
}

void testIsFinal() {
    tictactoe twon = {
        {AI, AI, PLAYER},
        {AI, PLAYER, AI},
        {PLAYER, PLAYER, AI}
    }, tlost = {
        {AI, AI, AI},
        {AI, PLAYER, PLAYER},
        {PLAYER, PLAYER, AI}
    }, ttie = {
        {PLAYER, AI, AI},
        {AI, PLAYER, PLAYER},
        {PLAYER, AI, AI}
    }, tnotfinished = {
        {AI, EMPTY, AI},
        {AI, PLAYER, PLAYER},
        {PLAYER, PLAYER, AI}
    };

    if (!isfinal(twon)) throw std::runtime_error("twon: It should be finished");
    if (!isfinal(tlost)) throw std::runtime_error("tlost: It should be finished");
    if (!isfinal(ttie)) throw std::runtime_error("ttie: It should be finished");
    if (isfinal(tnotfinished)) throw std::runtime_error("tnotfinished: It should not be finished");

    std::cout << "testIsFinal Test passed !" << std::endl;
}

void testIsWin() {
    tictactoe twon = {
        {AI, AI, PLAYER},
        {AI, PLAYER, AI},
        {PLAYER, PLAYER, AI}
    }, tlost = {
        {AI, AI, AI},
        {AI, PLAYER, PLAYER},
        {PLAYER, PLAYER, AI}
    }, ttie = {
        {PLAYER, AI, AI},
        {AI, PLAYER, PLAYER},
        {PLAYER, AI, AI}
    }, tnotfinished = {
        {AI, EMPTY, AI},
        {AI, PLAYER, PLAYER},
        {PLAYER, PLAYER, AI}
    };

    if (iswin(twon) != PLAYER) throw "Player should have win this game !";

    if (iswin(tlost) != AI) throw "AI should have win this game !";

    if (iswin(ttie) != TIE) throw "Nobody should have win the game !";

    if (iswin(tnotfinished) != EMPTY) throw "The game is not finished !";

    std::cout << "testIsWin Test passed !" << std::endl;

}

double R(tictactoe state, Move action) {
    tictactoe next_state = state;
    next_state[action.first][action.second] = AIturn ? AI : PLAYER;

    switch (iswin(next_state)) {
        case AI: //loose
            return -1000.0;
        case PLAYER:
            return 0.0;
        case TIE:
            return 0.0;
        default:
            return 0.0;
    }
}

tictactoe T(tictactoe state, Move action) {
    tictactoe next_state = state;
    next_state[action.first][action.second] = AIturn ? AI : PLAYER;
    return next_state;
}

std::vector<tictactoe> S() {
    //It is not mandatory to know all the state space
    return {};
}

bool is_a_winning_move(tictactoe state, Move action, piece whom) {
    tictactoe next_state = state;
    next_state[action.first][action.second] = whom;
    return iswin(next_state) == whom;
}

Move random_good_move(tictactoe state, piece whom) {
    std::vector<Move> mvs;
    std::vector<Move> all_actions = A(state);

    for (auto const &a : all_actions) {
        if (a.first == 2 && a.second == 2) return a;
        if (a.first == 0 && a.second == 2) return a;
        if (a.first == 2 && a.second == 0) return a;
        if (a.first == 1 && a.second == 1) return a;
        
        if (is_a_winning_move(state, a, whom)) {
            return a;
        }
    }
    Move a = *select_randomly(all_actions.begin(), all_actions.end());
    return mvs.begin() == mvs.end() ? a : *select_randomly(mvs.begin(), mvs.end());
}

char piece_to_char(piece p) {
    switch (p) {
        case AI:
            return 'X';
        case PLAYER:
            return 'O';
        default:
            return ' ';
    }
}

void print_tictactoe(tictactoe m) {
    for (auto const &row : m) {
        for (auto const &cell : row) {
            std::cout << '|' << piece_to_char(cell);
        }
        std::cout << '|' << std::endl;
    }
}

void clear() {
    // CSI[2J clears screen, CSI[H moves the cursor to top-left corner
    std::cout << "\x1B[2J\x1B[H";
}

#endif /* TICTACTOE_HPP */

