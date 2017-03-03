/* 
 * File:   gridworld.hpp
 * Author: xetql
 *
 * Created on March 2, 2017, 4:28 PM
 */

#ifndef GRIDWORLD_HPP
#define GRIDWORLD_HPP

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

#define EMPTY    0
#define START    1    
#define PLAYER   2
#define OBSTACLE 3
#define HOLE     4
#define STOP     5

using namespace std;

typedef std::vector<std::vector<int>> Maze;

typedef std::pair<int, int> Position;

typedef int Move;

#define LEFT  0
#define RIGHT 1
#define UP    2
#define DOWN  3

/**
 * O(n*m) algorithm to locate player in a maze
 * @param m
 * @return The player position in the maze
 */
Position locate_player(Maze m) {
    for (size_t i = 0; i < m.size(); i++) {
        for (size_t j = 0; j < m[i].size(); j++) {
            if (m[i][j] == PLAYER) return make_pair<int, int>(i, j);
        }
    }
    throw "No player on board !";
}

/**
 * O(n*m) algorithm to locate player in a maze
 * @param m
 * @return The player position in the maze
 */
Position locate(Maze m, int target_type) {
    for (size_t i = 0; i < m.size(); i++) {
        for (size_t j = 0; j < m[i].size(); j++) {
            if (m[i][j] == target_type) return make_pair<int, int>(i, j);
        }
    }
    return make_pair<int, int>(-1, -1);
}

void print_maze(Maze m) {
    cout << "--------------------" << endl;
    for (auto const &row : m) {
        for (auto const &cell : row) {
            switch (cell) {
            case PLAYER:
                cout << '|' << 'P';
                break;
            case EMPTY:
                cout << '|' << ' ';
                break;
            case OBSTACLE:
                cout << '|' << '#';
                break;
            case HOLE:
                cout << '|' << 'o';
                break;
            case STOP:
                cout << '|' << 'x';
                break;
            }
        }
        cout << endl << "--------------------" << endl;
    }
}

/**
 * Valid a move in the maze
 * @param maze
 * @param m
 * @return if the move is valid
 */

bool is_valid(Maze maze, int m, Position player_location) {
    switch (m) {
    case LEFT:
        return player_location.second > 0
                && maze[player_location.first][player_location.second - 1] != OBSTACLE;
        break;
    case RIGHT:
        return player_location.second + 1 < maze[0].size() &&
                maze[player_location.first][player_location.second + 1] != OBSTACLE;
        break;
    case UP:
        return player_location.first > 0 &&
                maze[player_location.first - 1][player_location.second] != OBSTACLE;
        break;
    case DOWN:
        return player_location.first + 1 < maze.size() &&
                maze[player_location.first + 1][player_location.second] != OBSTACLE;
        break;
    default:
        return false;
    }
}

/**
 * Apply a move to a player position
 * @param player_location
 * @param m
 * @return the new player position according to the move
 */
Position apply_move(Position player_location, Move m) {
    switch (m) {
    case LEFT:
        return std::make_pair(player_location.first, player_location.second - 1);
        break;
    case RIGHT:
        return std::make_pair(player_location.first, player_location.second + 1);
        break;
    case UP:
        return std::make_pair(player_location.first - 1, player_location.second);
        break;
    case DOWN:
        return std::make_pair(player_location.first + 1, player_location.second);
        break;
    default:
        throw "Unacceptable move !";
    }
}

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
        return 10.0;
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

void clear() {
    // CSI[2J clears screen, CSI[H moves the cursor to top-left corner
    std::cout << "\x1B[2J\x1B[H";
}


#endif /* GRIDWORLD_HPP */

