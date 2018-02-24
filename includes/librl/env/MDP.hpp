/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   MDP.hpp
 * Author: xetql
 *
 * Created on February 11, 2017, 2:00 PM
 */

#ifndef MDP_HPP
#define MDP_HPP
#include <functional>
#include <vector>
#include <random>
#include <utility>
#include "../utils/util.hpp"

typedef std::mt19937 rng_type;
namespace librl { namespace environment {
template<typename TState, typename TAction>
class MDP {
public:
    TState current_state;

    MDP(std::function<std::vector<TState>() > states,
        std::function<std::vector<TAction>(TState) > actions,
        std::function<double(TState, TAction) > reward_func,
        std::function<TState(TState, TAction) > transition_func,
        TState starting_state) {
        T = transition_func;
        S = states;
        R = reward_func;
        A = actions;
        current_state = starting_state;
    }

    MDP(std::function<std::vector<TAction>(TState) > actions,
        std::function<double(TState, TAction) > reward_func,
        std::function<TState(TState, TAction) > transition_func,
        TState starting_state) {
        T = transition_func;
        R = reward_func;
        A = actions;
        current_state = starting_state;
    }

    std::pair<TState, TState> perform_state_transition(TAction a) {
        TState prev_state = this->current_state, next_state = T(this->current_state, a);
        this->current_state = next_state;
        return std::make_pair(prev_state, next_state);
    }
    double get_reward(TAction a) { return R(this->current_state, a); }
    double get_reward(TState s, TAction a) { return R(s, a); }
    std::vector<TAction> get_available_actions() { return this->A(this->current_state); }

private:
    std::function<std::vector<TState>() >        S;
    std::function<std::vector<TAction>(TState) > A;
    std::function<double(TState, TAction) >      R;
    std::function<TState(TState, TAction) >      T;
};
}}
#endif /* MDP_HPP */

