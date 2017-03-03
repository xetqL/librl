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
#include "util.hpp"

typedef std::mt19937 rng_type;

template<typename TState, typename TAction>
class MDP {
public:
    TState current_state;

    /**
     Constructor with random starting state
     */
    MDP(std::function<std::vector<TState>() > state_func, std::function<std::vector<TAction>(TState) > actions,
            std::function<double(TState, TAction) > reward_func, std::function<TState(TState, TAction) > transition_func) {
        T = transition_func;
        S = state_func;
        R = reward_func;
        A = actions;
        states = S();
        current_state = *select_randomly(this->states.begin(), this->states.end());
        init_state = current_state;
    }

    /**
     Constructor with specified starting state
     */
    MDP(std::function<std::vector<TState>() > states, std::function<std::vector<TAction>(TState) > actions,
            std::function<double(TState, TAction) > reward_func, std::function<TState(TState, TAction) > transition_func, TState starting_state) {
        T = transition_func;
        S = states;
        R = reward_func;
        A = actions;
        states = S;
        current_state = starting_state;
        init_state = current_state;
    }

    std::pair<TState, TState> perform_state_transition(TAction a) {
        TState prev_state = this->current_state,
                next_state = T(this->current_state, a);
        this->current_state = next_state;
        return std::make_pair(prev_state, next_state);
    }

    double get_reward(TAction a) {
        return R(this->current_state, a);
    }

    double get_reward(TState s, TAction a) {
        return R(s, a);
    }

    std::vector<TAction> get_available_actions() {
        return this->A(this->current_state);
    }

    void reset() {
        this->current_state = this->init_state;
    }

    void randomize() {
        this->current_state = *select_randomly(this->states.begin(), this->states.end());
        this->init_state = current_state;
    }
    std::function<std::vector<TAction>(TState) > A;
private:
    std::vector<TState> states;
    TState init_state;
    std::function<TState(TState, TAction) > T;

    std::function<std::vector<TState>() > S;
    std::function<double(TState, TAction) > R;
};

/**
 * Environment Listener Markov Decision Process
 */
template<typename TState, typename TAction, typename TSignal>
class ELMDP {
public:
    TState current_state;

    std::vector<TAction> get_available_actions() {
        return this->A(this->current_state);
    }

    /**
     Constructor with random starting state
     */
    ELMDP(std::function<std::vector<TState>() > state_func, std::function<std::vector<TAction>(TState) > actions,
            std::function<double(TState, TAction, TSignal) > reward_func, std::function<TState(TState, TAction) > transition_func) {
        T = transition_func;
        S = state_func;
        R = reward_func;
        A = actions;
        states = S();
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<int> udist(0, states.size());
        int random_number = udist(mt);
        current_state = states[random_number];
    }

    /**
     Constructor with specified starting state
     */
    ELMDP(std::function<std::vector<TState>() > states, std::function<std::vector<TAction>(TState) > actions,
            std::function<double(TState, TAction, TSignal) > reward_func, std::function<TState(TState, TAction) > transition_func, TState starting_state) {
        T = transition_func;
        S = states;
        R = reward_func;
        A = actions;
        states = S();
        current_state = starting_state;
    }
private:

    std::vector<TState> states;

    std::function<TState(TState, TAction) > T;
    std::function<std::vector<TAction>(TState) > A;
    std::function<std::vector<TState>() > S;
    std::function<double(TState, TAction) > R;
};
#endif /* MDP_HPP */

