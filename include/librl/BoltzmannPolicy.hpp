#ifndef BOLTZMANNPOLICY_HPP
#define BOLTZMANNPOLICY_HPP
#include "Policy.hpp"
#include "GreedyPolicy.hpp"
#include <iostream>
#include <string>

template<typename TState, typename TAction>
class BoltzmannPolicy : public Policy<TState, TAction> {
public:

    BoltzmannPolicy(double temperature, double coolingFactor) {
        this->init_temperature = temperature;
        this->temperature = temperature;
        this->coolingFactor = coolingFactor;
    }

    void update() override {
        this->temperature = this->temperature * (1 - this->coolingFactor);
    }

    void reset() {
        this->temperature = this->init_temperature;
    }

    std::string getName() const {
        return "Boltzmann Exploration";
    }

    TAction choose_action(const RLAgent<TState, TAction>* agent) const {
        TAction selectedAction = boltzmannExploration(agent, this->temperature);
        return selectedAction;
    }

    std::map<TAction, double> get_probabilities(const RLAgent<TState, TAction>* agent, TState state) const {
        double pSIGMA = 0;

        std::map<TAction, double> probabilities;
        std::vector<TAction> actions = agent->get_available_actions();
        const double actual_temp = this->temperature;

        for (auto const &action : actions) {
            pSIGMA += (double) std::exp(agent->q->Q(state, action) / actual_temp);
        }

        //exploitation
        if (!std::isfinite(pSIGMA) || pSIGMA != pSIGMA || pSIGMA == 0) {
            GreedyPolicy<TState, TAction> g();
            return g.get_probabilities(agent, state);
        } else {//otherwise, exploration
            for (auto const &action : actions) {
                probabilities[action] = std::exp(agent->q->Q(state, action) / actual_temp) / pSIGMA;
            }
            return probabilities;
        }
    }
protected:

    TAction boltzmannExploration(const RLAgent<TState, TAction> *agent, double temperature) const {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, 1);
        double total = 0, p, pSIGMA = 0, r = dis(gen);
        const TState current_state = agent->current_state();
        const double actual_temp = temperature;
        for (auto const &action : agent->get_available_actions()) {
            pSIGMA += (double) std::exp(agent->q->Q(current_state, action) / actual_temp);
        }

        TAction selected_action;

        if (!std::isfinite(pSIGMA) || pSIGMA != pSIGMA || pSIGMA == 0) {
            GreedyPolicy<TState, TAction> g;
            selected_action = g.choose_action(agent);
        } else {
            for (auto const &action : agent->get_available_actions()) {
                p = std::exp(agent->q->Q(current_state, action) / actual_temp) / pSIGMA;
                total += p;
                if (r <= total) {
                    selected_action = action;
                    break;
                }
            }
            //std::cout << selected_action << std::endl;

        }

        return selected_action;
    }

    std::string to_string() override {
        return std::to_string(temperature);
    }

    double init_temperature;
    double temperature;
    double coolingFactor;
};
#endif // BOLTZMANNPOLICY_HPP
