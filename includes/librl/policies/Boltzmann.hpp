#ifndef BOLTZMANNPOLICY_HPP
#define BOLTZMANNPOLICY_HPP
#include "Policy.hpp"
#include "Greedy.hpp"
#include <iostream>
#include <string>
#include "../agents/RLAgent.hpp"
namespace librl { namespace policy {

        template<typename TState, typename TAction>
        class Boltzmann : public Policy<TState, TAction> {
        public:

            Boltzmann(double temperature, double coolingFactor) : gen((std::random_device())()), dis(0.0, 1.0) {
                this->init_temperature = temperature;
                this->temperature = temperature;
                this->coolingFactor = coolingFactor;
            }

            void update() {
                this->temperature = this->temperature * (1 - this->coolingFactor);
            }

            void reset() {
                this->temperature = this->init_temperature;
            }

            std::string getName() const {
                return "Boltzmann Exploration";
            }

            virtual TAction choose_action(const librl::approximator::ActionValueApproximator<TState, TAction>* f,
                                          const std::vector<TAction> &available_actions,
                                          const TState &at_state) {
                TAction selectedAction = boltzmannExploration(f, available_actions, at_state, this->temperature);
                this->update();
                return selectedAction;
            }

            virtual TAction predict_action(const librl::approximator::ActionValueApproximator<TState, TAction>* f,
                                          const std::vector<TAction> &available_actions,
                                          const TState &at_state) {
                TAction selectedAction = boltzmannExploration(f, available_actions, at_state, this->temperature);
                return selectedAction;
            }

            virtual std::unordered_map<TAction, double> get_probabilities(const librl::approximator::ActionValueApproximator<TState, TAction>* f,
                                                                const std::vector<TAction> &available_actions,
                                                                const TState &at_state) const {
                double pSIGMA = 0;

                std::unordered_map<TAction, double> probabilities;
                const double actual_temp = this->temperature;

                for (auto const &action : available_actions) {
                    pSIGMA += (double) std::exp(f->Q(at_state, action) / actual_temp);
                }

                //exploitation
                if (!std::isfinite(pSIGMA) || pSIGMA != pSIGMA || pSIGMA == 0) {
                    return Boltzmann::greedy_policy.get_probabilities(f, available_actions, at_state);
                } else {//otherwise, exploration
                    for (auto const &action : available_actions) {
                        probabilities[action] = std::exp(f->Q(at_state, action) / actual_temp) / pSIGMA;
                    }
                    return probabilities;
                }
            }

        protected:
            Greedy<TState, TAction> greedy_policy;
            std::mt19937 gen;
            std::uniform_real_distribution<double> dis;
            double init_temperature;
            double temperature;
            double coolingFactor;

            TAction boltzmannExploration(const librl::approximator::ActionValueApproximator<TState, TAction>* f,
                                         const std::vector<TAction> &available_actions,
                                         const TState &at_state,
                                         double temperature) {

                double total = 0, p, pSIGMA = 0, r = dis(gen);

                const double actual_temp = temperature;
                for (auto const &action : available_actions) {
                    pSIGMA += (double) std::exp(f->Q(at_state, action) / actual_temp);
                }
                TAction selected_action;
                if (!std::isfinite(pSIGMA) || pSIGMA != pSIGMA || pSIGMA == 0) {
                    selected_action = Boltzmann::greedy_policy.choose_action(f, available_actions, at_state);
                } else {
                    for (auto const &action : available_actions) {
                        p = std::exp(f->Q(at_state, action) / actual_temp) / pSIGMA;
                        total += p;
                        if (r <= total) {
                            selected_action = action;
                            break;
                        }
                    }
                }
                return selected_action;
            }

            std::string to_string() override {
                return std::to_string(temperature);
            }

        };
    }}
#endif // BOLTZMANNPOLICY_HPP
