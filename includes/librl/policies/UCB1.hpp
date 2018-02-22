#ifndef UCB1POLICY_HPP
#define UCB1POLICY_HPP

#include <cassert>
#include <unordered_map>
#include "Policy.hpp"

namespace librl { namespace policy {
    template<typename TState, typename TAction>
    class UCB1 : public Policy<TState, TAction> {
    public:

        virtual TAction choose_action(const librl::approximator::ActionValueApproximator<TState, TAction>* f,
                                      const std::vector<TAction> &available_actions,
                                      const TState &at_state) {
            auto action = ucb1Exploration(f, available_actions, at_state);
            this->step_counter++;
            return action;
        }

        std::string getName() const {
            return "UCB1 Exploration";
        }

        void reset() {
            this->step_counter = 0;
            this->numberOfTimeAction.clear();
        }

        virtual std::unordered_map<TAction, double> get_probabilities(const librl::approximator::ActionValueApproximator<TState, TAction>* f,
                                                            const std::vector<TAction> &available_actions,
                                                            const TState &at_state) const {
            const int nb_actions = available_actions.size();
            std::unordered_map<TAction, double> probabilities;
            double ucbValue, max = std::numeric_limits<double>::lowest();
            TAction selected_action;
            for (auto const &action : available_actions) {
                ucbValue = f->Q(at_state, action) + std::sqrt(2.0 * std::log(step_counter / this->numberOfTimeAction[action]));
                if (ucbValue > max) {
                    max = ucbValue;
                    selected_action = action;
                }
                probabilities[action] = 0.0;
            }
            probabilities[selected_action] = 1.0;
            return probabilities;
        }

        TAction ucb1Exploration(const librl::approximator::ActionValueApproximator<TState, TAction>* f,
                                const std::vector<TAction> &available_actions,
                                const TState &at_state) const {
            double max = std::numeric_limits<double>::lowest(), ucbValue;
            const int nb_actions = available_actions.size();
            TAction selected_action;
            for (auto const &action : available_actions) {
                ucbValue = f->Q(at_state, action) + std::sqrt(2.0 * std::log(step_counter / this->numberOfTimeAction[action]));
                if (ucbValue > max) {
                    max = ucbValue;
                    selected_action = action;
                }
            }
            return selected_action;
        }
    protected:
        unsigned int step_counter = 0;
        std::unordered_map<TAction, int> numberOfTimeAction;
    };
}}
#endif // UCB1POLICY_HPP
