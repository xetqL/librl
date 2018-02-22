#ifndef QLEARNING_AGENT
#define QLEARNING_AGENT

#include "RLAgent.hpp"
#include "../approximators/FunctionApproximator.hpp"
#include <iostream>
#include <limits>
#include <map>
namespace librl { namespace agent {

        template<typename TState, typename TAction>
        class QLearning : public RLAgent<TState, TAction> {
        public:

            QLearning(
                    librl::policy::Policy<TState, TAction> *pi,
                    librl::environment::MDP<TState, TAction> *mdp,
                    librl::approximator::ActionValueApproximator<TState, TAction> *ava,
                    double discount_factor)
                    : RLAgent<TState, TAction>(pi, mdp, ava, discount_factor) {
            }

            /**
             * @brief Method for asking the RL agent to starting to perform an action
             * @return Gives the id of the action to perform
             */
            TAction choose_action() const {
                TAction action = this->pi->choose_action(this->q, this->get_available_actions(), this->current_state());
                return action;
            }

            std::string getName() const {
                return "QLearning";
            }

            void reset() {
                this->pi->reset();
                this->q->reset();
            }

            void set_learning_parameters(std::vector<double> parameters) {
                this->gamma = parameters[1];
            }

            /**
             * @brief Method for telling the agent that the performed action is terminated
             * @param action the id of the performed action
             * @param reward the reward associated with the performed action
             */
            void learn(TState prev_state, TAction action, TState next_state, double reward) {
                double value = this->get_reinforcement(prev_state, action, next_state, reward);
                this->q->Q(prev_state, action, value);
            }

        protected:
            double get_reinforcement(TState prev_state, TAction action, TState next_state, double reward) const {
                return reward + this->gamma * this->q->max(next_state);
            }
        };
    }}
#endif
