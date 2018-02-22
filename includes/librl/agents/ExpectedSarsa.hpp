#ifndef EXPECTEDSARSAAGENT_HPP
#define EXPECTEDSARSAAGENT_HPP

#include "RLAgent.hpp" // Base class: RLAgent
#include "../utils/array.hpp"

namespace librl { namespace agent {

        template<typename TState, typename TAction>
        class ExpectedSarsa : public RLAgent<TState, TAction> {
        public:

            ExpectedSarsa(
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
                return this->pi->choose_action(this->q, this->get_available_actions(), this->current_state());
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

            std::string getName() const {
                return "Expected Sarsa";
            }

            void reset() {
                this->stats = std::make_shared<librl::stats::AgentStatistics>();
                this->pi->reset();
                this->q->reset();
            }

            void set_learning_parameters(std::vector<double> parameters) {
                this->gamma = parameters[1];
            }

        protected:
            double get_reinforcement(TState prev_state, TAction action, TState next_state, double reward) const {
                double Vs = 0;
                std::unordered_map<TAction, double> probabilities = this->pi->get_probabilities(this->q, this->get_available_actions(), next_state);
                for (auto const &actionProbabilities : probabilities) { //Weighted E(a | s)
                    Vs += actionProbabilities.second * this->q->Q(next_state, actionProbabilities.first);
                }
                return reward + this->gamma * Vs;
            }
        };
    }}
#endif // EXPECTEDSARSAAGENT_HPP
