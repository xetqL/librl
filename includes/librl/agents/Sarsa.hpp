#ifndef SARSAAGENT_HPP
#define SARSAAGENT_HPP

#include "RLAgent.hpp" // Base class: RLAgent
namespace librl { namespace agent {

        template<typename TState, typename TAction>
        class Sarsa : public RLAgent<TState, TAction> {
        public:
            const std::string name = "Sarsa";

            Sarsa(
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
            TAction choose_action(const TState& state, const std::vector<TAction>& actions) {
                this->current_state   = state;
                this->current_actions = actions;
                return this->pi->choose_action(this->q, actions, state);
            }

            /**
             * @brief Method for telling the agent that the performed action is terminated
             * @param action the id of the performed action
             * @param reward the reward associated with the performed action
             */
            void learn(TState prev_state, TAction action, TState next_state, double reward) {
                //side effect : change the current state due to sarsa and set the next action
                double value = this->get_reinforcement(prev_state, action, next_state, reward);
                this->q->Q(prev_state, action, value);
            }

            void reset() {
                this->pi->reset();
                this->q->reset();
            }

            void set_learning_parameters(std::vector<double> parameters) {
                this->gamma = parameters[1];
            }
        protected:
            double get_reinforcement(TState prev_state, TAction action, TState next_state, double reward) const {
                TAction future_action = this->pi->predict_action(this->q, current_actions, next_state);
                return reward + this->gamma * this->q->Q(next_state, future_action);
            }

            TState state;
            std::vector<TAction> actions;
        };
    }}
#endif // SARSAAGENT_HPP
