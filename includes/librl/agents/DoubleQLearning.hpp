#ifndef DOUBLEQLEARNING_HPP
#define DOUBLEQLEARNING_HPP

#include "RLAgent.hpp" // Base class: RLAgent
namespace librl { namespace agent {

        template<typename TState, typename TAction>
        class DoubleQLearning : public RLAgent<TState, TAction> {
        public:
            const std::string name = "Double Q-Learning";

            DoubleQLearning(
                    librl::policy::Policy<TState, TAction> *pi,
                    librl::approximator::DoubleApproximator<TState, TAction> *dba,
                    double discount_factor)
                    : RLAgent<TState, TAction>(pi, dba, discount_factor) {
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

            void set_learning_parameters(std::vector<double> parameters) {
                this->gamma = parameters[1];
                this->alpha = parameters[0];
            }

            void reset() {
                this->pi->reset();
                this->q->reset();
            }

            TAction choose_action(const TState& state, const std::vector<TAction>& actions) {
                TAction action = this->pi->choose_action(this->q, actions, state);
                return action;
            }

        protected:
            double get_reinforcement(TState prev_state, TAction action, TState next_state, double reward) const {
                auto function_approximators = std::dynamic_pointer_cast<librl::approximator::DoubleApproximator<TState, TAction> >(
                        this->q)->get_FA_update_pair();
                auto fstar = function_approximators.first->argmax(next_state);
                double q_value = function_approximators.first->Q(prev_state, action);
                return reward + gamma * function_approximators.second->Q(next_state, fstar);
            }

        };
    }}
#endif // DOUBLEQLEARNING_HPP
