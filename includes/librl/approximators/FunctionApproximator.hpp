/*
 * File:   FunctionApproximator.hpp
 * Author: xetql
 *
 * Created on February 20, 2017, 5:05 PM
 */

#ifndef FUNCTIONAPPROXIMATOR_HPP
#define FUNCTIONAPPROXIMATOR_HPP
#include <vector>
#include <set>
#include <unordered_map>
#include <limits>
#include <iostream>
#include <utility>
#include <type_traits>
#include <random>

#include "../utils/util.hpp"
#include <armadillo>
namespace librl{ namespace approximator {
        class FunctionApproximator {
            virtual double get_learning_rate() const = 0;
        };

        template<typename TState, typename TAction>
        class ActionValueApproximator : public FunctionApproximator {
        public:

            ActionValueApproximator(double _alpha) : alpha(_alpha){}

            /**
             * Get argmax_a Q(s,a), it may contains several indices
             * @param state
             * @return argmax_a Q(s,a)
             */

            virtual TAction argmax(const TState state, std::vector<TAction> available_actions = std::vector<TAction>()) const = 0;

            /**
             * @brief Get max_a Q(s,a)
             */
            virtual double max(TState state) const = 0;

            /**
             * @brief Get Q(s,a)
             */
            virtual double Q(TState state, TAction action) const = 0;

            /**
             * @brief reset the function approximation
             */
            virtual void reset() = 0;

            void set_learning_parameter(double alpha) {
                this->alpha = alpha;
            }

            /**
             * Setter of the Q function
             * @param state
             * @param action
             * @param value
             */
            virtual void Q(TState state, TAction action, double value) = 0;

            virtual double get_learning_rate() const {
                return alpha;
            }
            double alpha;
        };

        template<typename TState, typename TAction>
        class ArrayActionValueApproximator : public ActionValueApproximator<TState, TAction> {
        public:

            ArrayActionValueApproximator(double _alpha, double default_value=0.0) : ActionValueApproximator<TState, TAction>(_alpha), default_value(default_value) {};

            TAction argmax(const TState state, std::vector<TAction> available_actions = std::vector<TAction>()) const {
                double max = std::numeric_limits<double>::lowest();
                std::vector <TAction> all_actions;
                if(this->_Q.find(state) == this->_Q.end()) { //state have never been seen -> available action should help
                    if(available_actions.size() == 0) throw std::runtime_error("State has never been seen and no available action is given... in FunctionApproximator.hpp");
                    return *select_randomly(available_actions.begin(), available_actions.end());
                }

                if(available_actions.size() == 0) { // no action is provided by caller
                    auto state_map = this->_Q.at(state);
                    std::for_each(state_map.begin(), state_map.end(), [&](const auto& pair) mutable {available_actions.push_back(pair.first);});
                }

                if(this->_Q.find(state) != this->_Q.end()){
                    for(auto const &action : available_actions) { //go through actions
                        auto action_it = this->_Q.at(state).find(action);
                        double action_value = action_it != this->_Q.at(state).end() ? action_it->second : default_value;
                        if(max == action_value)
                            all_actions.push_back(action);
                        else if(max < action_value){
                            max = action_value;
                            all_actions.clear();
                            all_actions.push_back(action);
                        }
                    }
                    return *select_randomly(all_actions.begin(), all_actions.end());
                } else {
                    return *select_randomly(available_actions.begin(), available_actions.end());
                }
            }

            /**
             * @brief Get max_a Q(s,a)
             */
            double max(TState state) const {
                double max = std::numeric_limits<double>::lowest();
                if(this->_Q.find(state) == this->_Q.end()) return default_value;
                if(this->_Q.at(state).size() == 0) return default_value;
                for(auto const &action_value : this->_Q.at(state)){ //go through actions
                    double qv = action_value.second;
                    if (max <= qv) max = qv;
                }
                return max == std::numeric_limits<double>::lowest() ? default_value : max;
            }

            /**
             * Get the Expected Reward for a in s.
             * @param state
             * @param action
             * @return
             */
            double Q(TState state, TAction action) const {
                if (this->_Q.find(state) == this->_Q.end()) return 0.0;
                if (this->_Q.at(state).find(action) == this->_Q.at(state).end()) return 0.0;
                return this->_Q.at(state).at(action);
            }

            /**
             * @brief reset the function approximation
             */
            void reset() {
                this->_Q.clear();
            }

            int number_of_explored_states() {
                return this->_Q.size();
            }

            bool has_seen_state(TState s) {
                return (this->_Q.find(s) != this->_Q.end());
            }

            /**
             * Setter of the Q function with value from RL agent
             * @param state
             * @param action
             * @param value
             */
            void Q(TState state, TAction action, double value) {
                if (this->_Q.find(state) == this->_Q.end()) //init state
                    this->_Q[state] = std::unordered_map<TAction, double>();

                if (this->_Q.at(state).find(action) == this->_Q.at(state).end()) //init action
                    this->_Q[state][action] = 0.0;

                this->_Q[state][action] = this->Q(state, action) + this->alpha * (value - this->Q(state, action));
            }
        protected:
            double default_value;
            std::unordered_map <TState, std::unordered_map<TAction, double>> _Q; //The approximated function as a table
        };

        template<typename TState, typename TAction>
        class DoubleApproximator : public ActionValueApproximator<TState, TAction> {
        private:
            std::shared_ptr <ArrayActionValueApproximator<TState, TAction>> qa, qb;
            bool left_function_turn;
            std::mt19937 gen;
            std::uniform_int_distribution<> distribution;
        public:
            DoubleApproximator(double _alpha) :
                ActionValueApproximator<TState, TAction>(_alpha),
                qa(std::make_shared<ArrayActionValueApproximator<TState, TAction>>(_alpha)),
                qb(std::make_shared<ArrayActionValueApproximator<TState, TAction>>(_alpha)),
                gen((std::random_device())()) {
                distribution = std::uniform_int_distribution<int>(0, 1);
                left_function_turn = (bool) distribution(gen);
            };

            DoubleApproximator(
                    double _alpha,
                    std::shared_ptr <ArrayActionValueApproximator<TState, TAction>> left_approximator,
                    std::shared_ptr <ArrayActionValueApproximator<TState, TAction>> right_approximator) :
                    ActionValueApproximator<TState, TAction>(_alpha),
                    qa(left_approximator),
                    qb(right_approximator),
                    gen((std::random_device())()) {
                distribution = std::uniform_int_distribution<int>(0, 1);
                left_function_turn = (bool) distribution(gen);
            };

            /***************************************************************************
             * Get argmax Q(s,a)
             * @param state
             * @return argmax_a Q(s,a)
             **************************************************************************/
            TAction argmax(const TState state, std::vector<TAction> available_actions = std::vector<TAction>()) const {
                if(left_function_turn){
                    return this->qa->argmax(state, available_actions);
                } else {
                    return this->qb->argmax(state, available_actions);
                }
            }

            /***************************************************************************
             * Get a pair of function approximator, the first element of the pair
             * is the approximator that will be updated in the next update
             * and the second element is the other one.
             * @return argmax_a Q(s,a)
             *************************************************************************/
            std::pair<ActionValueApproximator<TState, TAction> *, ActionValueApproximator<TState, TAction> *> get_FA_update_pair() {
                auto update_next = left_function_turn ? this->qa : this->qb;
                auto other = left_function_turn ? this->qb : this->qa;
                return std::make_pair<
                        ArrayActionValueApproximator<TState, TAction> *,
                        ArrayActionValueApproximator<TState, TAction> * >(update_next, other);
            }

            /***************************************************************************
             * @brief Get max Q(s,a)
             *************************************************************************/
            double max(TState state) const {
                if (left_function_turn) {
                    return this->qa->max(state);
                } else {
                    return this->qb->max(state);
                }
            }

            /**
             * Get the merged Q values
             * @param  state  current state
             * @param  action action taken
             * @return        Expected value of the action given the state
             */
            double Q(TState state, TAction action) const {
                return merge(this->qa->Q(state, action), this->qb->Q(state, action));
            }

            /***************************************************************************
             * @brief reset the function approximation
             *************************************************************************/
            void reset() {
                this->qa->reset();
                this->qb->reset();
            }

            void Q(TState state, TAction action, double value) {
                // random update
                if (left_function_turn) {
                    this->qa->Q(state, action, value);
                } else {
                    this->qb->Q(state, action, value);
                }
                //select the fa to update in the next turn ...
                left_function_turn = (bool) distribution(gen);
            }

        protected:
            /***************************************************************************
             * Merging strategy
             * @param  QaV the value of the left function approximator
             * @param  QbV the value of the right function approximator
             * @return     The merged values of Qa and Qb
             ***************************************************************************/
            double merge(double QaV, double QbV) {
                return (QaV + QbV) / 2.0;
            }

        };

        template<typename TState, typename TAction>
        class AfterstateValueApproximator : public ActionValueApproximator<TState, TAction> {
        public:

            AfterstateValueApproximator(double _alpha, std::function<std::vector<TAction>(TState)> _available_actions,
                                        std::function<TState(TState, TAction)> _state_transition)
                    : ActionValueApproximator<TState, TAction>(_alpha, _available_actions),
                      state_transition(_state_transition) {
            };

            AfterstateValueApproximator(std::function<std::vector<TAction>(TState)> _available_actions,
                                        std::function<TState(TState, TAction)> _state_transition)
                    : ActionValueApproximator<TState, TAction>(0.1, _available_actions),
                      state_transition(_state_transition) {
            };

            /**
             * Get argmax_a Q(s,a), it may contains several indices
             * @param state
             * @return argmax_a Q(s,a)
             */
            TAction argmax(const TState state, std::vector<TAction> available_actions = std::vector<TAction>()) const {
                double max = this->max(state);

                std::vector <TAction> all_actions;
                if(this->_Q.find(state) == this->_Q.end()) { //state have never been seen -> available action should help
                    if(available_actions.size() == 0) throw std::runtime_error("State has never been seen and no available action is given... in FunctionApproximator.hpp");
                    return *select_randomly(available_actions.begin(), available_actions.end());
                }

                if(available_actions.size() == 0) { // no action is provided by caller
                    auto state_map = this->_Q.at(state);
                    std::for_each(state_map.begin(), state_map.end(), [&](const auto& pair) mutable {available_actions.push_back(pair.first);});
                }

                for(auto const &action : available_actions) { //go through actions
                    if (max == this->_Q.at(state).at(action))
                        all_actions.push_back(action);
                }

                return *select_randomly(all_actions.begin(), all_actions.end());
            }

            /**
             * Setter of the Q function with value from RL agent
             * @param state
             * @param action
             * @param value
             */
            void Q(TState state, TAction action, double value) {
                if (_afterstate.find(state_transition(state, action)) == _afterstate.end())
                    _afterstate[state_transition(state, action)] = 1.0;

                if (this->_Q.find(state) == this->_Q.end())
                    this->_Q[state] = std::unordered_map<TAction, double>();

                if (this->_Q.at(state).find(action) == this->_Q.at(state).end())
                    this->_Q[state][action] = _afterstate.at(state_transition(state, action));

                this->_afterstate[state_transition(state, action)] =
                        this->Q(state, action) + this->alpha * (value - this->Q(state, action));
                this->_Q[state][action] = this->_afterstate[state_transition(state, action)];
            }

            /**
             * @brief Get max_a Q(s,a)
             */
            double max(TState state) const {
                double max = std::numeric_limits<double>::lowest();
                if(this->_Q.find(state) == this->_Q.end()) return 0.0;
                if(this->_Q.at(state).size() == 0) return 0.0;

                for(auto const &state_action : this->_Q){ //go through states
                    for(auto const &action_value : state_action.second){ //go through actions
                        double qv = action_value.second;
                        if (max <= qv) max = qv;
                    }
                }
                return max == std::numeric_limits<double>::lowest() ? 0.0 : max;
            }

            /**
             * Get the Expected Reward for a in s.
             * @param state
             * @param action
             * @return
             */
            double Q(TState state, TAction action) const {
                if (_afterstate.find(state_transition(state, action)) == _afterstate.end())
                    return 1.0;

                if (this->_Q.find(state) == this->_Q.end())
                    return _afterstate.at(state_transition(state, action));

                if (this->_Q.at(state).find(action) == this->_Q.at(state).end())
                    return _afterstate.at(state_transition(state, action));

                return this->_Q.at(state).at(action);
            }

            /**
             * @brief reset the function approximation
             */
            void reset() {
                this->_Q.clear();
            }

            bool has_seen_state(TState s) {
                return (this->_Q.find(s) != this->_Q.end());
            }

            int number_of_explored_states() {
                return this->_Q.size();
            }

        protected:
            std::unordered_map <TState, std::unordered_map<TAction, double>> _Q;
            std::unordered_map<TState, double> _afterstate;
            std::function<TState(TState, TAction)> state_transition;
        };
//
//        template<typename TState=arma::mat>
//        class MLPActionValueApproximator : public ActionValueApproximator<TState, int> {
//        public:
//
//            MLPActionValueApproximator(int _input_neurons,
//                                       int _hidden_neurons,
//                                       int _output_neurons,
//                                       std::function<std::vector<int>(TState)> _available_actions)
//                    : ActionValueApproximator<TState, int>(0.3, _available_actions) {
//                if (net.create_standard(3, _input_neurons,_hidden_neurons, _output_neurons)) {
//                    std::cout << "Network created" << std::endl;
//                    net.set_activation_function_hidden(FANN::LINEAR);
//                    net.set_activation_function_output(FANN::LINEAR);
//                    net.randomize_weights(-1, 1);
//                    net.print_parameters();
//                } else {
//                    std::cout << "Network not created" << std::endl;
//                }
//            }
//
//            /**
//             * TODO: Randomize the weights of the network
//             */
//            void reset() {
//                net.randomize_weights(-1, 1);
//            };
//
//            /**
//             * There is no learning parameter when using neural network as a function
//             * approximation
//             * @param alpha
//             */
//            void set_learning_parameter(double alpha) {
//                net.set_learning_rate(alpha);
//            }
//
//            /**
//             * Get argmax_a Q(s,a), it may contains several indices
//             * Query NN
//             * @param state
//             * @return argmax_a Q(s,a)
//             */
//            int argmax(TState state) {
//                //verify sizes of input and topology
//                assert(state_as_vect.size() == net.get_num_input());
//
//                fann_type in[net.get_num_input()];
//                fann_type *run_out;
//
//                for (size_t i = 0; i < net.get_num_input(); ++i) in[i] = state_as_vect[i];
//
//                //predict value for state
//                run_out = net.run(in);
//
//                //find argmax in prediction value which is available in that state
//                return filtered_argmax(run_out, state, this->available_actions);
//            }
//
//            /**
//             * Setter of the Q function
//             * @param state
//             * @param action
//             * @param value
//             */
//            void Q(TState state, int action, double value) {
//
//                //verify sizes of input and topology
//                assert(state_as_vect.size() == net.get_num_input());
//
//                fann_type in[net.get_num_input()];
//                fann_type *run_out, desired_out[net.get_num_output()];
//                for (size_t i = 0; i < net.get_num_input(); ++i) {
//                    in[i] = state_as_vect[i];
//                }
//
//                //predict value for state
//                run_out = net.run(in);
//
//                for (size_t i = 0; i < net.get_num_output(); ++i) {
//                    if (i == action) desired_out[action] = value;
//                    else desired_out[i] = run_out[i];
//                }
//
//                net.train(in, desired_out);
//
//                //std::cout <<"MSE " <<net.get_MSE() << std::endl;
//            }
//
//            /**
//             * @brief Get max Q(s,a)
//             */
//            double max(TState state) {
//                //verify sizes of input and topology
//                assert(state_as_vect.size() == net.get_num_input());
//
//                fann_type in[net.get_num_input()];
//                fann_type *run_out;
//                for (size_t i = 0; i < net.get_num_input(); ++i) in[i] = state_as_vect[i];
//                //predict value for state
//                run_out = net.run(in);
//                //find argmax in prediction value which is available in that state
//                return filtered_max(run_out, state, this->available_actions);
//            }
//
//            int filtered_argmax(fann_type *matrix, TState state, std::function<std::vector<int>(TState)> A) {
//                double max = filtered_max(matrix, state, A);
//                std::vector<int> actions = A(state), argsmax;
//                for (auto const &action : actions) {
//                    if (matrix[action] == max) argsmax.push_back(action);
//                }
//                return *select_randomly(argsmax.begin(), argsmax.end());
//            }
//
//            double filtered_max(fann_type *matrix, TState state, std::function<std::vector<int>(TState)> A) {
//                std::vector<int> actions = A(state);
//                double max = std::numeric_limits<double>::lowest();
//                for (auto const &action : actions) {
//                    if (matrix[action] > max) max = matrix[action];
//                }
//                return max;
//            }
//
//            /**
//             * @brief Get Q(s,a)
//             */
//            double Q(TState state, int action) {
//                //verify sizes of input and topology
//                assert(state_as_vect.size() == net.get_num_input());
//
//                fann_type in[net.get_num_input()];
//                fann_type *out;
//
//                for (size_t i = 0; i < net.get_num_input(); ++i) in[i] = state_as_vect[i];
//
//                //predict value for state
//                out = net.run(in);
//
//                //find argmax in prediction value which is available in that state
//                return out[action];
//            }
//
//            FANN::neural_net net;
//        };

        template<typename TState>
        class StateValueApproximator : public FunctionApproximator {
        public:

            StateValueApproximator(double beta) : beta(beta) {
            }

            /**
             * Get argmax_s V(s) from the states already seen by the agent
             * @param state
             * @return argmax_a Q(s,a)
             */
            virtual TState argmax() const = 0;

            /**
             * Setter of the V function
             * @param state
             * @param action
             * @param value
             */
            virtual void V(TState state, double value) = 0;

            /**
             * @brief Get max V(s)
             */
            virtual double max() const = 0;

            /**
             * @brief Get V(s)
             */
            virtual double V(TState state) const = 0;

            /**
             * @brief reset the function approximation
             */
            virtual void reset() = 0;

            /**
             * @brief Set the beta learning parameter
             * @param beta
             */
            void set_learning_parameter(double beta) {
                this->beta = beta;
            };

            virtual double get_learning_rate() const {
                return beta;
            }

        protected:
            double beta;
        };

        template<typename TState>
        class ArrayStateValueApproximator : public StateValueApproximator<TState> {
        public:

            ArrayStateValueApproximator(double beta) : StateValueApproximator<TState>(beta) {
            }

            /**
             * Get argmax_a Q(s,a), it may contains several indices
             * @param state
             * @return argmax_a Q(s,a)
             */
            std::vector <TState> argmax() const {
                double max = this->max();
                std::vector <TState> idxV;
                for (auto const &actionValue : this->_V) {
                    if (max == actionValue.second) {
                        idxV.push_back(actionValue.first);
                    }
                }
                return *select_randomly(idxV.begin(), idxV.end());
            }

            /**
             * Setter of the Q function
             * @param state
             * @param action
             * @param value
             */
            void V(TState state, double value) {
                this->_V[state] = this->V(state) + this->beta * (value - this->V(state));
            }

            /**
             * @brief Get max V(s)
             */
            double max() const {
                double max = std::numeric_limits<double>::lowest();
                for (auto const &actionValue : this->_V) {
                    if (max <= actionValue.second)
                        max = actionValue.second;
                }
                return max;
            }

            /**
             * Get the Expected Reward for a in s.
             * @param state
             * @param action
             * @return
             */
            double V(TState state) const {
                if (this->_V.find(state) == this->_V.end())
                    this->_V[state] = 0.0;
                return this->_V[state];
            }

            /**
             * @brief reset the function approximation
             */
            void reset() {
                this->_V.clear();
            }

            std::map<TState, double> _V;
        };
}}
#endif /* FUNCTIONAPPROXIMATOR_HPP */
