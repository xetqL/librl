#ifndef GREEDYPOLICY_HPP
#define GREEDYPOLICY_HPP
#include "Policy.hpp"
#include "../utils/util.hpp"
#include <algorithm>
namespace librl { namespace policy {

        template<typename TState, typename TAction>
        class InitPolicy {
        protected:
            int repeat;
        public:
            InitPolicy(int repeat=1) : repeat(repeat){}
            virtual inline void reset() = 0;
            virtual inline bool released(const TState& state) const = 0;
            virtual void choose_action(const TState& state, const std::vector<TAction>& a, TAction &next_action) = 0;
        };

        template<typename TState, typename TAction>
        class SoftRoundRobin : public InitPolicy <TState, TAction>{
            int action_index = 0;
            bool release_state = false;
        public:
            SoftRoundRobin(int repeat) : InitPolicy<TState, TAction>(repeat){}
            inline void reset(){ action_index = 0; release_state = false; }
            inline bool released(const TState& state) const { return release_state; }
            void choose_action(const TState& state, const std::vector<TAction>& a, TAction &next_action) {
                next_action = a.at(action_index);
                action_index = action_index + 1;
                release_state = a.size() == action_index;
                return;
            }
        };

        template<typename TState, typename TAction>
        class FullRoundRobin : public InitPolicy <TState, TAction>{
            //count how many actions have been tried in a given state
            std::unordered_map<TState, int> action_indices;
            //release only if we are in a given state otherwise continue round robin
            std::unordered_map<TState, bool> release_states;
        public:
            FullRoundRobin(int repeat) : InitPolicy<TState, TAction>(repeat){}
            inline void reset() { action_indices.clear(); release_states.clear(); }
            inline bool released(const TState& state) const {
                return release_states.find(state) != release_states.end() && release_states.at(state);
            }
            void choose_action(const TState& state, const std::vector<TAction>& a, TAction &next_action) {
                if(release_states.find(state) == release_states.end()) release_states[state] = false;
                if(action_indices.find(state) == action_indices.end()) action_indices[state] = 0;
                next_action = a.at(action_indices.at(state) % a.size());
                action_indices[state] = action_indices.at(state) + 1;
                release_states[state] = a.size()*this->repeat == action_indices.at(state);
                return;
            }
        };

        template<typename TState, typename TAction>
        class NothingToDo : public InitPolicy<TState, TAction> {
        public:
            NothingToDo(int repeat) : InitPolicy<TState, TAction>(repeat){}
            inline void reset() {};
            inline bool released(const TState& state) const { return true; }
            void choose_action(const TState& state, const std::vector<TAction>& a, TAction &next_action) {}
        };

        template<
                typename TState,
                typename TAction,
                template <typename TTState, typename TTAction> class _InitPolicy = NothingToDo,
                int HowManyRepeat = 1
        >
        class GreedyPolicy : public Policy<TState, TAction> {
            _InitPolicy<TState, TAction> init_pol{_InitPolicy<TState, TAction>(HowManyRepeat)};
        public:
            virtual TAction choose_action(const librl::approximator::ActionValueApproximator<TState, TAction>* f,
                                          const std::vector<TAction> &available_actions,
                                          const TState &at_state) {
                if(init_pol.released(at_state))
                    return greedyExploration(f, available_actions, at_state);

                TAction next;
                init_pol.choose_action(at_state, available_actions, next);
                return next;
            }

            virtual std::map<TAction, double> get_probabilities(const librl::approximator::ActionValueApproximator<TState, TAction>* f,
                                                                const std::vector<TAction> &available_actions,
                                                                const TState &at_state) const {
                double max = f->max(at_state);
                std::map<TAction, double> probabilities;
                int number_of_optimal_action = 0;
                for (auto const &action : available_actions)
                    if (f->Q(at_state, action) == max) number_of_optimal_action++;
                for (auto const &action : available_actions)
                    probabilities[action] = f->Q(at_state, action) == max ? 1.0 / number_of_optimal_action : 0.0;
                return probabilities;
            }

            std::string getName() const {
                return "Greedy Exploration";
            }

            void reset() {}

            /**
             * @brief Always take the best estimated reward
             * @param agent
             * @return best ID
             */
            TAction greedyExploration(const librl::approximator::ActionValueApproximator<TState, TAction>* f,
                                      const std::vector<TAction> &available_actions,
                                      const TState &at_state) const {
                //get either the best indice of a random one among the action space
                auto action = f->argmax(at_state, available_actions);
                return action;
            }
        };
    }}
#endif // GREEDYPOLICY_HPP
