#ifndef POLICY_HPP
#define POLICY_HPP
#include <unordered_map>
#include <string>
#include <limits>
#include <queue>
#include "../approximators/FunctionApproximator.hpp"

namespace librl { namespace policy {

        /** Action Selection Policy **/
        template<typename TState, typename TAction>
        class Policy {
        public:
            virtual std::unordered_map<TAction, double>
            get_probabilities(const librl::approximator::ActionValueApproximator<TState, TAction>* f,
                              const std::vector<TAction> &available_actions,
                              const TState &at_state) const = 0;

            virtual TAction choose_action(const librl::approximator::ActionValueApproximator<TState, TAction>* f,
                                          const std::vector<TAction> &available_actions,
                                          const TState &at_state) = 0;

            virtual std::string getName() const = 0;

            virtual void reset() = 0;

            virtual void update() {};

            virtual std::string to_string() { return ""; };
        };
}}
#endif // POLICY_HPP
