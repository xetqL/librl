//
// Created by xetql on 22.02.18.
//

#ifndef PROJECT_LAYEREDPOLICY_HPP
#define PROJECT_LAYEREDPOLICY_HPP
#include <queue>
#include <utility>
#include <memory>

#include "Boltzmann.hpp"
#include "EGreedy.hpp"
#include "Greedy.hpp"
#include "UCB1.hpp"

namespace librl { namespace policy {

    template<typename TState, typename TAction>
    class Layered : public Policy<TState, TAction> {

        using PtrLayerTypes = std::shared_ptr<Policy<TState, TAction>>;
        std::deque< std::pair<PtrLayerTypes, unsigned long int> > layers;

    public:
        template <template<class _TState, class _TAction> class LayerType, unsigned long int duration, class... Args>
        void add(Args... args) { layers.push_back(std::make_pair(std::make_shared< LayerType<TState, TAction> >(args...), duration)); }

        template <template<class _TState, class _TAction> class LayerType, class... Args> //by default a layer is executed infinitely
        void add(Args... args) { layers.push_back(std::make_pair(std::make_shared< LayerType<TState, TAction> >(args...), std::numeric_limits<unsigned long int>::max())); }

        virtual TAction choose_action(const librl::approximator::ActionValueApproximator<TState, TAction>* f,
                                      const std::vector<TAction> &available_actions,
                                      const TState &at_state) {
            auto policy = layers.front().first;

            auto action = policy->choose_action(f, available_actions, at_state);

            if(layers.size() > 1) layers.front().second--;

            if(layers.front().second == 0 && layers.size() > 1)
                layers.pop_front();

            return action;
        }

        std::string getName() const { return layers.front().first->getName(); }

        void reset() {}

        virtual std::unordered_map<TAction, double> get_probabilities(const librl::approximator::ActionValueApproximator<TState, TAction>* f,
                                                                      const std::vector<TAction> &available_actions,
                                                                      const TState &at_state) const {
            auto policy = layers.front().first;
            return policy->get_probabilities(f, available_actions, at_state);
        }
    };
}}
#endif //PROJECT_LAYEREDPOLICY_HPP
