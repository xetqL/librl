#ifndef POLICY_HPP
#define POLICY_HPP
#include <map>
#include <string>
#include <limits>
#include "RLAgent.hpp"

template<typename TState, typename TAction>
class Policy
{
public:
  //TODO: choose action does not depend on the agent but on the approximated function
  virtual std::map<TAction, double> get_probabilities(const RLAgent<TState, TAction>* agent, TState state) const  = 0;
  //TODO: choose action does not depend on the agent but on the approximated function
  virtual TAction choose_action(const RLAgent<TState, TAction>* agent) const = 0;
  virtual std::string getName() const = 0;
  virtual void reset() = 0;
  virtual void update(){};
  virtual std::string to_string(){ return "";};
};

#endif // POLICY_HPP
