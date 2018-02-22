/* 
 * File:   reward.hpp
 * Author: xetql
 *
 * Created on March 3, 2017, 3:17 PM
 */

#ifndef REWARD
#define REWARD

#include <functional>
#include <numeric>
#include <cassert>
#include <memory>

template<typename T>
class Reward{
public:
  virtual double get() {
    return value;
  }
  virtual void set_value(int action, T raw_values){
    this->action = action;
    this->value = apply(raw_values);
  }
protected:
  double value;
  int action;

  virtual double apply(T raw_value) = 0;
};

class DefaultReward: public Reward<double>{
public:
  DefaultReward(){}
protected:
  double apply(double raw_value){
    return raw_value;
  }
};

template<typename T>
class IReward : public Reward<T>{
public:
  IReward(std::function<double(T)> apply_function){
    this->apply_func = apply_function;
  }
  IReward(){}
  void set_function(std::function<double(T)> apply_function){
    this->apply_func = apply_function;
  }
private:
  std::function<double(T)> apply_func;
protected:
  double apply(T raw_values){
    return apply_func(raw_values);
  }
};

class CombinedWithUtilityFunctionReward: public Reward<std::vector<double>> {
public:
  CombinedWithUtilityFunctionReward(std::vector<double> combination_percentages) {
    comb_percentages = combination_percentages;
  }
  virtual void set(int action, std::vector<double> raw_values){
    assert(std::accumulate(comb_percentages.begin(), comb_percentages.end(), 0.0) == 100.0 && 
           raw_values.size() == comb_percentages.size());
    this->action = action;
    this->value = apply(raw_values);
  }
protected:
  std::vector<double> comb_percentages;
  double apply(std::vector<double> raw_values) {
    double result = 0.0;
    for(size_t i = 0; i < raw_values.size(); i++){
      result += raw_values[i] * comb_percentages[i];
    }
    return result;
  }
};
#endif  


