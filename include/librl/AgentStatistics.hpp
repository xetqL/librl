#ifndef AGENTSTAT_HPP
#define AGENTSTAT_HPP
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <functional>
#include <vector>
#include <cassert>
#include <iostream>

class AgentStatistics {
  public:
    unsigned int numberOfStepsUntilNow   = 0, numberOfActions, numberOfStates;
    double meanReward           = 0.0, 
           varianceReward       = 0.0, 
           sqrReward            = 0.0,
           currentTemp          = 100000;
    std::vector<std::vector<double>> meanRewardForAction,
      varianceRewardForAction,
      sqrRewardForAction;
    std::vector<std::vector<int>> numberOfTimeAction;

    void update(int state, int action, double reward){
      numberOfStepsUntilNow++;  //increment number of global action done
      numberOfTimeAction[state][action]++; //increment number of time action A has been done

      // global reward mean
      meanReward = nextMean(meanReward, reward, numberOfStepsUntilNow);
      meanRewardForAction[state][action] = nextMean(meanRewardForAction[state][action], reward, numberOfTimeAction[state][action]);    

      sqrReward += reward * reward;
      sqrRewardForAction[state][action] += reward * reward;

      // variance with th. König-Huygens
      varianceReward = std::fabs((sqrReward / (double) numberOfStepsUntilNow) - (meanReward * meanReward));
      varianceRewardForAction[state][action] = std::fabs((sqrRewardForAction[state][action] / (double) numberOfTimeAction[state][action]) - (meanRewardForAction[state][action] * meanRewardForAction[state][action]));

      // meanRewardForEachAS[state][action] = nextMean(meanRewardForEachAS[state][action], reward, numberOfStepsUntilNow);
      assert(varianceReward >= 0);
    }

    AgentStatistics(unsigned int numberOfStates, unsigned int numberOfActions) {

      numberOfTimeAction.resize(numberOfStates, std::vector<int> ( numberOfActions, 0 ));

      meanRewardForAction.resize(numberOfStates, std::vector<double> ( numberOfActions, 0 ));


      sqrRewardForAction.resize(numberOfStates, std::vector<double> ( numberOfActions, 0 ));

      varianceRewardForAction.resize(numberOfStates, std::vector<double> ( numberOfActions, 0 ));

      this->numberOfActions = numberOfActions;
      this->numberOfStates  = numberOfStates;
    }

    ~AgentStatistics(){}
  private:
    double nextMean(double previousMean, double nextValue, int step) {
      return (double)((previousMean * (step-1)) + nextValue) / (double)step;
    }
};
#endif
