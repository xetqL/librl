#ifndef UTIL_HPP
#define UTIL_HPP

#include <boost/uuid/uuid.hpp>            // uuid class
#include <boost/uuid/uuid_generators.hpp> // generators
#include <boost/uuid/uuid_io.hpp>         // streaming operators etc.
#include <boost/lexical_cast.hpp>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

std::string generate_random_uuid() {
  boost::uuids::random_generator gen;
  boost::uuids::uuid u = gen();
  return boost::lexical_cast<std::string>(u);
}

template<typename Iter, typename RandomGenerator>
Iter select_randomly(Iter start, Iter end, RandomGenerator& g) {
    std::uniform_int_distribution<> dis(0, std::distance(start, end) - 1);
    std::advance(start, dis(g));
    return start;
}

template<typename T>
T select_randomly(T start, T end) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return select_randomly(start, end, gen);
}

template<typename T, typename U>
std::ostream& operator<<(std::ostream &strm, const std::pair<T,U> &a) {
    return strm << a.first << ';' << a.second << std::endl;
}
template<typename T>
std::ostream& operator<<(std::ostream &strm, const std::vector<T> &a) {
    for (auto const &mv : a)
        strm << mv << ';';
    return strm << std::endl;
}

double plus(double a, double b) { return a + b; }

double mean(double a, double b) { return (a + b) / 2.0; }

void rescale_mul(std::vector<double>& v, double mul){
  const double min = *std::min_element(v.begin(),v.end());
  const double max = *std::max_element(v.begin(),v.end());
  const int size = v.size();
  for(int i = 0; i < size; i++){
    v[i] = ((v[i] - min) / (max - min)) * mul;
  }
}

void rescale_add(std::vector<double>& v, double add){
  const double min = *std::min_element(v.begin(), v.end());
  const double max = *std::max_element(v.begin(), v.end());
  const int size = v.size();
  for(int i = 0; i < size; i++){
    v[i] = ((v[i] - min) / (max - min)) + add;
  }
}

#endif
