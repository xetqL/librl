#pragma once

#include <vector>
#include <iostream>
#include <map>

template<class T>
void print2DArray(std::vector<std::vector<T>> array){
  for(unsigned int i = 0; i < array.size(); i++) {
    for(unsigned int j = 0; j < array[0].size(); j++){
      std::cout << "Array[" <<i<<"]["<<j<<"] = "<<array[i][j]<<std::endl;
    }
  }
}

template<class K, class VK, class VV>
void print2DMap(std::map<K, std::map<VK, VV>> array){
  for(unsigned int i = 0; i < array.size(); i++) {
    for(unsigned int j = 0; j < array[0].size(); j++){
      std::cout << "Array[" <<i<<"]["<<j<<"] = "<<array[i][j]<<std::endl;
    }
  }
}

template<class T> 
void print1DArray(std::vector<T> array){
  for(int i = 0; i < array.size(); i++) {
      std::cout << "Array[" <<i<<"] = "<<array[i]<<std::endl;
  }
}

