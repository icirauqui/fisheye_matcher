#ifndef AUX_H
#define AUX_H

#include <math.h>
#include <cmath>
#include <iostream>
#include <vector>




float RadToDeg(float rad) {
  return rad * 180.0 / M_PI;
}


float DegToRad(float deg) {
  return deg * M_PI / 180.0;
}


std::ostream& bold_on(std::ostream& os) {
    return os << "\e[1m";
}

std::ostream& bold_off(std::ostream& os) {
    return os << "\e[0m";
}


int CountMaxIdx(std::vector<std::vector<double>> &v) {
  int count_max = 0;
  int max_idx = 0;
  for (auto v1 : v) {
    int count = 0;
    for (auto d : v1) {
      if (d > 0.0) {
        count++;
      }
    }
    if (count > count_max) {
      count_max = count;
      max_idx = count;
    }
  }
  return max_idx;
}




#endif