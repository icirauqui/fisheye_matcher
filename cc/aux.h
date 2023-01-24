#ifndef AUX_H
#define AUX_H

#include <math.h>
#include <cmath>




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





#endif