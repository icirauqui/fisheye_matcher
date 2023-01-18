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



#endif