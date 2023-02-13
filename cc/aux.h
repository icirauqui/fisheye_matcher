#ifndef AUX_H
#define AUX_H

#include <math.h>
#include <cmath>
#include <iostream>
#include <vector>




float RadToDeg(float rad);


float DegToRad(float deg);


std::ostream& bold_on(std::ostream& os);


std::ostream& bold_off(std::ostream& os);


int CountMaxIdx(std::vector<std::vector<double>> &v);




#endif