#ifndef TOOLS
#define TOOLS

#include <cmath>

double to_radians(double degrees) { return (degrees * (M_PI / 180.0F)); }

double to_degrees(double radians) { return (radians * (180.0F / M_PI)); }

#endif