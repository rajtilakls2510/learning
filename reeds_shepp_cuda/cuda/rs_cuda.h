#ifndef RS_CUDA_H
#define RS_CUDA_H

#include <stdio.h>
#include <math.h>
#include <iostream>

namespace RS_CUDA {

    constexpr double EPS1 =  1.0e-12;
    constexpr double EPS2 =  1.0e-12;
    constexpr double EPS3 =  1.0e-12;
    constexpr double EPS4 =  1.0e-12;
    constexpr double INFINITY_VAL = 10000.0;

    constexpr double MPI =  3.1415926536;
    constexpr double MPIMUL2 =  6.2831853072;
    constexpr double MPIDIV2 =  1.5707963268;


    /**
     * @brief Point class to store (x,y,t)
     */
    struct Point {
        double x, y, t;
    };

    /**
     * @brief Stores a reeds-shepp curve in an analytical format (not in discretized form)
     */
    struct AnalyticCurve {
        int n;  // Maneuver number (1-48)
        double t, u, v; // Amount of arc lengths
        double rad_curv; // Radius of curvatures of turns
    };

    // /**
    //  * @brief Calculates the minimum length RS curve from a start to a goal given a radius of curvature
    //  * @param start The start point of the curve
    //  * @param goal The goal point of the curve
    //  * @param ac The calculated curve and its parameters
    //  * @param rad_curv Radius of curvatures of turns
    //  * @return The length of the curve (includes arc lengths)   
    //  */
    // double min_length_rs (
    //     Point start,
    //     Point goal,
    //     AnalyticCurve* ac,
    //     double rad_curv
    // );


    // /**
    //  * @brief Densifies an analytical curve (calculated by min_length_rs() method) and stores the path
    //  * @param ac The curve in its analytical form
    //  * @param step The discretization resolution 
    //  * @param path The path to store (the size can be calculated using the length returned by min_length_rs and step)
    //  * @return The number of points in the path
    //  */
    // int construct_rs(
    //     AnalyticCurve ac,
    //     double step,
    //     Point* path
    // );



}   // namespace RS_CUDA

#endif