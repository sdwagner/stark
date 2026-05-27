#pragma once
#include "../../../core/Stark.h"
#include "../PointDynamics.h"
#include "../../types.h"

namespace stark
{
    class EnergyTriangleArea
    {
    public:
        /* Types */
        struct Params
        {
            STARK_PARAM_SCALE()
            STARK_PARAM_NON_NEGATIVE(double, stiffness, 1e9)
        };
        const spPointDynamics dyn;
        symx::LabelledConnectivity<5> conn_complete{ { "idx", "group", "i", "j", "k" } };
        // Input
        std::vector<double> scale;  // per group
        std::vector<double> stiffness;  // group
        std::vector<double> rest_area;
        struct Handler { STARK_COMMON_HANDLER_CONTENTS(EnergyTriangleArea, Params) };
        EnergyTriangleArea(Stark& stark, spPointDynamics dyn);
        Handler add(const PointSetHandler& set, const std::vector<std::array<int, 3>>& triangles, const Params& params);
        Handler add(const PointSetHandler& set, const std::vector<std::array<int, 3>>& triangles, const std::map<std::pair<int,int>, double>& stitched_vertices, const Params& params);
        Params get_params(const Handler& handler) const;
        void set_params(const Handler& handler, const Params& params);
    };
}