#pragma once
#include "../../../core/Stark.h"
#include "../PointDynamics.h"
#include "../../types.h"

namespace stark
{
	class EnergyTriangleStretching
	{
	public:
		/* Types */
		struct Params 
		{
			STARK_PARAM_SCALE()
			STARK_PARAM_NON_NEGATIVE(double, stiffness, 1e9)
		};
		struct Handler { STARK_COMMON_HANDLER_CONTENTS(EnergyTriangleStretching, Params) };

	private:
		/* Fields */
		const spPointDynamics dyn;
		symx::LabelledConnectivity<4> conn_complete{ { "idx", "group", "i", "j" } };

		// Input
		std::vector<double> scale;  // per group
		std::vector<double> stiffness;  // group
		std::vector<double> rest_length;
		
	public:
		/* Methods */
		EnergyTriangleStretching(Stark& stark, spPointDynamics dyn);
		Handler add(const PointSetHandler& set, const std::vector<std::array<int, 3>>& triangles, const Params& params);
		Handler add(const PointSetHandler& set, const std::vector<std::array<int, 3>>& triangles, const std::map<std::pair<int,int>, double>& stitched_vertices, const Params& params);
		Params get_params(const Handler& handler) const;
		void set_params(const Handler& handler, const Params& params);
	};
}
