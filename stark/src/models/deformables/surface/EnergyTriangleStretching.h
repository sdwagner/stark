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
		struct Handler {
			STARK_COMMON_HANDLER_CONTENTS(EnergyTriangleStretching, Params)
			inline void update_rest_lengths(const std::vector<Eigen::Vector3d>& vertices)
			{
				this->get_model()->update_rest_lengths(*this, vertices);
			}
		};

	private:
		/* Fields */
		const spPointDynamics dyn;
		symx::LabelledConnectivity<4> conn_complete{ { "idx", "group", "i", "j" } };

		// Input
		std::vector<double> scale;  // per group
		std::vector<double> stiffness;  // group
		std::vector<double> rest_length;
		// Local edges are retained so a kinematic reference can be updated at runtime.
		std::vector<std::vector<std::array<int, 2>>> group_edges;
		
	public:
		/* Methods */
		EnergyTriangleStretching(Stark& stark, spPointDynamics dyn);
		Handler add(const PointSetHandler& set, const std::vector<std::array<int, 3>>& triangles, const Params& params);
		Handler add(const PointSetHandler& set, const std::vector<std::array<int, 2>>& edges, const Params& params);
		Handler add(const PointSetHandler& set, const std::vector<std::array<int, 3>>& triangles, const std::map<std::pair<int,int>, double>& stitched_vertices, const Params& params);
		Params get_params(const Handler& handler) const;
		void set_params(const Handler& handler, const Params& params);
		void update_rest_lengths(const Handler& handler, const std::vector<Eigen::Vector3d>& vertices);
	};
}
