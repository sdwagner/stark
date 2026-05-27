#pragma once
#include <unordered_map>

#include "../../../core/Stark.h"
#include "../PointDynamics.h"
#include "../../types.h"


namespace stark
{
	/**
	*   Energy definition and data structure for the penalty based point prescribed positions used as boundary conditions.
	*   It uses an additional "tolerance" parameter to define the maximum distance between the prescribed and the actual position.
	*   If the violation is larger than the tolerance, the penalty bending_stiffness is increased.
	*/
	class EnergyPrescribedPositionsTetBarycentric
	{
	public:
		/* Types */
		struct Params 
		{ 
			STARK_PARAM_STIFFNESS() 
			STARK_PARAM_TOLERANCE() 
		};
		struct Handler
		{
			STARK_COMMON_HANDLER_CONTENTS(EnergyPrescribedPositionsTetBarycentric, Params)
			inline void set_target_position(int prescribed_idx, const Eigen::Vector3d& t)
			{
				this->get_model()->set_target_position(*this, prescribed_idx, t);
			}
			inline void set_stiffness(int idx, double stiffness)
			{
				this->get_model()->set_stiffness(*this, idx, stiffness);
			}

		};

	private:
		/* Fields */
		const spPointDynamics dyn;

		std::vector<Eigen::Vector3d> target_positions; // per point
		std::vector<double> stiffness; // per point
		std::vector<double> tolerance; // per group

		std::vector<std::array<int, 2>> group_begin_end; // per group  (for tolerance checking purposes)
		symx::LabelledConnectivity<6> conn{ {"idx", "group", "tet0", "tet1", "tet2", "tet3"} };
		std::vector<std::array<double, 4>> barycentric_coords;  // per index

	public:
		/* Methods */
		EnergyPrescribedPositionsTetBarycentric(Stark& stark, spPointDynamics dyn);
		Handler add(const PointSetHandler& set, const std::vector<std::array<int, 4>>& tets, const std::vector<std::array<double, 4>>& bary, const Params& params);
		Params get_params(const Handler& handler) const;
		void set_params(const Handler& handler, const Params& params);
		void set_stiffness(const Handler& handler, int prescribed_idx, double stiffness);
		void set_target_position(const Handler& handler, int prescribed_idx, const Eigen::Vector3d& t);
		void _write_diff(Stark& stark);

	private:
		bool _is_converged_state_valid(Stark& stark);
	};
}

