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
	class EnergyPrescribedPositions
	{
	public:
		/* Types */
		struct Params 
		{ 
			STARK_PARAM_STIFFNESS() 
			STARK_PARAM_TOLERANCE() 
		};
		struct AnisoParams
		{
			STARK_PARAM_NON_NEGATIVE(double, stiffness_normal, 1e5)
			STARK_PARAM_NON_NEGATIVE(double, stiffness_tangent, 1e3)
			STARK_PARAM_TOLERANCE()
		};
		struct Handler
		{
			STARK_COMMON_HANDLER_CONTENTS(EnergyPrescribedPositions, Params)
			inline void set_transformation(const Eigen::Vector3d& t, const Eigen::Matrix3d& R = Eigen::Matrix3d::Identity())
			{
				this->get_model()->set_transformation(*this, t, R);
			}
			inline void set_transformation(const Eigen::Vector3d& t, const double angle_deg, const Eigen::Vector3d& axis)
			{
				this->get_model()->set_transformation(*this, t, angle_deg, axis);
			}
			inline void set_target_position(int prescribed_idx, const Eigen::Vector3d& t)
			{
				this->get_model()->set_target_position(*this, prescribed_idx, t);
			}
			inline void set_stiffness(int idx, double stiffness)
			{
				this->get_model()->set_stiffness(*this, idx, stiffness);
			}
			inline void set_target_position_aniso(int prescribed_idx, const Eigen::Vector3d& t)
			{
				this->get_model()->set_target_position_aniso(*this, prescribed_idx, t);
			}
			inline void set_normal_aniso(int prescribed_idx, const Eigen::Vector3d& t)
			{
				this->get_model()->set_normal_aniso(*this, prescribed_idx, t);
			}
			inline void set_stiffness_aniso_normal(int idx, double stiffness)
			{
				this->get_model()->set_stiffness_aniso_normal(*this, idx, stiffness);
			}
			inline void set_stiffness_aniso_tangent(int idx, double stiffness)
			{
				this->get_model()->set_stiffness_aniso_tangent(*this, idx, stiffness);
			}

		};

	private:
		/* Fields */
		const spPointDynamics dyn;
		symx::LabelledConnectivity<3> conn{ { "idx", "point", "group" }};

		std::vector<Eigen::Vector3d> target_positions; // per point
		std::vector<double> stiffness; // per group
		std::vector<double> tolerance; // per group

		std::vector<std::array<int, 2>> group_begin_end; // per group  (for tolerance checking purposes)
		std::vector<Eigen::Vector3d> rest_positions; // per point  (for transformation purposes)

		symx::LabelledConnectivity<3> conn_aniso{ { "idx", "point", "group" }};
		std::vector<Eigen::Vector3d> target_positions_aniso; // per point
		std::vector<double> stiffness_aniso_normal; // per group
		std::vector<double> stiffness_aniso_tangent; // per group
		std::vector<double> tolerance_aniso; // per group
		std::vector<Eigen::Vector3d> normal_aniso;
		std::vector<std::array<int, 2>> group_begin_end_aniso; // per group  (for tolerance checking purposes)


	public:
		/* Methods */
		EnergyPrescribedPositions(Stark& stark, spPointDynamics dyn);
		Handler add(const PointSetHandler& set, const std::vector<int>& points, const Params& params);
		Handler add_aniso(const PointSetHandler& set, const std::vector<int>& points, const AnisoParams& params);
		Handler add_inside_aabb(const PointSetHandler& set, const Eigen::Vector3d& aabb_center, const Eigen::Vector3d& aabb_dim, const Params& params);
		Handler add_outside_aabb(const PointSetHandler& set, const Eigen::Vector3d& aabb_center, const Eigen::Vector3d& aabb_dim, const Params& params);
		Params get_params(const Handler& handler) const;
		void set_params(const Handler& handler, const Params& params);
		void set_transformation(const Handler& handler, const Eigen::Vector3d& t, const Eigen::Matrix3d& R);
		void set_transformation(const Handler& handler, const Eigen::Vector3d& t, const double angle_deg, const Eigen::Vector3d& axis);
		void set_stiffness(const Handler& handler, int prescribed_idx, double stiffness);
		void set_target_position(const Handler& handler, int prescribed_idx, const Eigen::Vector3d& t);
		void set_stiffness_aniso_normal(const Handler& handler, int prescribed_idx, double stiffness);
		void set_stiffness_aniso_tangent(const Handler& handler, int prescribed_idx, double stiffness);
		void set_target_position_aniso(const Handler& handler, int prescribed_idx, const Eigen::Vector3d& t);
		void set_normal_aniso(const Handler& handler, int prescribed_idx, const Eigen::Vector3d& t);
		void _write_diff(Stark& stark);

	private:
		bool _is_converged_state_valid(Stark& stark);
	};
}

