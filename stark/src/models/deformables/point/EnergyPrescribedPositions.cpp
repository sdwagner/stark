#include "EnergyPrescribedPositions.h"

#include "../../time_integration.h"
#include "../../../utils/include.h"

using namespace stark;
using namespace symx;

EnergyPrescribedPositions::EnergyPrescribedPositions(Stark& stark, spPointDynamics dyn)
	: dyn(dyn)
{
	// Callbacks
	stark.callbacks->newton->add_is_converged_state_valid([&]() { return this->_is_converged_state_valid(stark); });
	stark.callbacks->add_write_frame([&]() {_write_diff(stark);});


	// Declare the energy
	stark.global_potential->add_potential("EnergyPrescribedPositions", this->conn,
		[&](MappedWorkspace<double>& mws, Element& node)
		{
			// Create symbols
			Vector v1 = mws.make_vector(this->dyn->v1.data, node["point"]);
			Vector x0 = mws.make_vector(this->dyn->x0.data, node["point"]);
			Vector x1_prescribed = mws.make_vector(this->target_positions, node["idx"]);
			Scalar k = mws.make_scalar(this->stiffness, node["group"]);
			Scalar dt = mws.make_scalar(stark.dt);

			// Time integration
			Vector x1 = time_integration(x0, v1, dt);

			// Energy
			Scalar E = 0.5 * k * (x1 - x1_prescribed).squared_norm();
			return E;
		}
	);
	stark.global_potential->add_potential("EnergyPrescribedPositionsAnisotropic", this->conn_aniso,
		[&](MappedWorkspace<double>& mws, Element& node)
		{
			// Create symbols
			Vector x0 = mws.make_vector(this->dyn->x0.data, node["point"]);
			Vector x1_prescribed = mws.make_vector(this->target_positions_aniso, node["idx"]);
			Vector n = mws.make_vector(this->normal_aniso, node["idx"]);
			Scalar k_n = mws.make_scalar(this->stiffness_aniso_normal, node["idx"]);
			Scalar k_t = mws.make_scalar(this->stiffness_aniso_tangent, node["idx"]);
			Scalar dt = mws.make_scalar(stark.dt);

			Vector v1 = mws.make_vector(this->dyn->v1.data, node["point"]);
			Vector x1 = time_integration(x0, v1, dt);
			Vector e = x1 - x1_prescribed;
			Vector e_n = n.dot(e) * n;
			Vector e_t = e - e_n;
			// Energy
			Scalar E = 0.5 * k_n * e_n.squared_norm() + 0.5 * k_t * e_t.squared_norm();
			return E;
		}
	);

}
EnergyPrescribedPositions::Handler EnergyPrescribedPositions::add(const PointSetHandler& set, const std::vector<int>& points, const Params& params)
{
	set.exit_if_not_valid("EnergyPrescribedPositions::add");
	const int group = (int)this->tolerance.size();
	this->tolerance.push_back(params.tolerance);

	const int begin = (int)this->target_positions.size();
	for (int i = 0; i < (int)points.size(); i++) {
		const int glob_idx = set.get_global_index(points[i]);
		this->target_positions.push_back(this->dyn->x1[glob_idx]);
		this->rest_positions.push_back(this->dyn->x1[glob_idx]);
		this->conn.numbered_push_back({ glob_idx, group });
		this->stiffness.push_back(params.stiffness);
	}
	const int end = (int)this->target_positions.size();
	this->group_begin_end.push_back({ begin, end });

	return Handler(this, group);
}
EnergyPrescribedPositions::Handler EnergyPrescribedPositions::add_aniso(const PointSetHandler& set, const std::vector<int>& points, const AnisoParams& params)
{
	set.exit_if_not_valid("EnergyPrescribedPositions::add");
	const int group = (int)this->tolerance_aniso.size();
	this->tolerance_aniso.push_back(params.tolerance);

	const int begin = (int)this->target_positions_aniso.size();
	for (int i = 0; i < (int)points.size(); i++) {
		const int glob_idx = set.get_global_index(points[i]);
		this->target_positions_aniso.push_back(this->dyn->x1[glob_idx]);
		this->normal_aniso.emplace_back(0, 0, 1);
		this->conn_aniso.numbered_push_back({ glob_idx, group });
		this->stiffness_aniso_normal.push_back(params.stiffness_normal);
		this->stiffness_aniso_tangent.push_back(params.stiffness_tangent);
	}
	const int end = (int)this->target_positions_aniso.size();
	this->group_begin_end_aniso.push_back({ begin, end });

	return Handler(this, group);
}

EnergyPrescribedPositions::Handler EnergyPrescribedPositions::add_inside_aabb(const PointSetHandler& set, const Eigen::Vector3d& aabb_center, const Eigen::Vector3d& aabb_dim, const Params& params)
{
	set.exit_if_not_valid("EnergyPrescribedPositions::add_inside_aabb");
	Eigen::AlignedBox3d aabb(aabb_center - 0.5*aabb_dim, aabb_center + 0.5*aabb_dim);

	std::vector<int> points;
	for (int i = 0; i < set.size(); i++) {
		if (aabb.contains(set.get_position(i))) {
			points.push_back(i);
		}
	}
	return this->add(set, points, params);
}
EnergyPrescribedPositions::Handler EnergyPrescribedPositions::add_outside_aabb(const PointSetHandler& set, const Eigen::Vector3d& aabb_center, const Eigen::Vector3d& aabb_dim, const Params& params)
{
	set.exit_if_not_valid("EnergyPrescribedPositions::add_outside_aabb");
	Eigen::AlignedBox3d aabb(aabb_center - 0.5*aabb_dim, aabb_center + 0.5*aabb_dim);

	std::vector<int> points;
	for (int i = 0; i < set.size(); i++) {
		if (!aabb.contains(set.get_position(i))) {
			points.push_back(i);
		}
	}
	return this->add(set, points, params);
}
EnergyPrescribedPositions::Params EnergyPrescribedPositions::get_params(const Handler& handler) const
{
	handler.exit_if_not_valid("EnergyPrescribedPositions::get_params");

	const int group = handler.get_idx();
	if (group < 0 || group >= (int)this->stiffness.size()) {
		std::cout << "Stark error: EnergyPrescribedPositions::get_params() found an invalid index." << std::endl;
		exit(-1);
	}

	Params params;
	params.stiffness = this->stiffness[group];
	params.tolerance = this->tolerance[group];
	return params;
}
void EnergyPrescribedPositions::set_params(const Handler& handler, const Params& params)
{
	handler.exit_if_not_valid("EnergyPrescribedPositions::set_params");

	const int group = handler.get_idx();
	if (group < 0 || group >= (int)this->stiffness.size()) {
		std::cout << "Stark error: EnergyPrescribedPositions::set_params() found an invalid index." << std::endl;
		exit(-1);
	}

	this->stiffness[group] = params.stiffness;
	this->tolerance[group] = params.tolerance;
}
void EnergyPrescribedPositions::set_transformation(const Handler& handler, const Eigen::Vector3d& t, const Eigen::Matrix3d& R)
{
	handler.exit_if_not_valid("EnergyPrescribedPositions::set_transformation");
	const int group = handler.get_idx();
	const auto [begin, end] = this->group_begin_end[group];
	for (int i = begin; i < end; i++) {
		this->target_positions[i] = R * this->rest_positions[i] + t;
	}
}

void EnergyPrescribedPositions::set_transformation(const Handler& handler, const Eigen::Vector3d& t, const double angle_deg, const Eigen::Vector3d& axis)
{
	const Eigen::Matrix3d R = Eigen::AngleAxisd(deg2rad(angle_deg), axis.normalized()).toRotationMatrix();
	this->set_transformation(handler, t, R);
}

void EnergyPrescribedPositions::set_stiffness(const Handler& handler, int prescribed_idx, double stiffness)
{
	handler.exit_if_not_valid("EnergyPrescribedPositionsTetBarycentric::set_stiffness");
	const int group = handler.get_idx();
	const auto [begin, end] = this->group_begin_end[group];
	const int idx = begin + prescribed_idx;
	this->stiffness[idx] = stiffness;
}

void EnergyPrescribedPositions::set_target_position(const Handler& handler, int prescribed_idx, const Eigen::Vector3d& t)
{
	handler.exit_if_not_valid("EnergyPrescribedPositions::set_target_position");
	const int group = handler.get_idx();
	const auto [begin, end] = this->group_begin_end[group];
	const int idx = begin + prescribed_idx;
	this->target_positions[idx] = t;
}

void EnergyPrescribedPositions::set_stiffness_aniso_normal(const Handler& handler, int prescribed_idx, double stiffness)
{
	handler.exit_if_not_valid("EnergyPrescribedPositionsTetBarycentric::set_stiffness_aniso_normal");
	const int group = handler.get_idx();
	const auto [begin, end] = this->group_begin_end_aniso[group];
	const int idx = begin + prescribed_idx;
	this->stiffness_aniso_normal[idx] = stiffness;
}
void EnergyPrescribedPositions::set_stiffness_aniso_tangent(const Handler& handler, int prescribed_idx, double stiffness)
{
	handler.exit_if_not_valid("EnergyPrescribedPositionsTetBarycentric::set_stiffness_aniso_tangent");
	const int group = handler.get_idx();
	const auto [begin, end] = this->group_begin_end_aniso[group];
	const int idx = begin + prescribed_idx;
	this->stiffness_aniso_tangent[idx] = stiffness;
}
void EnergyPrescribedPositions::set_target_position_aniso(const Handler& handler, int prescribed_idx, const Eigen::Vector3d& t)
{
	handler.exit_if_not_valid("EnergyPrescribedPositions::set_target_position_aniso");
	const int group = handler.get_idx();
	const auto [begin, end] = this->group_begin_end_aniso[group];
	const int idx = begin + prescribed_idx;
	this->target_positions_aniso[idx] = t;
}
void EnergyPrescribedPositions::set_normal_aniso(const Handler& handler, int prescribed_idx, const Eigen::Vector3d& t)
{
	handler.exit_if_not_valid("EnergyPrescribedPositions::set_normal_aniso");
	const int group = handler.get_idx();
	const auto [begin, end] = this->group_begin_end_aniso[group];
	const int idx = begin + prescribed_idx;
	this->normal_aniso[idx] = t;
}
void EnergyPrescribedPositions::_write_diff(Stark& stark)
{
	if (conn.size() > 0)
	{
		double att_diff = 0.0;
		for (int i = 0; i < (int)this->conn.size(); i++) {
			auto [idx, point_idx, group] = this->conn[i];
			const Eigen::Vector3d& x1 = this->dyn->get_x1(point_idx, stark.dt);
			const Eigen::Vector3d& x1_prescribed = this->target_positions[idx];
			att_diff += (x1 - x1_prescribed).squaredNorm();
		}
		stark.context->logger->append("att_diff", att_diff/this->conn.size());
	}
	if (conn_aniso.size() > 0)
	{
		double att_diff = 0.0;
		for (int i = 0; i < (int)this->conn_aniso.size(); i++) {
			auto [idx, point_idx, group] = this->conn_aniso[i];
			const Eigen::Vector3d& x1 = this->dyn->get_x1(point_idx, stark.dt);
			const Eigen::Vector3d& x1_prescribed = this->target_positions_aniso[idx];
			att_diff += (x1 - x1_prescribed).squaredNorm();
		}
		stark.context->logger->append("att_diff", att_diff/this->conn_aniso.size());
	}
}


bool EnergyPrescribedPositions::_is_converged_state_valid(Stark& stark)
{
	const double dt = stark.dt;
	bool is_valid = true;

	auto label_idx = this->conn.get_label_indices({ "idx", "point", "group" });
	for (int i = 0; i < (int)this->conn.size(); i++) {
		auto [idx, point_idx, group] = this->conn.get(i, label_idx);
		const Eigen::Vector3d& x1 = this->dyn->get_x1(point_idx, dt);
		const Eigen::Vector3d& x1_prescribed = this->target_positions[idx];
		const double sq_distance = (x1 - x1_prescribed).squaredNorm();
		const double tol = this->tolerance[group];
		if (sq_distance > tol*tol) {
			is_valid = false;
			this->stiffness[group] *= 2.0;
			break;
		}
	}

	if (!is_valid) {
		stark.context->output->print("Deformable prescribed position constraints are not within tolerance. Stiffness hardened.\n", symx::Verbosity::Summary);
	}

	return is_valid;
}
