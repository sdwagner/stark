#include "EnergyPrescribedPositionsTetBarycentric.h"

#include "../../time_integration.h"
#include "../../../utils/include.h"


stark::EnergyPrescribedPositionsTetBarycentric::EnergyPrescribedPositionsTetBarycentric(Stark& stark, spPointDynamics dyn)
	: dyn(dyn)
{
	// Callbacks
	stark.callbacks->newton->add_is_converged_state_valid([&]() { return this->_is_converged_state_valid(stark); });
	stark.callbacks->add_write_frame([&]() {_write_diff(stark);});

	stark.global_potential->add_potential("EnergyPrescribedPositionsTetBarycentric", this->conn,
		[&](symx::MappedWorkspace<double>& mws, symx::Element& conn)
		{
			auto nodes = { conn["tet0"], conn["tet1"], conn["tet2"], conn["tet3"] };

			// Create symbols
			std::vector<symx::Vector> x0 = mws.make_vectors(this->dyn->x0.data, nodes);
			symx::Vector x1_prescribed = mws.make_vector(this->target_positions, conn["idx"]);
			symx::Scalar k = mws.make_scalar(this->stiffness, conn["idx"]);
			symx::Vector bary = mws.make_vector(this->barycentric_coords, conn["idx"]);
			symx::Scalar dt = mws.make_scalar(stark.dt);

			// Time integration
			std::vector<symx::Vector> v1 = mws.make_vectors(this->dyn->v1.data, nodes);
			std::vector<symx::Vector> x1 = time_integration(x0, v1, dt);

			// Projection
			symx::Vector q = bary[0] * x1[0] + bary[1] * x1[1] + bary[2] * x1[2] + bary[3] * x1[3];

			// Energy
			symx::Scalar E = 0.5 * k * (q - x1_prescribed).squared_norm();
			return E;
		}
	);

}
stark::EnergyPrescribedPositionsTetBarycentric::Handler stark::EnergyPrescribedPositionsTetBarycentric::add(const PointSetHandler& set, const std::vector<std::array<int, 4>>& tets, const std::vector<std::array<double, 4>>& bary, const Params& params)
{
	set.exit_if_not_valid("EnergyPrescribedPositionsTetBarycentric::add");

	const int n = (int)tets.size();
	if (bary.size() != n) {
		std::cout << "Stark error: EnergyPrescribedPositionsTetBarycentric::add() found an invalid input sizes." << std::endl;
		exit(-1);
	}

	const int group = (int)this->tolerance.size();
	this->tolerance.push_back(params.tolerance);
	const int begin = (int)this->target_positions.size();
	for (int i = 0; i < (int)tets.size(); i++) {
		this->target_positions.push_back(Eigen::Vector3d::Zero());
		this->conn.numbered_push_back({ group, set.get_global_index(tets[i][0]), set.get_global_index(tets[i][1]), set.get_global_index(tets[i][2]), set.get_global_index(tets[i][3]) });
		this->barycentric_coords.push_back({ bary[i][0], bary[i][1], bary[i][2], bary[i][3] });
		this->stiffness.push_back(params.stiffness);
	}
	const int end = (int)this->target_positions.size();
	this->group_begin_end.push_back({ begin, end });

	return Handler(this, group);
}
stark::EnergyPrescribedPositionsTetBarycentric::Params stark::EnergyPrescribedPositionsTetBarycentric::get_params(const Handler& handler) const
{
	handler.exit_if_not_valid("EnergyPrescribedPositionsTetBarycentric::get_params");

	const int group = handler.get_idx();
	if (group < 0 || group >= (int)this->tolerance.size()) {
		std::cout << "Stark error: EnergyPrescribedPositionsTetBarycentric::get_params() found an invalid index." << std::endl;
		exit(-1);
	}

	Params params;
	//params.stiffness = this->stiffness[group];
	params.tolerance = this->tolerance[group];
	return params;
}
void stark::EnergyPrescribedPositionsTetBarycentric::set_params(const Handler& handler, const Params& params)
{
	handler.exit_if_not_valid("EnergyPrescribedPositionsTetBarycentric::set_params");

	const int group = handler.get_idx();
	if (group < 0 || group >= (int)this->tolerance.size()) {
		std::cout << "Stark error: EnergyPrescribedPositionsTetBarycentric::set_params() found an invalid index." << std::endl;
		exit(-1);
	}

	//this->stiffness[group] = params.stiffness;
	this->tolerance[group] = params.tolerance;
}

void stark::EnergyPrescribedPositionsTetBarycentric::set_stiffness(const Handler& handler, int prescribed_idx, double stiffness)
{
	handler.exit_if_not_valid("EnergyPrescribedPositionsTetBarycentric::set_stiffness");
	const int group = handler.get_idx();
	const auto [begin, end] = this->group_begin_end[group];
	const int idx = begin + prescribed_idx;
	this->stiffness[idx] = stiffness;
}

void stark::EnergyPrescribedPositionsTetBarycentric::set_target_position(const Handler& handler, int prescribed_idx, const Eigen::Vector3d& t)
{
	handler.exit_if_not_valid("EnergyPrescribedPositionsTetBarycentric::set_target_position");
	const int group = handler.get_idx();
	const auto [begin, end] = this->group_begin_end[group];
	const int idx = begin + prescribed_idx;
	this->target_positions[idx] = t;
}

void stark::EnergyPrescribedPositionsTetBarycentric::_write_diff(Stark& stark)
{
	if (conn.size() > 0)
	{
		double att_diff = 0.0;
		for (int i = 0; i < (int)this->conn.size(); i++) {
			auto [idx, group, tet_0, tet_1, tet_2, tet_3] = this->conn[i];
			const Eigen::Vector3d& x1_0 = this->dyn->get_x1(tet_0, stark.dt);
			const Eigen::Vector3d& x1_1 = this->dyn->get_x1(tet_1, stark.dt);
			const Eigen::Vector3d& x1_2 = this->dyn->get_x1(tet_2, stark.dt);
			const Eigen::Vector3d& x1_3 = this->dyn->get_x1(tet_3, stark.dt);
			const std::array<double, 4>& bary = this->barycentric_coords[idx];
			Eigen::Vector3d q = bary[0] * x1_0 + bary[1] * x1_1 + bary[2] * x1_2 + bary[3] * x1_3;
			const Eigen::Vector3d& x1_prescribed = this->target_positions[idx];
			att_diff += (q - x1_prescribed).squaredNorm();
		}
		stark.context->logger->append("att_diff", att_diff/this->conn.size());
	}
}

bool stark::EnergyPrescribedPositionsTetBarycentric::_is_converged_state_valid(Stark& stark)
{
	const double dt = stark.dt;
	bool is_valid = true;

	for (int i = 0; i < (int)this->conn.size(); i++) {
		auto [idx, group, tet_0, tet_1, tet_2, tet_3] = this->conn[i];
		const Eigen::Vector3d& x1_0 = this->dyn->get_x1(tet_0, dt);
		const Eigen::Vector3d& x1_1 = this->dyn->get_x1(tet_1, dt);
		const Eigen::Vector3d& x1_2 = this->dyn->get_x1(tet_2, dt);
		const Eigen::Vector3d& x1_3 = this->dyn->get_x1(tet_3, dt);
		const std::array<double, 4>& bary = this->barycentric_coords[idx];
		Eigen::Vector3d q = bary[0] * x1_0 + bary[1] * x1_1 + bary[2] * x1_2 + bary[3] * x1_3;
		const Eigen::Vector3d& x1_prescribed = this->target_positions[idx];
		const double sq_distance = (q - x1_prescribed).squaredNorm();
		const double tol = this->tolerance[group];
		if (sq_distance > tol*tol) {
			is_valid = false;
			this->stiffness[i] *= 2.0;
		}
	}

	if (!is_valid) {
		stark.context->output->print("Attachment constraints are not within tolerance. Stiffness hardened.\n", symx::Verbosity::Summary);
	}

	return is_valid;
}
