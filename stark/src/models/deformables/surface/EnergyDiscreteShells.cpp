#include "EnergyDiscreteShells.h"

#include "../../time_integration.h"
#include "../../../utils/include.h"

using namespace stark;
using namespace symx;
constexpr double EPSILON = 1e-12;

EnergyDiscreteShells::EnergyDiscreteShells(Stark& stark, spPointDynamics dyn)
	: dyn(dyn)
{
	auto dihedral_angle_rad_f = [](std::vector<Vector>& x)
		{
			auto e0 = x[1] - x[0];
			auto e1 = x[2] - x[0];
			auto e2 = x[3] - x[0];

			auto n0 = e0.cross3(e1);
			auto n1 = -e0.cross3(e2);

			auto dihedral_angle_rad = ((1.0 - EPSILON) * n0.normalized().dot(n1.normalized())).acos();
			return dihedral_angle_rad;
		};


	stark.global_potential->add_potential("EnergyDiscreteShells", this->conn_complete,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			// Unpack connectivity
			std::vector<Index> internal_edge = conn.slice(2, 6);

			// Create symbols
			std::vector<Vector> v1 = mws.make_vectors(this->dyn->v1.data, internal_edge);
			std::vector<Vector> x0 = mws.make_vectors(this->dyn->x0.data, internal_edge);
			Scalar rest_dihedral_angle = mws.make_scalar(this->rest_dihedral_angle_rad, conn["idx"]);
			Scalar rest_edge_length = mws.make_scalar(this->rest_edge_length, conn["idx"]);
			Scalar rest_height = mws.make_scalar(this->rest_height, conn["idx"]);
			Scalar scale = mws.make_scalar(this->scale, conn["group"]);
			Scalar stiffness = mws.make_scalar(this->bending_stiffness, conn["group"]);
			Scalar damping = mws.make_scalar(this->bending_damping, conn["group"]);
			Scalar dt = mws.make_scalar(stark.dt);

			// Time integration
			std::vector<Vector> x1 = time_integration(x0, v1, dt);

			// Scaling
			Scalar scaled_rest_edge_length = rest_edge_length * scale;
			Scalar scaled_rest_height = rest_height * scale;

			// Bending (da: dihedral angle)
			Scalar da_1 = dihedral_angle_rad_f(x1);
			Scalar da_delta = da_1 - rest_dihedral_angle;
			Scalar Energy_bending = stiffness * (da_delta * da_delta) * (scaled_rest_edge_length / scaled_rest_height);

			// Damping
			Scalar da_0 = dihedral_angle_rad_f(x0);
			Scalar Energy_damping = damping * 1.0 / dt * (0.5 * da_1.powN(2) - da_0 * da_1) * (scaled_rest_edge_length / scaled_rest_height);

			// Total energy
			return Energy_bending + Energy_damping;
		}
	);
	stark.global_potential->add_potential("EnergyDiscreteShells_Elasticity_Only", this->conn_elasticity_only,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			// Unpack connectivity
			std::vector<Index> internal_edge = conn.slice(2, 6);

			// Create symbols
			std::vector<Vector> v1 = mws.make_vectors(this->dyn->v1.data, internal_edge);
			std::vector<Vector> x0 = mws.make_vectors(this->dyn->x0.data, internal_edge);
			Scalar rest_dihedral_angle = mws.make_scalar(this->rest_dihedral_angle_rad, conn["idx"]);
			Scalar rest_edge_length = mws.make_scalar(this->rest_edge_length, conn["idx"]);
			Scalar rest_height = mws.make_scalar(this->rest_height, conn["idx"]);
			Scalar scale = mws.make_scalar(this->scale, conn["group"]);
			Scalar stiffness = mws.make_scalar(this->bending_stiffness, conn["group"]);
			Scalar dt = mws.make_scalar(stark.dt);

			// Time integration
			std::vector<Vector> x1 = time_integration(x0, v1, dt);

			// Scaling
			Scalar scaled_rest_edge_length = rest_edge_length * scale;
			Scalar scaled_rest_height = rest_height * scale;

			// Bending (da: dihedral angle)
			Scalar da_1 = dihedral_angle_rad_f(x1);
			Scalar da_delta = da_1 - rest_dihedral_angle;
			Scalar Energy_bending = stiffness * (da_delta * da_delta) * (scaled_rest_edge_length / scaled_rest_height);

			// Total energy
			return Energy_bending;
		}
	);
}
EnergyDiscreteShells::Handler EnergyDiscreteShells::add(const PointSetHandler& set, const std::vector<std::array<int, 3>>& triangles, const Params& params)
{
	set.exit_if_not_valid("EnergyDiscreteShells::add");
	const int group = (int)this->bending_stiffness.size();

	this->elasticity_only.push_back(params.elasticity_only);
	this->scale.push_back(params.scale);
	this->bending_stiffness.push_back(params.stiffness);
	this->bending_damping.push_back(params.damping);
	this->flat_rest_angle.push_back(params.flat_rest_angle);

	// Find internal_angles (dihedral) connectivity
	std::vector<std::array<int, 4>> internal_angles;
	find_internal_angles(internal_angles, triangles, set.size());

	// Initialize structures
	symx::LabelledConnectivity<6>* conn = params.elasticity_only == true ? &this->conn_elasticity_only : &this->conn_complete;
	for (int internal_angle_i = 0; internal_angle_i < (int)internal_angles.size(); internal_angle_i++) {

		// Connectivity
		const std::array<int, 4>& conn_loc = internal_angles[internal_angle_i];
		const std::array<int, 4> conn_glob = set.get_global_indices(conn_loc);
		conn->numbered_push_back({ group, conn_glob[0], conn_glob[1], conn_glob[2], conn_glob[3] });

		// Fetch coordinates and compute edges
		const Eigen::Vector3d e0 = this->dyn->X[conn_glob[1]] - this->dyn->X[conn_glob[0]];
		const Eigen::Vector3d e1 = this->dyn->X[conn_glob[2]] - this->dyn->X[conn_glob[0]];
		const Eigen::Vector3d e2 = this->dyn->X[conn_glob[3]] - this->dyn->X[conn_glob[0]];

		// Edge length
		const double edge_length = e0.norm();
		this->rest_edge_length.push_back(edge_length);

		const Eigen::Vector3d n0 = e0.cross(e1);
		const Eigen::Vector3d n1 = -e0.cross(e2);

		double dihedral_angle_rad = 0.0;
		if (!params.flat_rest_angle)
			dihedral_angle_rad = std::acos((1.0 - EPSILON) * n0.normalized().dot(n1.normalized()));
		this->rest_dihedral_angle_rad.push_back(dihedral_angle_rad);

		// Heights
		const double A0 = 0.5 * n0.norm();
		const double A1 = 0.5 * n1.norm();

		const double h1 = 2.0 * A0 / edge_length;
		const double h2 = 2.0 * A1 / edge_length;

		const double height_factor = 1.0 / 6.0 * (h1 + h2);
		this->rest_height.push_back(height_factor);
	}

	return Handler(this, group);
}

EnergyDiscreteShells::Handler EnergyDiscreteShells::add(const PointSetHandler& set, const std::vector<std::array<int, 3>>& triangles, const std::map<std::pair<int,int>, double>& stitch_vertices, const Params& params)
{
	set.exit_if_not_valid("EnergyDiscreteShells::add");
	const int group = (int)this->bending_stiffness.size();

	this->elasticity_only.push_back(params.elasticity_only);
	this->scale.push_back(params.scale);
	this->bending_stiffness.push_back(params.stiffness);
	this->bending_damping.push_back(params.damping);
	this->flat_rest_angle.push_back(params.flat_rest_angle);

	// Find internal_angles (dihedral) connectivity
	std::vector<std::array<int, 4>> internal_angles;
	find_internal_angles(internal_angles, triangles, set.size());

	// Initialize structures
	symx::LabelledConnectivity<6>* conn = params.elasticity_only == true ? &this->conn_elasticity_only : &this->conn_complete;
	for (int internal_angle_i = 0; internal_angle_i < (int)internal_angles.size(); internal_angle_i++) {

		// Connectivity
		const std::array<int, 4>& conn_loc = internal_angles[internal_angle_i];
		const std::array<int, 4> conn_glob = set.get_global_indices(conn_loc);

		conn->numbered_push_back({ group, conn_glob[0], conn_glob[1], conn_glob[2], conn_glob[3] });

		// Fetch coordinates and compute edges
		const Eigen::Vector3d e0 = this->dyn->X[conn_glob[1]] - this->dyn->X[conn_glob[0]];
		const Eigen::Vector3d e1 = this->dyn->X[conn_glob[2]] - this->dyn->X[conn_glob[0]];
		const Eigen::Vector3d e2 = this->dyn->X[conn_glob[3]] - this->dyn->X[conn_glob[0]];

		int min_idx = std::min(conn_loc[0], conn_loc[1]);
		int max_idx = std::max(conn_loc[0], conn_loc[1]);
		std::pair<int, int> edge = { min_idx, max_idx };
		double e = stitch_vertices.at(edge);
		// Edge length
		this->rest_edge_length.push_back(e);

		const Eigen::Vector3d n0 = e0.cross(e1);
		const Eigen::Vector3d n1 = -e0.cross(e2);

		double dihedral_angle_rad = 0.0;
		if (!params.flat_rest_angle)
			dihedral_angle_rad = std::acos((1.0 - EPSILON) * n0.normalized().dot(n1.normalized()));
		this->rest_dihedral_angle_rad.push_back(dihedral_angle_rad);


		std::array<int, 3> edges0 = { conn_loc[0], conn_loc[1], conn_loc[2] };
		std::array<int, 3> edges1 = { conn_loc[0], conn_loc[1], conn_loc[3] };
		std::array<double, 3> l0 = get_edge_lengths(edges0, stitch_vertices);
		std::array<double, 3> l1 = get_edge_lengths(edges1, stitch_vertices);
		double A0 = triangle_area(l0);
		double A1 = triangle_area(l1);

		const double h1 = 2.0 * A0 / e;
		const double h2 = 2.0 * A1 / e;

		const double height_factor = 1.0 / 6.0 * (h1 + h2);
		this->rest_height.push_back(height_factor);
	}

	return Handler(this, group);
}


EnergyDiscreteShells::Params EnergyDiscreteShells::get_params(const Handler& handler) const
{
	handler.exit_if_not_valid("EnergyDiscreteShells::get_params");

	const int group = handler.get_idx();

	Params params;
	params.scale = this->scale[group];
	params.stiffness = this->bending_stiffness[group];
	params.damping = this->bending_damping[group];
	params.flat_rest_angle = this->flat_rest_angle[group];
	return params;
}
void EnergyDiscreteShells::set_params(const Handler& handler, const Params& params)
{
	handler.exit_if_not_valid("EnergyDiscreteShells::set_params");

	const int group = handler.get_idx();
	if (this->flat_rest_angle[group] != params.flat_rest_angle) {
		std::cout << "Error: EnergyDiscreteShells::set_params(): flat_rest_angle cannot be changed" << std::endl;
		exit(-1);
	}

	this->scale[group] = params.scale;
	this->bending_stiffness[group] = params.stiffness;
	this->bending_damping[group] = params.damping;
	this->flat_rest_angle[group] = params.flat_rest_angle;
}
