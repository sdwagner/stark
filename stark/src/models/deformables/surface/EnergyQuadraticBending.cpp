//
// Created by Sven Wagner on 21.10.25.
//

#include "EnergyQuadraticBending.h"

#include "../deformable_tools.h"
#include "../../time_integration.h"
#include "../../../utils/include.h"
#include "fmt/core.h"

stark::EnergyQuadraticBending::EnergyQuadraticBending(Stark& stark, spPointDynamics dyn)
	: dyn(dyn)
{
    stark.global_potential->add_potential("EnergyQuadraticBending", this->conn_complete,
        [&](symx::MappedWorkspace<double>& mws, symx::Element& conn)
        {
            // Unpack connectivity
            std::vector<symx::Index> internal_edge = conn.slice(2, 6);

            // Create symbols
            std::vector<symx::Vector> x0 = mws.make_vectors(this->dyn->x0.data, internal_edge);
            symx::Vector cotangent_weight = mws.make_vector(this->cotangent_weights, conn["idx"]);
            symx::Scalar stiffness = mws.make_scalar(this->bending_stiffness, conn["group"]);
            symx::Scalar dt = mws.make_scalar(stark.dt);

			// Time integration
			std::vector<symx::Vector> v1 = mws.make_vectors(this->dyn->v1.data, internal_edge);
			std::vector<symx::Vector> x1 = time_integration(x0, v1, dt);

            symx::Matrix x1_mat = symx::Matrix(symx::collect_scalars({x1[0], x1[1], x1[2], x1[3]}), {4, 3}).transpose();

            symx::Scalar Energy_bending = 0.5 * stiffness * (x1_mat * cotangent_weight).squared_norm();

            // Total energy
			return Energy_bending;
        }
    );
}

stark::EnergyQuadraticBending::Handler stark::EnergyQuadraticBending::add(const PointSetHandler& set, const std::vector<std::array<int, 3>>& triangles, const Params& params)
{
	set.exit_if_not_valid("EnergyDiscreteShells::add");
	const int group = (int)this->bending_stiffness.size();

	this->bending_stiffness.push_back(params.stiffness);
	this->elasticity_only.push_back(true);

	// Find internal_angles (dihedral) connectivity
	std::vector<std::array<int, 4>> internal_angles;
	find_internal_angles(internal_angles, triangles, set.size());

	// Initialize structures
	symx::LabelledConnectivity<6>* conn = &this->conn_complete;
	for (int internal_angle_i = 0; internal_angle_i < (int)internal_angles.size(); internal_angle_i++) {

		// Connectivity
		const std::array<int, 4>& conn_loc = internal_angles[internal_angle_i];
		const std::array<int, 4> conn_glob = set.get_global_indices(conn_loc);
		conn->numbered_push_back({ group, conn_glob[0], conn_glob[1], conn_glob[2], conn_glob[3] });

		// Fetch coordinates and compute edges
		const Eigen::Vector3d e0 = this->dyn->X[conn_glob[1]] - this->dyn->X[conn_glob[0]];
		const Eigen::Vector3d e1 = this->dyn->X[conn_glob[2]] - this->dyn->X[conn_glob[0]];
		const Eigen::Vector3d e2 = this->dyn->X[conn_glob[3]] - this->dyn->X[conn_glob[0]];
		const Eigen::Vector3d e3 = this->dyn->X[conn_glob[2]] - this->dyn->X[conn_glob[1]];
		const Eigen::Vector3d e4 = this->dyn->X[conn_glob[3]] - this->dyn->X[conn_glob[1]];

		const double e0_len = e0.norm();
		const double e1_len = e1.norm();
		const double e2_len = e2.norm();
		const double e3_len = e3.norm();
		const double e4_len = e4.norm();

		std::array l0 = { e0_len, e3_len, e1_len };
		std::array l1 = { e0_len, e4_len, e2_len };

		std::array<double, 3> cot_0 = cotangents(l0);
		std::array<double, 3> cot_1 = cotangents(l1);

		Eigen::Vector4d K = Eigen::Vector4d::Zero();

		K[0] = cot_0[1] + cot_1[1];
		K[1] = cot_0[0] + cot_1[0];
		K[2] = -(cot_0[0] + cot_0[1]);
		K[3] = -(cot_1[0] + cot_1[1]);

		double A0 = triangle_area(l0);
		double A1 = triangle_area(l1);

		K *= sqrt(3.0/(A0+A1));

		this->cotangent_weights.push_back(K);

	}

	return Handler(this, group);
}


stark::EnergyQuadraticBending::Handler stark::EnergyQuadraticBending::add(const PointSetHandler& set, const std::vector<std::array<int, 3>>& triangles, const std::map<std::pair<int,int>, double>& stitch_vertices, const Params& params)
{
	set.exit_if_not_valid("EnergyDiscreteShells::add");
	const int group = (int)this->bending_stiffness.size();

	this->bending_stiffness.push_back(params.stiffness);
	this->elasticity_only.push_back(true);

	// Find internal_angles (dihedral) connectivity
	std::vector<std::array<int, 4>> internal_angles;
	find_internal_angles(internal_angles, triangles, set.size());

	// Initialize structures
	symx::LabelledConnectivity<6>* conn = &this->conn_complete;
	for (int internal_angle_i = 0; internal_angle_i < (int)internal_angles.size(); internal_angle_i++) {

		// Connectivity
		const std::array<int, 4>& conn_loc = internal_angles[internal_angle_i];
		const std::array<int, 4> conn_glob = set.get_global_indices(conn_loc);
		conn->numbered_push_back({ group, conn_glob[0], conn_glob[1], conn_glob[2], conn_glob[3] });

		std::array<double, 3> l0 = get_edge_lengths({conn_loc[0], conn_loc[1], conn_loc[2]}, stitch_vertices);
		std::array<double, 3> l1 = get_edge_lengths({conn_loc[0], conn_loc[1], conn_loc[3]}, stitch_vertices);

		std::array<double, 3> cot_0 = cotangents(l0);
		std::array<double, 3> cot_1 = cotangents(l1);

		Eigen::Vector4d K = Eigen::Vector4d::Zero();

		K[0] = cot_0[1] + cot_1[1];
		K[1] = cot_0[0] + cot_1[0];
		K[2] = -(cot_0[0] + cot_0[1]);
		K[3] = -(cot_1[0] + cot_1[1]);

		double A0 = triangle_area(l0);
		double A1 = triangle_area(l1);

		K *= sqrt(3/(A0+A1));

		this->cotangent_weights.push_back(K);
	}

	return Handler(this, group);
}

stark::EnergyQuadraticBending::Params stark::EnergyQuadraticBending::get_params(const Handler& handler) const
{
	handler.exit_if_not_valid("EnergyDiscreteShells::get_params");

	const int group = handler.get_idx();

	Params params;
	params.elasticity_only = this->elasticity_only[group];
	params.stiffness = this->bending_stiffness[group];
	return params;
}
void stark::EnergyQuadraticBending::set_params(const Handler& handler, const Params& params)
{
	handler.exit_if_not_valid("EnergyDiscreteShells::set_params");

	const int group = handler.get_idx();
	if (this->elasticity_only[group] != params.elasticity_only) {
		std::cout << "Error: EnergyDiscreteShells::set_params(): elasticity_only cannot be changed" << std::endl;
		exit(-1);
	}

	this->bending_stiffness[group] = params.stiffness;
}
