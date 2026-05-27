#include "EnergyTriangleStretching.h"

#include "../deformable_tools.h"
#include "../../time_integration.h"
#include "../../../utils/include.h"


stark::EnergyTriangleStretching::EnergyTriangleStretching(Stark& stark, spPointDynamics dyn)
	: dyn(dyn)
{

	stark.global_potential->add_potential("EnergyTriangleStretching", this->conn_complete,
		[&](symx::MappedWorkspace<double>& mws, symx::Element& conn)
		{
			// Unpack connectivity
			std::vector<symx::Index> edge = conn.slice(2, 4);

			// Create symbols
			std::vector<symx::Vector> x0 = mws.make_vectors(this->dyn->x0.data, edge);
			symx::Scalar rest_length = mws.make_scalar(this->rest_length, conn["idx"]);
			symx::Scalar scale = mws.make_scalar(this->scale, conn["group"]);
			symx::Scalar stiffness = mws.make_scalar(this->stiffness, conn["group"]);
			symx::Scalar dt = mws.make_scalar(stark.dt);

			// Time integration
			std::vector<symx::Vector> v1 = mws.make_vectors(this->dyn->v1.data, edge);
			std::vector<symx::Vector> x1 = time_integration(x0, v1, dt);
			// Total
			symx::Scalar Energy =  stiffness * (1.0 - (x1[0] - x1[1]).norm()/ (scale*rest_length)).powN(2)*scale*rest_length;
	return Energy;
		}
	);
}
stark::EnergyTriangleStretching::Handler stark::EnergyTriangleStretching::add(const PointSetHandler& set, const std::vector<std::array<int, 3>>& triangles, const Params& params)
{
	set.exit_if_not_valid("EnergyTriangleStretching::add");
	const int group = (int)this->stiffness.size();

	this->scale.push_back(params.scale);
	this->stiffness.push_back(params.stiffness);

	// Connectivity
	symx::LabelledConnectivity<4>* conn = &this->conn_complete;
	for (int tri_i = 0; tri_i < (int)triangles.size(); tri_i++) {
		const std::array<int, 3>& conn_loc = triangles[tri_i];
		const std::array<int, 3> conn_glob = set.get_global_indices(conn_loc);
		conn->numbered_push_back({ group, conn_glob[0], conn_glob[1] });
		rest_length.push_back((this->dyn->X[conn_glob[0]]-this->dyn->X[conn_glob[1]]).norm());
		conn->numbered_push_back({ group, conn_glob[1], conn_glob[2] });
		rest_length.push_back((this->dyn->X[conn_glob[1]]-this->dyn->X[conn_glob[2]]).norm());
		conn->numbered_push_back({ group, conn_glob[2], conn_glob[0] });
		rest_length.push_back((this->dyn->X[conn_glob[2]]-this->dyn->X[conn_glob[0]]).norm());
	}

	return Handler(this, group);
}

stark::EnergyTriangleStretching::Handler stark::EnergyTriangleStretching::add(const PointSetHandler& set, const std::vector<std::array<int, 3>>& triangles, const std::map<std::pair<int,int>, double>& stitched_vertices, const Params& params)
{
	set.exit_if_not_valid("EnergyTriangleStretching::add");
	const int group = (int)this->stiffness.size();

	this->scale.push_back(params.scale);
	this->stiffness.push_back(params.stiffness);

	// Connectivity
	symx::LabelledConnectivity<4>* conn = &this->conn_complete;
	for (int tri_i = 0; tri_i < (int)triangles.size(); tri_i++)
	{
		const std::array<int, 3>& conn_loc = triangles[tri_i];
		const std::array<int, 3> conn_glob = set.get_global_indices(conn_loc);

		auto el = get_edge_lengths(conn_loc, stitched_vertices);
		conn->numbered_push_back({ group, conn_glob[0], conn_glob[1] });
		conn->numbered_push_back({ group, conn_glob[1], conn_glob[2] });
		conn->numbered_push_back({ group, conn_glob[2], conn_glob[0] });
		rest_length.push_back(el[0]);
		rest_length.push_back(el[1]);
		rest_length.push_back(el[2]);
	}
	return Handler(this, group);
}

stark::EnergyTriangleStretching::Params stark::EnergyTriangleStretching::get_params(const Handler& handler) const
{
	handler.exit_if_not_valid("EnergyTriangleStrain::get_params");

	const int group = handler.get_idx();

	Params params;
	params.scale = this->scale[group];
params.stiffness = this->stiffness[group];
	return params;
}
void stark::EnergyTriangleStretching::set_params(const Handler& handler, const Params& params)
{
	handler.exit_if_not_valid("EnergyTriangleStrain::set_params");

	const int group = handler.get_idx();

	this->scale[group] = params.scale;
	this->stiffness[group] = params.stiffness;
}
