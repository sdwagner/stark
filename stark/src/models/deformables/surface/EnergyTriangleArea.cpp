//
// Created by Sven Wagner on 20.10.25.
//

#include "EnergyTriangleArea.h"

#include "../deformable_tools.h"
#include "../../time_integration.h"
#include "../../../utils/include.h"


stark::EnergyTriangleArea::EnergyTriangleArea(Stark& stark, spPointDynamics dyn)
	: dyn(dyn)
{

	stark.global_potential->add_potential("EnergyTriangleArea", this->conn_complete,
		[&](symx::MappedWorkspace<double>& mws, symx::Element& conn)
		{
			// Unpack connectivity
			std::vector<symx::Index> triangle = conn.slice(2, 5);

			// Create symbols
			std::vector<symx::Vector> x0 = mws.make_vectors(this->dyn->x0.data, triangle);
			symx::Scalar rest_area = mws.make_scalar(this->rest_area, conn["idx"]);
			symx::Scalar scale = mws.make_scalar(this->scale, conn["group"]);
			symx::Scalar stiffness = mws.make_scalar(this->stiffness, conn["group"]);
			symx::Scalar dt = mws.make_scalar(stark.dt);

			// Time integration
			std::vector<symx::Vector> v1 = mws.make_vectors(this->dyn->v1.data, triangle);
			std::vector<symx::Vector> x1 = time_integration(x0, v1, dt);
			symx::Scalar area = 0.5 * ((x1[0] - x1[2]).cross3(x1[1] - x1[2])).norm();
			// Total
			symx::Scalar Energy =  stiffness * (1.0 - area/(scale*rest_area)).powN(2) *scale* rest_area;
			return Energy;
		}
	);
}
stark::EnergyTriangleArea::Handler stark::EnergyTriangleArea::add(const PointSetHandler& set, const std::vector<std::array<int, 3>>& triangles, const Params& params)
{
	set.exit_if_not_valid("EnergyTriangleArea::add");
	const int group = (int)this->stiffness.size();

	this->scale.push_back(params.scale);
	this->stiffness.push_back(params.stiffness);

	// Connectivity
	symx::LabelledConnectivity<5>* conn = &this->conn_complete;
	for (int tri_i = 0; tri_i < (int)triangles.size(); tri_i++) {
		const std::array<int, 3>& conn_loc = triangles[tri_i];
		const std::array<int, 3> conn_glob = set.get_global_indices(conn_loc);
		conn->numbered_push_back({ group, conn_glob[0], conn_glob[1], conn_glob[2] });
		double area = triangle_area(this->dyn->X[conn_glob[0]], this->dyn->X[conn_glob[1]], this->dyn->X[conn_glob[2]]);
		rest_area.push_back(area);
	}

	return Handler(this, group);
}

stark::EnergyTriangleArea::Handler stark::EnergyTriangleArea::add(const PointSetHandler& set, const std::vector<std::array<int, 3>>& triangles, const std::map<std::pair<int,int>, double>& stitched_vertices, const Params& params)
{
	set.exit_if_not_valid("EnergyTriangleArea::add");
	const int group = (int)this->stiffness.size();

	this->scale.push_back(params.scale);
	this->stiffness.push_back(params.stiffness);

	// Connectivity
	symx::LabelledConnectivity<5>* conn = &this->conn_complete;
	for (int tri_i = 0; tri_i < (int)triangles.size(); tri_i++)
	{
		const std::array<int, 3>& conn_loc = triangles[tri_i];
		const std::array<int, 3> conn_glob = set.get_global_indices(conn_loc);

		auto el = get_edge_lengths(conn_loc, stitched_vertices);
		double area = triangle_area(el);
		conn->numbered_push_back({ group, conn_glob[0], conn_glob[1], conn_glob[2] });
		rest_area.push_back(area);
	}
	return Handler(this, group);
}

stark::EnergyTriangleArea::Params stark::EnergyTriangleArea::get_params(const Handler& handler) const
{
	handler.exit_if_not_valid("EnergyTriangleStrain::get_params");

	const int group = handler.get_idx();

	Params params;
	params.scale = this->scale[group];
params.stiffness = this->stiffness[group];
	return params;
}
void stark::EnergyTriangleArea::set_params(const Handler& handler, const Params& params)
{
	handler.exit_if_not_valid("EnergyTriangleStrain::set_params");

	const int group = handler.get_idx();

	this->scale[group] = params.scale;
	this->stiffness[group] = params.stiffness;
}
