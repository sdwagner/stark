#include "EnergyEpidermis.h"

#include "../deformable_tools.h"
#include "../../time_integration.h"
#include "../../../utils/include.h"

using namespace stark;
using namespace symx;

EnergyEpidermis::EnergyEpidermis(Stark& stark, spPointDynamics dyn)
	: dyn(dyn)
{
	stark.global_potential->add_potential("EnergyEpidermis", this->conn_complete,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			const std::vector<Index> triangle = conn.slice(2, 5);

			const std::vector<Vector> x0 = mws.make_vectors(this->dyn->x0.data, triangle);
			const std::vector<Vector> v1 = mws.make_vectors(this->dyn->v1.data, triangle);
			const Scalar scale = mws.make_scalar(this->scale, conn["group"]);
			const Scalar lambda_e = mws.make_scalar(this->lambda_e, conn["group"]);
			const Scalar gamma = mws.make_scalar(this->gamma, conn["group"]);
			const Scalar rest_area_unscaled = mws.make_scalar(this->rest_area, conn["idx"]);
			const Scalar dt = mws.make_scalar(stark.dt);

			const std::vector<Vector> x1 = time_integration(x0, v1, dt);
			const Scalar rest_area = scale.powN(2) * rest_area_unscaled;
			const Scalar area = 0.5 * ((x1[0] - x1[2]).cross3(x1[1] - x1[2])).norm();
			const Scalar area_change = area / rest_area - 1.0;

			// Eq. 16 in Sheen et al. (2021): a 2D, inversion-robust area penalty.
			return rest_area * lambda_e / 12.0 * area_change.powN(2) * (gamma * area_change.powN(2) + 6.0);
		}
	);
}

EnergyEpidermis::Handler EnergyEpidermis::add(const PointSetHandler& set, const std::vector<std::array<int, 3>>& triangles, const Params& params)
{
	set.exit_if_not_valid("EnergyEpidermis::add");
	const int group = static_cast<int>(this->lambda_e.size());

	this->scale.push_back(params.scale);
	this->lambda_e.push_back(params.lambda_e);
	this->gamma.push_back(params.gamma);

	for (const std::array<int, 3>& triangle_local : triangles) {
		const std::array<int, 3> triangle_global = set.get_global_indices(triangle_local);
		this->conn_complete.numbered_push_back({ group, triangle_global[0], triangle_global[1], triangle_global[2] });
		this->rest_area.push_back(triangle_area(
			this->dyn->X[triangle_global[0]],
			this->dyn->X[triangle_global[1]],
			this->dyn->X[triangle_global[2]]));
	}

	return Handler(this, group);
}

EnergyEpidermis::Handler EnergyEpidermis::add(const PointSetHandler& set, const std::vector<std::array<int, 3>>& triangles, const std::map<std::pair<int, int>, double>& stitched_vertices, const Params& params)
{
	set.exit_if_not_valid("EnergyEpidermis::add");
	const int group = static_cast<int>(this->lambda_e.size());

	this->scale.push_back(params.scale);
	this->lambda_e.push_back(params.lambda_e);
	this->gamma.push_back(params.gamma);

	for (const std::array<int, 3>& triangle_local : triangles) {
		const std::array<int, 3> triangle_global = set.get_global_indices(triangle_local);
		this->conn_complete.numbered_push_back({ group, triangle_global[0], triangle_global[1], triangle_global[2] });
		this->rest_area.push_back(triangle_area(get_edge_lengths(triangle_local, stitched_vertices)));
	}

	return Handler(this, group);
}

EnergyEpidermis::Params EnergyEpidermis::get_params(const Handler& handler) const
{
	handler.exit_if_not_valid("EnergyEpidermis::get_params");
	const int group = handler.get_idx();

	Params params;
	params.scale = this->scale[group];
	params.lambda_e = this->lambda_e[group];
	params.gamma = this->gamma[group];
	return params;
}

void EnergyEpidermis::set_params(const Handler& handler, const Params& params)
{
	handler.exit_if_not_valid("EnergyEpidermis::set_params");
	const int group = handler.get_idx();

	this->scale[group] = params.scale;
	this->lambda_e[group] = params.lambda_e;
	this->gamma[group] = params.gamma;
}
