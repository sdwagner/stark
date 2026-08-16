#pragma once

#include "../../../core/Stark.h"
#include "../../types.h"
#include "../PointDynamics.h"

namespace stark
{
	/**
	 * Area-preserving epidermis potential from Sheen et al. (2021), Eq. 16.
	 *
	 * For every surface triangle the potential is
	 * A_0 * lambda_e / 12 * (J - 1)^2 * (gamma * (J - 1)^2 + 6),
	 * where J is the current-to-rest area ratio.
	 */
	class EnergyEpidermis
	{
	public:
		struct Params
		{
			STARK_PARAM_SCALE()
			STARK_PARAM_NON_NEGATIVE(double, lambda_e, 50)
			STARK_PARAM_NON_NEGATIVE(double, gamma, 1.0)
		};
		struct Handler { STARK_COMMON_HANDLER_CONTENTS(EnergyEpidermis, Params) };

	private:
		const spPointDynamics dyn;
		symx::LabelledConnectivity<5> conn_complete{ { "idx", "group", "i", "j", "k" } };

		// Per-group input
		std::vector<double> scale;
		std::vector<double> lambda_e;
		std::vector<double> gamma;

		// Per-triangle input, before the per-group scale is applied.
		std::vector<double> rest_area;

	public:
		EnergyEpidermis(Stark& stark, spPointDynamics dyn);
		Handler add(const PointSetHandler& set, const std::vector<std::array<int, 3>>& triangles, const Params& params);
		Handler add(const PointSetHandler& set, const std::vector<std::array<int, 3>>& triangles, const std::map<std::pair<int, int>, double>& stitched_vertices, const Params& params);
		Params get_params(const Handler& handler) const;
		void set_params(const Handler& handler, const Params& params);
	};
}
