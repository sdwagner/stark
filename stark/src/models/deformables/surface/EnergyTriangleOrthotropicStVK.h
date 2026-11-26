#pragma once
#include "../../../core/Stark.h"
#include "../PointDynamics.h"
#include "../../types.h"

namespace stark
{
	/**
	* @brief Orthotropic St. Venant-Kirchhoff membrane for woven cloth (plane stress).
	*
	* The material frame comes from per-corner UV coordinates: the UV v axis is the
	* warp direction, the UV u axis the weft direction. The rest shape (metric) comes
	* from the 3D rest positions or the stitched rest edge lengths; the UVs only fix
	* the orientation of the material frame inside each rest triangle.
	*
	* Energy density (Green strain E in the weft/warp frame, index 0 = weft, 1 = warp):
	*   psi = 1/2 (C_warp E_vv^2 + 2 C_12 E_uu E_vv + C_weft E_uu^2) + 2 G E_uv^2
	* with C_warp = E_warp / (1 - nu_12 nu_21), C_weft = E_weft / (1 - nu_12 nu_21),
	* C_12 = nu_12 E_weft / (1 - nu_12 nu_21), nu_21 = nu_12 E_weft / E_warp.
	* nu_12 (poissons_ratio) is the weft contraction under warp tension. It may be
	* negative; the material is stable for nu_12^2 < E_warp / E_weft.
	*
	* stiffness_scale multiplies the elastic stresses (all four constants) and is
	* meant for switching the cloth on/off or stiffening it during a simulation.
	*/
	class EnergyTriangleOrthotropicStVK
	{
	public:
		/* Types */
		struct Params
		{
			STARK_PARAM_ELASTICITY_ONLY()
			STARK_PARAM_SCALE()
			STARK_PARAM_NON_NEGATIVE(double, thickness, 0.001)
			STARK_PARAM_POSITIVE(double, youngs_modulus_warp, 1e5)
			STARK_PARAM_POSITIVE(double, youngs_modulus_weft, 1e5)
			STARK_PARAM_NON_NEGATIVE(double, shear_modulus, 1e4)
			// May be negative (auxetic knits); admissibility is checked in add/set_params
			STARK_PARAM_NO_VALIDATION(double, poissons_ratio, 0.3)
			STARK_PARAM_NON_NEGATIVE(double, stiffness_scale, 1.0)
			STARK_PARAM_DAMPING()
			STARK_PARAM_STRAIN_LIMITING()
		};
		struct Handler { STARK_COMMON_HANDLER_CONTENTS(EnergyTriangleOrthotropicStVK, Params) };

	private:
		/* Fields */
		const spPointDynamics dyn;
		symx::LabelledConnectivity<5> conn_elasticity_only{ { "idx", "group", "i", "j", "k" } };
		symx::LabelledConnectivity<5> conn_complete{ { "idx", "group", "i", "j", "k" } };

		// Input
		std::vector<bool> elasticity_only;  // per group
		std::vector<double> scale;  // per group
		std::vector<double> thickness;  // per group
		std::vector<double> youngs_modulus_warp;  // per group
		std::vector<double> youngs_modulus_weft;  // per group
		std::vector<double> shear_modulus;  // per group
		std::vector<double> poissons_ratio;  // per group
		std::vector<double> stiffness_scale;  // per group
		std::vector<double> strain_damping;  // per group
		std::vector<double> strain_limit;  // per group
		std::vector<double> strain_limit_stiffness;  // per group
		std::vector<double> rest_area;  // per triangle (unscaled)
		std::vector<Eigen::Vector4d> inv_rest_jacobian;  // per triangle, maps to the (weft, warp) frame (unscaled)

		/* Methods */
		static void validate(const Params& params, const std::string& where);
		Handler add_impl(const PointSetHandler& set, const std::vector<std::array<int, 3>>& triangles, const std::vector<Eigen::Vector2d>& uvs, const std::vector<std::array<double, 3>>& edge_lengths, const Params& params);

	public:
		/* Methods */
		EnergyTriangleOrthotropicStVK(Stark& stark, spPointDynamics dyn);
		/**
		* @param uvs Per-corner UV coordinates, 3 per triangle in the order of `triangles`.
		*            Exits if they are missing or degenerate.
		*/
		Handler add(const PointSetHandler& set, const std::vector<std::array<int, 3>>& triangles, const std::vector<Eigen::Vector2d>& uvs, const Params& params);
		Handler add(const PointSetHandler& set, const std::vector<std::array<int, 3>>& triangles, const std::vector<Eigen::Vector2d>& uvs, const std::map<std::pair<int,int>, double>& stitched_vertices, const Params& params);
		Params get_params(const Handler& handler) const;
		void set_params(const Handler& handler, const Params& params);
	};
}
