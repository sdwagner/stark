#include "EnergyTriangleOrthotropicStVK.h"

#include <cmath>

#include "../deformable_tools.h"
#include "../../time_integration.h"
#include "../../../utils/include.h"

using namespace stark;
using namespace symx;

namespace
{
	// Orthotropic plane-stress elastic energy density in the (weft, warp) frame.
	Scalar orthotropic_stvk_density(const Matrix& E, const Scalar& e_warp, const Scalar& e_weft, const Scalar& g, const Scalar& nu_12)
	{
		const Scalar nu_21 = nu_12 * e_weft / e_warp;
		const Scalar denom = 1.0 - nu_12 * nu_21;
		const Scalar c_warp = e_warp / denom;
		const Scalar c_weft = e_weft / denom;
		const Scalar c_12 = nu_12 * e_weft / denom;

		const Scalar& e_uu = E(0, 0);  // weft
		const Scalar& e_vv = E(1, 1);  // warp
		const Scalar& e_uv = E(0, 1);  // shear

		return 0.5 * (c_warp * e_vv * e_vv + 2.0 * c_12 * e_uu * e_vv + c_weft * e_uu * e_uu) + 2.0 * g * e_uv * e_uv;
	}
}

EnergyTriangleOrthotropicStVK::EnergyTriangleOrthotropicStVK(Stark& stark, spPointDynamics dyn)
	: dyn(dyn)
{
	stark.global_potential->add_potential("EnergyTriangleOrthotropicStVK", this->conn_complete,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			// Unpack connectivity
			std::vector<Index> triangle = conn.slice(2, 5);

			// Create symbols
			std::vector<Vector> x0 = mws.make_vectors(this->dyn->x0.data, triangle);
			std::vector<Vector> v1 = mws.make_vectors(this->dyn->v1.data, triangle);
			Scalar scale = mws.make_scalar(this->scale, conn["group"]);
			Scalar thickness = mws.make_scalar(this->thickness, conn["group"]);
			Scalar e_warp = mws.make_scalar(this->youngs_modulus_warp, conn["group"]);
			Scalar e_weft = mws.make_scalar(this->youngs_modulus_weft, conn["group"]);
			Scalar g = mws.make_scalar(this->shear_modulus, conn["group"]);
			Scalar nu = mws.make_scalar(this->poissons_ratio, conn["group"]);
			Scalar stiffness_scale = mws.make_scalar(this->stiffness_scale, conn["group"]);
			Scalar strain_damping = mws.make_scalar(this->strain_damping, conn["group"]);
			Scalar strain_limit = mws.make_scalar(this->strain_limit, conn["group"]);
			Scalar strain_limit_stiffness = mws.make_scalar(this->strain_limit_stiffness, conn["group"]);
			Scalar rest_area_unscaled = mws.make_scalar(this->rest_area, conn["idx"]);
			Vector DXinv_vec = mws.make_vector(this->inv_rest_jacobian, conn["idx"]);
			Scalar dt = mws.make_scalar(stark.dt);
			Scalar rest_area = scale * scale * rest_area_unscaled;

			// Time integration
			std::vector<Vector> x1 = time_integration(x0, v1, dt);

			// Kinematics: F maps the (weft, warp) material frame to world space
			Matrix DXinv = Matrix(collect_scalars({ DXinv_vec }), { 2, 2 }) / scale;
			Matrix Dx1_32 = Matrix(collect_scalars({ x1[1] - x1[0], x1[2] - x1[0] }), { 2, 3 }).transpose();
			Matrix F1_32 = Dx1_32 * DXinv;  // 3x2
			Matrix E1 = 0.5 * (F1_32.transpose() * F1_32 - mws.make_identity_matrix(2));

			Matrix Dx0_32 = Matrix(collect_scalars({ x0[1] - x0[0], x0[2] - x0[0] }), { 2, 3 }).transpose();
			Matrix F0_32 = Dx0_32 * DXinv;  // 3x2
			Matrix E0 = 0.5 * (F0_32.transpose() * F0_32 - mws.make_identity_matrix(2));
			Matrix dE_dt = (E1 - E0) / dt;

			// Orthotropic StVK
			Scalar elastic_energy_density = stiffness_scale * orthotropic_stvk_density(E1, e_warp, e_weft, g, nu);

			// Damping
			Scalar damping_energy_density = 0.5 * strain_damping * dE_dt.frobenius_norm_sq();

			// Strain limiting (principal Green strains)
			std::array<Scalar, 2> s = eigenvalues_sym_2x2(E1);
			Scalar strain_limiting_energy_density = mws.make_zero();
			for (int i = 0; i < 2; i++) {
				Scalar dl = s[i] - strain_limit;
				strain_limiting_energy_density += branch(dl > 0.0, strain_limit_stiffness * dl.powN(3) / 3.0, 0.0);
			}

			// Total
			Scalar Energy = thickness * rest_area * (elastic_energy_density + damping_energy_density + strain_limiting_energy_density);
			return Energy;
		}
	);

	stark.global_potential->add_potential("EnergyTriangleOrthotropicStVK_Elasticity_Only", this->conn_elasticity_only,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			// Unpack connectivity
			std::vector<Index> triangle = conn.slice(2, 5);

			// Create symbols
			std::vector<Vector> x0 = mws.make_vectors(this->dyn->x0.data, triangle);
			std::vector<Vector> v1 = mws.make_vectors(this->dyn->v1.data, triangle);
			Scalar scale = mws.make_scalar(this->scale, conn["group"]);
			Scalar thickness = mws.make_scalar(this->thickness, conn["group"]);
			Scalar e_warp = mws.make_scalar(this->youngs_modulus_warp, conn["group"]);
			Scalar e_weft = mws.make_scalar(this->youngs_modulus_weft, conn["group"]);
			Scalar g = mws.make_scalar(this->shear_modulus, conn["group"]);
			Scalar nu = mws.make_scalar(this->poissons_ratio, conn["group"]);
			Scalar stiffness_scale = mws.make_scalar(this->stiffness_scale, conn["group"]);
			Scalar rest_area_unscaled = mws.make_scalar(this->rest_area, conn["idx"]);
			Vector DXinv_vec = mws.make_vector(this->inv_rest_jacobian, conn["idx"]);
			Scalar dt = mws.make_scalar(stark.dt);
			Scalar rest_area = scale * scale * rest_area_unscaled;

			// Time integration
			std::vector<Vector> x1 = time_integration(x0, v1, dt);

			// Kinematics
			Matrix DXinv = Matrix(collect_scalars({ DXinv_vec }), { 2, 2 }) / scale;
			Matrix Dx1_32 = Matrix(collect_scalars({ x1[1] - x1[0], x1[2] - x1[0] }), { 2, 3 }).transpose();
			Matrix F1_32 = Dx1_32 * DXinv;  // 3x2
			Matrix E1 = 0.5 * (F1_32.transpose() * F1_32 - mws.make_identity_matrix(2));

			// Orthotropic StVK
			Scalar elastic_energy_density = stiffness_scale * orthotropic_stvk_density(E1, e_warp, e_weft, g, nu);

			// Total
			Scalar Energy = thickness * rest_area * elastic_energy_density;
			return Energy;
		}
	);
}

void EnergyTriangleOrthotropicStVK::validate(const Params& params, const std::string& where)
{
	// Plane-stress positive definiteness: 1 - nu_12 * nu_21 > 0 with nu_21 = nu_12 * E_weft / E_warp
	const double nu_21 = params.poissons_ratio * params.youngs_modulus_weft / params.youngs_modulus_warp;
	if (1.0 - params.poissons_ratio * nu_21 <= 0.0) {
		std::cout << "Error: " << where << ": poissons_ratio " << params.poissons_ratio
			<< " is not admissible for E_warp = " << params.youngs_modulus_warp << ", E_weft = " << params.youngs_modulus_weft
			<< " (requires poissons_ratio^2 < E_warp / E_weft)" << std::endl;
		exit(-1);
	}
}

EnergyTriangleOrthotropicStVK::Handler EnergyTriangleOrthotropicStVK::add(const PointSetHandler& set, const std::vector<std::array<int, 3>>& triangles, const std::vector<Eigen::Vector2d>& uvs, const Params& params)
{
	set.exit_if_not_valid("EnergyTriangleOrthotropicStVK::add");
	std::vector<std::array<double, 3>> edge_lengths;
	edge_lengths.reserve(triangles.size());
	for (const std::array<int, 3>& conn_loc : triangles) {
		const std::array<int, 3> conn_glob = set.get_global_indices(conn_loc);
		edge_lengths.push_back({
			(this->dyn->X[conn_glob[0]] - this->dyn->X[conn_glob[1]]).norm(),
			(this->dyn->X[conn_glob[1]] - this->dyn->X[conn_glob[2]]).norm(),
			(this->dyn->X[conn_glob[2]] - this->dyn->X[conn_glob[0]]).norm()
		});
	}
	return this->add_impl(set, triangles, uvs, edge_lengths, params);
}

EnergyTriangleOrthotropicStVK::Handler EnergyTriangleOrthotropicStVK::add(const PointSetHandler& set, const std::vector<std::array<int, 3>>& triangles, const std::vector<Eigen::Vector2d>& uvs, const std::map<std::pair<int,int>, double>& stitched_vertices, const Params& params)
{
	set.exit_if_not_valid("EnergyTriangleOrthotropicStVK::add");
	std::vector<std::array<double, 3>> edge_lengths;
	edge_lengths.reserve(triangles.size());
	for (const std::array<int, 3>& conn_loc : triangles) {
		edge_lengths.push_back(get_edge_lengths(conn_loc, stitched_vertices));
	}
	return this->add_impl(set, triangles, uvs, edge_lengths, params);
}

EnergyTriangleOrthotropicStVK::Handler EnergyTriangleOrthotropicStVK::add_impl(const PointSetHandler& set, const std::vector<std::array<int, 3>>& triangles, const std::vector<Eigen::Vector2d>& uvs, const std::vector<std::array<double, 3>>& edge_lengths, const Params& params)
{
	const std::string where = "EnergyTriangleOrthotropicStVK::add";
	validate(params, where);

	// UVs are mandatory: they define the warp/weft frame
	if (uvs.empty()) {
		std::cout << "Error: " << where << ": no UV coordinates found. The orthotropic model needs per-corner UVs to define the warp/weft directions." << std::endl;
		exit(-1);
	}
	if (uvs.size() != 3 * triangles.size()) {
		std::cout << "Error: " << where << ": expected 3 UVs per triangle (" << 3 * triangles.size() << "), got " << uvs.size() << "." << std::endl;
		exit(-1);
	}

	const int group = (int)this->youngs_modulus_warp.size();
	this->elasticity_only.push_back(params.elasticity_only);
	this->scale.push_back(params.scale);
	this->thickness.push_back(params.thickness);
	this->youngs_modulus_warp.push_back(params.youngs_modulus_warp);
	this->youngs_modulus_weft.push_back(params.youngs_modulus_weft);
	this->shear_modulus.push_back(params.shear_modulus);
	this->poissons_ratio.push_back(params.poissons_ratio);
	this->stiffness_scale.push_back(params.stiffness_scale);
	this->strain_damping.push_back(params.damping);
	this->strain_limit.push_back(params.strain_limit);
	this->strain_limit_stiffness.push_back(params.strain_limit_stiffness);

	symx::LabelledConnectivity<5>* conn = params.elasticity_only ? &this->conn_elasticity_only : &this->conn_complete;
	int n_skewed = 0;
	double max_skew_deg = 0.0;
	for (int tri_i = 0; tri_i < (int)triangles.size(); tri_i++) {
		const std::array<int, 3> conn_glob = set.get_global_indices(triangles[tri_i]);
		const std::array<double, 3>& l = edge_lengths[tri_i];

		// Rest triangle in a local 2D frame: v0 at the origin, v1 on the x axis
		const double area = triangle_area(l);
		const double height = 2.0 * area / l[0];
		double new_p = std::sqrt(std::fmax(0.0, l[2] * l[2] - height * height));
		new_p = l[2] * l[2] + l[0] * l[0] < l[1] * l[1] ? -new_p : new_p;
		Eigen::Matrix2d DX;
		DX.col(0) = Eigen::Vector2d(l[0], 0.0);
		DX.col(1) = Eigen::Vector2d(new_p, height);

		// UV edges of the same triangle
		const Eigen::Vector2d& uv0 = uvs[3 * tri_i + 0];
		const Eigen::Vector2d& uv1 = uvs[3 * tri_i + 1];
		const Eigen::Vector2d& uv2 = uvs[3 * tri_i + 2];
		Eigen::Matrix2d Duv;
		Duv.col(0) = uv1 - uv0;
		Duv.col(1) = uv2 - uv0;
		const double uv_det = Duv.determinant();
		const double uv_norms = Duv.col(0).norm() * Duv.col(1).norm();
		if (!std::isfinite(uv_det) || uv_norms == 0.0 || std::abs(uv_det) < 1e-6 * uv_norms || area <= 0.0) {
			std::cout << "Error: " << where << ": degenerate UV or rest triangle " << tri_i << " (UV det " << uv_det << ", rest area " << area << ")." << std::endl;
			exit(-1);
		}

		// Map UV directions into the local rest frame, warp = UV v axis
		const Eigen::Matrix2d A = DX * Duv.inverse();
		const Eigen::Vector2d weft_raw = A.col(0);
		const Eigen::Vector2d warp = A.col(1).normalized();
		const double orientation = A.determinant() < 0.0 ? -1.0 : 1.0;  // mirrored panels
		const Eigen::Vector2d weft = orientation * Eigen::Vector2d(warp.y(), -warp.x());

		// Diagnostics: how far the UV map is from conformal in this triangle
		const double skew_deg = std::acos(std::clamp(weft_raw.normalized().dot(weft), -1.0, 1.0)) * 180.0 / M_PI;
		max_skew_deg = std::max(max_skew_deg, skew_deg);
		if (skew_deg > 5.0) n_skewed++;

		// Material coordinates: X_mat = R^T X_local, so DX_mat^-1 = DX^-1 R
		Eigen::Matrix2d R;
		R.col(0) = weft;
		R.col(1) = warp;
		const Eigen::Matrix2d DXinv = DX.inverse() * R;

		conn->push_back({ (int)this->rest_area.size(), group, conn_glob[0], conn_glob[1], conn_glob[2] });
		this->rest_area.push_back(area);
		this->inv_rest_jacobian.emplace_back(DXinv(0, 0), DXinv(0, 1), DXinv(1, 0), DXinv(1, 1));
	}

	if (n_skewed > 0) {
		std::cout << "Warning: " << where << ": " << n_skewed << " of " << triangles.size()
			<< " triangles have UV axes more than 5 deg from orthogonal in the rest shape (max " << max_skew_deg << " deg). Warp is kept exact there." << std::endl;
	}

	return Handler(this, group);
}

EnergyTriangleOrthotropicStVK::Params EnergyTriangleOrthotropicStVK::get_params(const Handler& handler) const
{
	handler.exit_if_not_valid("EnergyTriangleOrthotropicStVK::get_params");
	const int group = handler.get_idx();

	Params params;
	params.elasticity_only = this->elasticity_only[group];
	params.scale = this->scale[group];
	params.thickness = this->thickness[group];
	params.youngs_modulus_warp = this->youngs_modulus_warp[group];
	params.youngs_modulus_weft = this->youngs_modulus_weft[group];
	params.shear_modulus = this->shear_modulus[group];
	params.poissons_ratio = this->poissons_ratio[group];
	params.stiffness_scale = this->stiffness_scale[group];
	params.damping = this->strain_damping[group];
	params.strain_limit = this->strain_limit[group];
	params.strain_limit_stiffness = this->strain_limit_stiffness[group];
	return params;
}

void EnergyTriangleOrthotropicStVK::set_params(const Handler& handler, const Params& params)
{
	handler.exit_if_not_valid("EnergyTriangleOrthotropicStVK::set_params");
	const int group = handler.get_idx();

	if (this->elasticity_only[group] != params.elasticity_only) {
		std::cout << "Error: EnergyTriangleOrthotropicStVK::set_params(): elasticity_only cannot be changed" << std::endl;
		exit(-1);
	}
	validate(params, "EnergyTriangleOrthotropicStVK::set_params");

	this->scale[group] = params.scale;
	this->thickness[group] = params.thickness;
	this->youngs_modulus_warp[group] = params.youngs_modulus_warp;
	this->youngs_modulus_weft[group] = params.youngs_modulus_weft;
	this->shear_modulus[group] = params.shear_modulus;
	this->poissons_ratio[group] = params.poissons_ratio;
	this->stiffness_scale[group] = params.stiffness_scale;
	this->strain_damping[group] = params.damping;
	this->strain_limit[group] = params.strain_limit;
	this->strain_limit_stiffness[group] = params.strain_limit_stiffness;
}
