#include "EnergyTetStrain.h"

#include "../deformable_tools.h"
#include "../../time_integration.h"
#include "../../../utils/include.h"

using namespace stark;
using namespace symx;

EnergyTetStrain::EnergyTetStrain(Stark& stark, spPointDynamics dyn)
	: dyn(dyn)
{
	stark.global_potential->add_potential("EnergyTetStrain", this->conn_complete,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			// Unpack connectivity
			std::vector<Index> tet = conn.slice(2, 6);

			// Create symbols
			std::vector<Vector> v1 = mws.make_vectors(this->dyn->v1.data, tet);
			std::vector<Vector> x0 = mws.make_vectors(this->dyn->x0.data, tet);
			std::vector<Vector> X = mws.make_vectors(this->dyn->X.data, tet);
			Matrix DXinv = mws.make_matrix(this->inv_rest_jacobian, {3, 3}, conn["idx"]);
			Scalar lambda_ = mws.make_scalar(this->lambda_, conn["group"]);
			Scalar rest_volume = mws.make_scalar(this->rest_volume, conn["idx"]);
			Scalar mu_ = mws.make_scalar(this->mu_, conn["group"]);
			Scalar scale = mws.make_scalar(this->scale, conn["group"]);
			Scalar strain_limit = mws.make_scalar(this->strain_limit, conn["group"]);
			Scalar strain_limit_stiffness = mws.make_scalar(this->strain_limit_stiffness, conn["group"]);
			Scalar damping = mws.make_scalar(this->strain_damping, conn["group"]);
			Scalar dt = mws.make_scalar(stark.dt);

			// Time integration
			std::vector<Vector> x1 = time_integration(x0, v1, dt);

			// Scaling
			std::vector<Vector> Xs = { scale * X[0], scale * X[1], scale * X[2], scale * X[3] };

			// Kinematics
			Matrix DX = tet_jacobian(Xs);
			Matrix Dx1 = tet_jacobian(x1);
			Matrix F1 = Dx1 * DXinv;
			Matrix E1 = 0.5*(F1.transpose() * F1 - mws.make_identity_matrix(3));

			Matrix Dx0 = Matrix(collect_scalars({ x0[1] - x0[0], x0[2] - x0[0], x0[3] - x0[0] }), { 3, 3 }).transpose();
			Matrix F0 = Dx0 * DXinv;
			Matrix E0 = 0.5*(F0.transpose() * F0 - mws.make_identity_matrix(3));

			Matrix dE_dt = (E1 - E0) / dt;

			// [Smith et al. 2022] Stable Neo-Hookean Flesh Simulation
			// Eq. 49 from [Smith et al. 2022]
			Scalar detF = F1.det();
			Scalar Ic = F1.frobenius_norm_sq();
			Scalar alpha = 1.0 + mu_ / lambda_;//See Kim et al. 2022 (p. 86) - mu_ / (4.0 * lambda_);
			Scalar elastic_energy_density = 0.5 * mu_ * (Ic - 3.0) + 0.5 * lambda_ * (detF - alpha).powN(2);//See Kim et al. 2022 (p. 86) - 0.5 * mu_ * symx::log(Ic + 1.0);

			// Damping
			Scalar damping_energy_density = 0.5 * damping * dE_dt.frobenius_norm_sq();

			// Upper-bound-like smooth proxy for the largest eigenvalue of E
			Scalar trE = E1.trace();
			Matrix I = mws.make_identity_matrix(3);
			Matrix devE = E1 - (trE / 3.0) * I;
			Scalar dev_norm = sqrt(devE.frobenius_norm_sq());
			Scalar largest_eigv_approx = trE / 3.0 + sqrt(2.0 / 3.0) * dev_norm;

			Scalar dl = largest_eigv_approx - strain_limit;
			Scalar strain_limiting_energy_density = branch(dl > 0.0, strain_limit_stiffness*dl.powN(3)/3.0, 0.0);

			// Total
			Scalar Energy = rest_volume * (elastic_energy_density + damping_energy_density + strain_limiting_energy_density);
			return Energy;
		}
	);

	stark.global_potential->add_potential("EnergyTetStrain_Elasticity_Only", this->conn_elasticity_only,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			// Unpack connectivity
			std::vector<Index> tet = conn.slice(2, 6);

			// Create symbols
			std::vector<Vector> v1 = mws.make_vectors(this->dyn->v1.data, tet);
			std::vector<Vector> x0 = mws.make_vectors(this->dyn->x0.data, tet);
			std::vector<Vector> X = mws.make_vectors(this->dyn->X.data, tet);
			Matrix DXinv = mws.make_matrix(this->inv_rest_jacobian, {3, 3}, conn["idx"]);
			Scalar lambda_ = mws.make_scalar(this->lambda_, conn["group"]);
			Scalar rest_volume = mws.make_scalar(this->rest_volume, conn["idx"]);
			Scalar mu_ = mws.make_scalar(this->mu_, conn["group"]);
			Scalar scale = mws.make_scalar(this->scale, conn["group"]);
			Scalar dt = mws.make_scalar(stark.dt);

			// Time integration
			std::vector<Vector> x1 = time_integration(x0, v1, dt);

			// Scaling
			std::vector<Vector> Xs = { scale * X[0], scale * X[1], scale * X[2], scale * X[3] };

			// Kinematics
			Matrix DX = Matrix(collect_scalars({ Xs[1] - Xs[0], Xs[2] - Xs[0] , Xs[3] - Xs[0] }), { 3, 3 }).transpose();
			Matrix Dx1 = Matrix(collect_scalars({ x1[1] - x1[0], x1[2] - x1[0], x1[3] - x1[0] }), { 3, 3 }).transpose();
			Matrix F1 = Dx1 * DXinv;

			// [Smith et al. 2022] Stable Neo-Hookean Flesh Simulation
			// Eq. 49 from [Smith et al. 2022]
			Scalar detF = F1.det();
			Scalar Ic = F1.frobenius_norm_sq();
			Scalar alpha = 1.0 + mu_ / lambda_;//See Kim et al. 2022 (p. 86) - mu_ / (4.0 * lambda_);
			Scalar elastic_energy_density = 0.5 * mu_ * (Ic - 3.0) + 0.5 * lambda_ * (detF - alpha).powN(2);//See Kim et al. 2022 (p. 86) - 0.5 * mu_ * symx::log(Ic + 1.0);

			// Total
			Scalar Energy = rest_volume * elastic_energy_density;
			return Energy;
		}
	);
}

EnergyTetStrain::Handler EnergyTetStrain::add(const PointSetHandler& set, const std::vector<std::array<int, 4>>& tets, const Params& params)
{
	set.exit_if_not_valid("EnergyTetStrain::add");
	const int group = (int)this->youngs_modulus.size();

	this->elasticity_only.push_back(params.elasticity_only);
	this->scale.push_back(params.scale);
	double mu = params.youngs_modulus / (2.0 * (1.0 + params.poissons_ratio));
	double lambda = (params.youngs_modulus * params.poissons_ratio) / ((1.0 + params.poissons_ratio) * (1.0 - 2.0 * params.poissons_ratio)); // 3D
	double mu_ = 4.0 / 3.0 * mu;
	double lambda_ = lambda + 5.0 / 6.0 * mu;
	this->youngs_modulus.push_back(params.youngs_modulus);
	this->poissons_ratio.push_back(params.poissons_ratio);
	this->lambda_.push_back(lambda_);
	this->mu_.push_back(mu_);
	this->strain_damping.push_back(params.damping);
	this->strain_limit.push_back(params.strain_limit);
	this->strain_limit_stiffness.push_back(params.strain_limit_stiffness);

	// Connectivity
	LabelledConnectivity<6>* conn = params.elasticity_only == true ? &this->conn_elasticity_only : &this->conn_complete;
	for (int tet_i = 0; tet_i < (int)tets.size(); tet_i++) {
		const std::array<int, 4>& conn_loc = tets[tet_i];
		const std::array<int, 4> conn_glob = set.get_global_indices(conn_loc);
		conn->numbered_push_back({ group, conn_glob[0], conn_glob[1], conn_glob[2], conn_glob[3] });
		Eigen::MatrixXd x = Eigen::MatrixXd::Zero(3,4);
		x.col(0) = params.scale*this->dyn->X[conn_glob[0]];
		x.col(1) = params.scale*this->dyn->X[conn_glob[1]];
		x.col(2) = params.scale*this->dyn->X[conn_glob[2]];
		x.col(3) = params.scale*this->dyn->X[conn_glob[3]];
		Eigen::Matrix3d tet_jacobian;
		tet_jacobian.col(0) = x.col(1) - x.col(0);
		tet_jacobian.col(1) = x.col(2) - x.col(0);
		tet_jacobian.col(2) = x.col(3) - x.col(0);
		this->rest_volume.push_back(tet_jacobian.determinant() / 6.0);
		tet_jacobian = tet_jacobian.inverse().eval();
		this->inv_rest_jacobian.emplace_back(tet_jacobian(0,0), tet_jacobian(0,1), tet_jacobian(0,2), tet_jacobian(1,0), tet_jacobian(1,1), tet_jacobian(1,2), tet_jacobian(2,0), tet_jacobian(2,1), tet_jacobian(2,2));
	}

	return Handler(this, group);

}

EnergyTetStrain::Params EnergyTetStrain::get_params(const Handler& handler) const
{
	handler.exit_if_not_valid("EnergyTetStrain::get_params");

	const int group = handler.get_idx();

	Params params;
	params.elasticity_only = this->elasticity_only[group];
	params.scale = this->scale[group];
	params.youngs_modulus = this->youngs_modulus[group];
	params.poissons_ratio = this->poissons_ratio[group];
	params.damping = this->strain_damping[group];
	params.strain_limit = this->strain_limit[group];
	params.strain_limit_stiffness = this->strain_limit_stiffness[group];
	return params;
}

void EnergyTetStrain::set_params(const Handler& handler, const Params& params)
{
	handler.exit_if_not_valid("EnergyTetStrain::set_params");

	const int group = handler.get_idx();
	if (this->elasticity_only[group] != params.elasticity_only) {
		std::cout << "Error: EnergyTetStrain::set_params(): elasticity_only cannot be changed" << std::endl;
		exit(-1);
	}

	this->scale[group] = params.scale;
	this->youngs_modulus[group] = params.youngs_modulus;
	this->poissons_ratio[group] = params.poissons_ratio;
	double mu = params.youngs_modulus / (2.0 * (1.0 + params.poissons_ratio));
	double lambda = (params.youngs_modulus * params.poissons_ratio) / ((1.0 + params.poissons_ratio) * (1.0 - 2.0 * params.poissons_ratio)); // 3D
	double mu_ = 4.0 / 3.0 * mu;
	double lambda_ = lambda + 5.0 / 6.0 * mu;
	this->lambda_[group] = lambda_;
	this->mu_[group] = mu_;
	this->strain_damping[group] = params.damping;
	this->strain_limit[group] = params.strain_limit;
	this->strain_limit_stiffness[group] = params.strain_limit_stiffness;
}

