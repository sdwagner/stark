#pragma once
#ifdef STARK_USE_IPC_TOOLKIT

#include <memory>
#include <stdexcept>

#include <ipc/collision_mesh.hpp>
#include <ipc/candidates/candidates.hpp>
#include <ipc/broad_phase/lbvh.hpp>
#include <ipc/distance/distance_type.hpp>
#include <ipc/ipc.hpp>
#include <ipc/collisions/normal/normal_collisions.hpp>
#include <ipc/ogc/trust_region.hpp>

#include "../../utils/unordered_array_set_and_map.h"
#include "../deformables/PointDynamics.h"
#include "../rigidbodies/RigidBodyHandler.h"
#include "../types.h"
#include "contact_and_friction_data.h"


namespace stark
{
	enum class ContactMethod { IPC, OGC };

	template<std::size_t N>
	struct WeightedConnectivity
	{
		symx::LabelledConnectivity<N> conn;
		std::vector<double> weights;

		explicit WeightedConnectivity(const std::array<std::string, N>& labels) : conn(labels) {}

		void push_back(const std::array<int32_t, N - 1>& values, double weight)
		{
			conn.numbered_push_back(values);
			weights.push_back(weight);
		}

		void clear()
		{
			conn.clear();
			weights.clear();
		}
	};

	template<std::size_t N>
	struct WeightedEdgeEdgeConnectivity
	{
		WeightedConnectivity<N> ea0_eb0;
		WeightedConnectivity<N> ea0_eb1;
		WeightedConnectivity<N> ea1_eb0;
		WeightedConnectivity<N> ea1_eb1;
		WeightedConnectivity<N> ea_eb0;
		WeightedConnectivity<N> ea_eb1;
		WeightedConnectivity<N> ea0_eb;
		WeightedConnectivity<N> ea1_eb;
		WeightedConnectivity<N> ea_eb;

		explicit WeightedEdgeEdgeConnectivity(const std::array<std::string, N>& labels)
			: ea0_eb0(labels), ea0_eb1(labels), ea1_eb0(labels), ea1_eb1(labels),
			  ea_eb0(labels), ea_eb1(labels), ea0_eb(labels), ea1_eb(labels), ea_eb(labels) {}

		WeightedConnectivity<N>& get(ipc::EdgeEdgeDistanceType type)
		{
			using EE = ipc::EdgeEdgeDistanceType;
			switch (type) {
				case EE::EA0_EB0: return ea0_eb0;
				case EE::EA0_EB1: return ea0_eb1;
				case EE::EA1_EB0: return ea1_eb0;
				case EE::EA1_EB1: return ea1_eb1;
				case EE::EA_EB0: return ea_eb0;
				case EE::EA_EB1: return ea_eb1;
				case EE::EA0_EB: return ea0_eb;
				case EE::EA1_EB: return ea1_eb;
				case EE::EA_EB: return ea_eb;
				default: throw std::invalid_argument("Unsupported OGC edge-edge distance type.");
			}
		}

		void clear()
		{
			ea0_eb0.clear(); ea0_eb1.clear(); ea1_eb0.clear(); ea1_eb1.clear();
			ea_eb0.clear(); ea_eb1.clear(); ea0_eb.clear(); ea1_eb.clear(); ea_eb.clear();
		}
	};

	struct OGCContactData
	{
		struct Deformable
		{
			WeightedConnectivity<5> vv{{"idx", "group_a", "group_b", "p", "q"}};
			WeightedConnectivity<6> ev{{"idx", "group_a", "group_b", "p", "e0", "e1"}};
			WeightedEdgeEdgeConnectivity<7> ee{{"idx", "group_a", "group_b", "ea0", "ea1", "eb0", "eb1"}};
			WeightedConnectivity<7> fv{{"idx", "group_a", "group_b", "p", "t0", "t1", "t2"}};
		} deformable;

		struct RigidBody
		{
			WeightedConnectivity<7> vv{{"idx", "group_a", "group_b", "a", "b", "p", "q"}};
			WeightedConnectivity<8> ev{{"idx", "group_a", "group_b", "a", "b", "p", "e0", "e1"}};
			WeightedEdgeEdgeConnectivity<9> ee{{"idx", "group_a", "group_b", "a", "b", "ea0", "ea1", "eb0", "eb1"}};
			WeightedConnectivity<9> fv{{"idx", "group_a", "group_b", "a", "b", "p", "t0", "t1", "t2"}};
		} rigid_body;

		struct Mixed
		{
			WeightedConnectivity<6> vv{{"idx", "group_a", "group_b", "rb", "p", "q"}};
			WeightedConnectivity<7> rb_vertex_d_edge{{"idx", "group_a", "group_b", "rb", "p", "e0", "e1"}};
			WeightedConnectivity<7> d_vertex_rb_edge{{"idx", "group_a", "group_b", "rb", "p", "e0", "e1"}};
			WeightedEdgeEdgeConnectivity<8> ee{{"idx", "group_a", "group_b", "rb", "ea0", "ea1", "eb0", "eb1"}};
			WeightedConnectivity<8> rb_vertex_d_face{{"idx", "group_a", "group_b", "rb", "p", "t0", "t1", "t2"}};
			WeightedConnectivity<8> d_vertex_rb_face{{"idx", "group_a", "group_b", "rb", "p", "t0", "t1", "t2"}};
		} mixed;

		void clear();
	};

	// Reuse the same barrier/friction type enums as the original class
	// (they are already defined in EnergyFrictionalContact.h but we redeclare
	//  to keep this file self-contained when the original is not included)
	enum class IPCBarrierTypeIPC { Log, Cubic };
	enum class IPCFrictionTypeIPC { C0, C1 };

	/**
	 * @brief Drop-in replacement for EnergyFrictionalContact that uses ipc-toolkit
	 *        for broad-phase collision detection, intersection testing, and CCD.
	 *
	 * Barrier and friction potentials are written as SymX symbolic expressions
	 * (identical math to EnergyFrictionalContact) so stark auto-differentiates them.
	 * The CMake option STARK_IPC_TOOLKIT_CONTACT must be ON to compile this class.
	 */
	class EnergyFrictionalContactIPC
	{
	public:
		/* Types */
		struct GlobalParams
		{
			ContactMethod contact_method = ContactMethod::IPC;
			inline GlobalParams set_contact_method(ContactMethod value) { this->contact_method = value; return *this; }
			inline ContactMethod get_contact_method() const { return this->contact_method; }

			STARK_PARAM(double, ogc_relaxed_radius_scaling, 0.9, [](double value) { return 0.0 < value && value < 1.0; })
			STARK_PARAM(double, ogc_update_threshold, 0.01, [](double value) { return 0.0 <= value && value <= 1.0; })
			STARK_PARAM_POSITIVE(double, ogc_max_query_radius, 0.01)

			STARK_PARAM_POSITIVE(double, default_contact_thickness, -1.0)

			STARK_PARAM_NON_NEGATIVE(double, min_contact_stiffness, 1e6)
			STARK_PARAM_NON_NEGATIVE(double, max_contact_stiffness, 1e20)
			STARK_PARAM_NON_NEGATIVE(double, friction_stick_slide_threshold, 0.1)

			STARK_PARAM_NO_VALIDATION(bool, collisions_enabled, true)
			STARK_PARAM_NO_VALIDATION(bool, friction_enabled, true)
			STARK_PARAM_NO_VALIDATION(bool, triangle_point_enabled, true)
			STARK_PARAM_NO_VALIDATION(bool, edge_edge_enabled, true)
			STARK_PARAM_NO_VALIDATION(bool, intersection_test_enabled, true)
		};
		struct Params
		{
			STARK_PARAM_POSITIVE(double, contact_thickness, 0.0)
		};
		struct Handler
		{
		private:
			EnergyFrictionalContactIPC* model = nullptr;
			int idx = -1;

		public:
			inline Handler() {}
			inline Handler(EnergyFrictionalContactIPC* model, int idx) : model(model), idx(idx) {}
			inline int get_idx() const { return this->idx; }
			inline EnergyFrictionalContactIPC* get_model() { return this->model; }
			inline const EnergyFrictionalContactIPC* get_model() const { return this->model; }
			inline void set_contact_thickness(double d) { this->model->set_contact_thickness(*this, d); }
			inline void set_friction(const Handler& other, double coulombs_mu) { this->model->set_friction(*this, other, coulombs_mu); }
			inline void disable_collision(const Handler& other) { this->model->disable_collision(*this, other); }
			inline void enable_collision(const Handler& other) { this->model->enable_collision(*this, other); }
			inline bool is_collision_enabled(const Handler& other) { return this->model->is_collision_enabled(*this, other); }
			inline bool is_valid() const { return this->model != nullptr; }
			inline void exit_if_not_valid(const std::string& where_) const { if (!this->is_valid()) { std::cout << "stark error: Invalid handler found in " << where_ << std::endl; exit(-1); } }
		};

	private:
		/* Fields */
		const spPointDynamics dyn;
		const spRigidBodyDynamics rb;
		bool is_initialized = false;
		bool configuration_locked = false;

		// Parameters
		GlobalParams global_params;
		double contact_stiffness = 1e3;
		IPCBarrierTypeIPC ipc_barrier_type = IPCBarrierTypeIPC::Cubic;
		IPCFrictionTypeIPC ipc_friction_type = IPCFrictionTypeIPC::C0;
		const double edge_edge_cross_norm_sq_cutoff = 1e-30;
		const double friction_displacement_perturbation = 1e-9;
		std::unique_ptr<ipc::LBVH> broad_phase;

		// SymX data
		std::vector<double> contact_thicknesses;
		IntervalVector<Eigen::Vector3d> rigidbody_local_vertices;
		Contacts_Deformables contacts_deformables;
		Contacts_RB contacts_rb;
		Contacts_RB_Deformables contacts_rb_deformables;
		Friction_Deformables friction_deformables;
		Friction_RB friction_rb;
		Friction_RB_Deformables friction_rb_deformables;

		// Mappings
		std::unordered_map<int, DeformableBookkeeping> group_to_deformable_bookkeeping;
		std::unordered_map<int, RigidBodyBookkeeping> group_to_rigidbody_bookkeeping;

		// Per-group collision meshes (same as original ContactMesh)
		std::vector<ContactMesh> meshes;

		// Pairwise data
		unordered_array_map<int, 2, double> pair_coulombs_mu;
		unordered_array_set<int, 2> disabled_collision_pairs;

		// ---- IPC-Toolkit structures (replace tmcd::ProximityDetection / IntersectionDetection) ----
		ipc::CollisionMesh ipc_mesh;           // Combined mesh for all registered handlers
		ipc::Candidates    ipc_candidates;     // Broad-phase output (rebuilt each Newton iteration)
		ipc::NormalCollisions ogc_collisions;
		std::unique_ptr<ipc::ogc::TrustRegion> ogc_trust_region;
		OGCContactData ogc_contacts;
		bool               ipc_mesh_dirty = true; // True until _build_ipc_mesh() is called
		bool               ogc_use_ccd_for_step = false;

		// Combined vertex matrix (n_total_verts × 3), kept up-to-date each step
		Eigen::MatrixXd V_combined;

		// Offsets into the combined arrays for each group
		std::vector<int> vertex_offsets; // vertex_offsets[g] = first global vertex index of group g
		std::vector<int> edge_offsets;   // edge_offsets[g]   = first global edge index of group g
		std::vector<int> tri_offsets;    // tri_offsets[g]    = first global triangle index of group g

	public:
		/* Methods */
		EnergyFrictionalContactIPC(Stark& stark, const spPointDynamics dyn, const spRigidBodyDynamics rb);

		// Deformables
		Handler add_triangles(const PointSetHandler& point_set, const std::vector<std::array<int, 3>>& triangles, const Params& params);
		Handler add_triangles(const PointSetHandler& point_set, const std::vector<std::array<int, 3>>& triangles, const std::vector<int>& point_set_map, const Params& params);
		Handler add_edges(const PointSetHandler& point_set, const std::vector<std::array<int, 2>>& edges, const Params& params);
		Handler add_edges(const PointSetHandler& point_set, const std::vector<std::array<int, 2>>& edges, const std::vector<int>& point_set_map, const Params& params);

		// Rigid bodies
		Handler add_triangles(const RigidBodyHandler& rb, const std::vector<Eigen::Vector3d>& vertices, const std::vector<std::array<int, 3>>& triangles, const Params& params);
		Handler add_edges(const RigidBodyHandler& rb, const std::vector<Eigen::Vector3d>& vertices, const std::vector<std::array<int, 2>>& edges, const Params& params);

		// Setters and getters
		GlobalParams get_global_params() const;
		void set_global_params(const GlobalParams& params);
		void set_contact_thickness(const Handler& obj, const double contact_thickness);
		double get_contact_stiffness() const;
		void set_friction(const Handler& obj0, const Handler& obj1, const double coulombs_coefficient);
		void disable_collision(const Handler& obj0, const Handler& obj1);
		void enable_collision(const Handler& obj0, const Handler& obj1);
		bool is_collision_enabled(const Handler& obj0, const Handler& obj1);
		bool is_empty() const;

	private:
		// IPC mesh management
		void _build_ipc_mesh();
		void _rebuild_collision_filter();
		void _update_vertices(Stark& stark, double dt);
		void _update_combined_V();

		// Group lookup helpers
		int _vertex_to_group(int global_v) const;
		int _edge_to_group(int global_e) const;
		int _tri_to_group(int global_t) const;
		bool _is_pair_disabled(int group_a, int group_b) const;
		void _validate_ogc_configuration() const;

		// Broad-phase + connectivity population
		void _update_vertices_and_candidates(Stark& stark, double dt);
		void _run_proximity_and_update_contacts(Stark& stark, double dt);
		void _run_proximity_and_update_friction(Stark& stark);
		void _run_ogc_proximity_and_update_friction(Stark& stark);
		void _before_newton_step__update_ogc_contacts(Stark& stark);
		void _populate_ogc_contacts();

		// Sub-type classification and dispatch helpers (mirrors EnergyFrictionalContact)
		std::array<int, 2> _get_pair_key(const Handler& obj0, const Handler& obj1);
		double _init_contact_thickness(double contact_thickness);
		Handler _add_edges_and_points(const PhysicalSystem& ps, int idx, const Params& params, int n_vertices, const std::vector<std::array<int, 2>>& edges);
		Handler _add_triangles_edges_and_points(const PhysicalSystem& ps, int idx, const Params& params, int n_vertices, const std::vector<std::array<int, 3>>& triangles);
		void _add_rigid_body(const RigidBodyHandler& rb, const Handler& handler, const std::vector<Eigen::Vector3d>& vertices);
		void _add_deformable(const PointSetHandler& point_set, const Handler& handler, const std::vector<int>& point_set_map);

		template<std::size_t N>
		std::array<int, N> _local_to_ps_global_indices(int group, const std::array<int, N>& local);
		double _get_friction(const int idx0, const int idx1);
		double _get_contact_distance(int group_a, int group_b) const;

		// SymX callbacks (identical structure to EnergyFrictionalContact)
		void _before_time_step__update_friction_contacts(Stark& stark);
		void _before_energy_evaluation__update_contacts(Stark& stark);
		bool _is_intermediate_state_valid(Stark& stark, bool is_initial_check);
		void _on_intermediate_state_invalid(Stark& stark);
		void _on_time_step_accepted(Stark& stark);
		bool _should_continue_execution(Stark& stark);
		bool _is_state_valid(Stark& stark);
		double _ccd_max_allowed_step(Stark& stark, const Eigen::VectorXd& x, const Eigen::VectorXd& du);
		double _filter_ogc_step(Stark& stark, const Eigen::VectorXd& x, Eigen::VectorXd& du);

		// SymX energy declarations (identical to EnergyFrictionalContact)
		void _energies_contact_deformables(Stark& stark);
		void _energies_contact_rb(Stark& stark);
		void _energies_contact_rb_deformables(Stark& stark);
		void _energies_friction_deformables(Stark& stark);
		void _energies_friction_rb(Stark& stark);
		void _energies_friction_rb_deformables(Stark& stark);
		void _energies_ogc(Stark& stark);

		// IPC potential helpers (same formulas as EnergyFrictionalContact, as SymX expressions)
		symx::Scalar _get_contact_distance(symx::MappedWorkspace<double>& mws, const symx::Index& group_a, const symx::Index& group_b);
		symx::Scalar _barrier_potential(const symx::Scalar& d, const symx::Scalar& dhat, const symx::Scalar& k);
		double _barrier_force(const double d, const double dhat, const double k);
		symx::Scalar _edge_edge_mollifier(const std::vector<symx::Vector>& ea, const std::vector<symx::Vector>& eb, const std::vector<symx::Vector>& ea_rest, const std::vector<symx::Vector>& eb_rest);
		symx::Scalar _friction_potential(const symx::Vector& v, const symx::Scalar& fn, const symx::Scalar& mu, const symx::Matrix& T, const symx::Scalar& epsv, const symx::Scalar& dt);

		symx::Scalar _barrier_potential(symx::MappedWorkspace<double>& mws, const Stark& stark, const symx::Scalar& d, const symx::Index& group_a, const symx::Index& group_b);
		symx::Scalar _edge_edge_mollified_barrier_potential(symx::MappedWorkspace<double>& mws, const Stark& stark, const symx::Scalar& d, const std::vector<symx::Vector>& ea, const std::vector<symx::Vector>& eb, const std::vector<symx::Vector>& ea_rest, const std::vector<symx::Vector>& eb_rest, const symx::Index& group_a, const symx::Index& group_b);
		symx::Scalar _friction_potential(symx::MappedWorkspace<double>& mws, const Stark& stark, const symx::Vector& v, const symx::Index& contact_idx, const FrictionContact& contact);
		std::pair<symx::Scalar, symx::Scalar> _ogc_barrier_potential(symx::MappedWorkspace<double>& mws, const symx::Scalar& d, const symx::Index& contact_idx, const symx::Index& group_a, const symx::Index& group_b, const std::vector<double>& weights);
		symx::Scalar _friction_point_edge(symx::MappedWorkspace<double>& mws, const Stark& stark, const symx::Vector& vp, const std::vector<symx::Vector>& ve, const symx::Index& contact_idx, FrictionPointEdge& data);
		symx::Scalar _friction_point_triangle(symx::MappedWorkspace<double>& mws, const Stark& stark, const symx::Vector& vp, const std::vector<symx::Vector>& vt, const symx::Index& contact_idx, FrictionPointTriangle& data);
		symx::Scalar _friction_edge_edge(symx::MappedWorkspace<double>& mws, const Stark& stark, const std::vector<symx::Vector>& vea, const std::vector<symx::Vector>& veb, const symx::Index& contact_idx, FrictionEdgeEdge& data);

		// SymX data-symbol getters (identical to EnergyFrictionalContact)
		std::vector<symx::Vector> _get_rb_v1(symx::MappedWorkspace<double>& mws, const Stark& stark, const symx::Index& rb_idx, const std::vector<symx::Index>& conn);
		std::vector<symx::Vector> _get_rb_x1(symx::MappedWorkspace<double>& mws, const Stark& stark, const symx::Index& rb_idx, const std::vector<symx::Index>& conn);
		std::vector<symx::Vector> _get_rb_X(symx::MappedWorkspace<double>& mws, const std::vector<symx::Index>& conn);
		std::array<std::vector<symx::Vector>, 2> _get_rb_edge(symx::MappedWorkspace<double>& mws, const Stark& stark, const symx::Index& rb_idx, const std::vector<symx::Index>& conn);
		std::array<std::vector<symx::Vector>, 3> _get_rb_edge_point(symx::MappedWorkspace<double>& mws, const Stark& stark, const symx::Index& rb_idx, const std::vector<symx::Index>& conn);

		std::vector<symx::Vector> _get_d_v1(symx::MappedWorkspace<double>& mws, const std::vector<symx::Index>& conn);
		std::vector<symx::Vector> _get_d_x1(symx::MappedWorkspace<double>& mws, const Stark& stark, const std::vector<symx::Index>& conn);
		std::vector<symx::Vector> _get_d_X(symx::MappedWorkspace<double>& mws, const std::vector<symx::Index>& conn);
		std::array<std::vector<symx::Vector>, 2> _get_d_edge(symx::MappedWorkspace<double>& mws, const Stark& stark, const std::vector<symx::Index>& conn);
		std::array<std::vector<symx::Vector>, 3> _get_d_edge_point(symx::MappedWorkspace<double>& mws, const Stark& stark, const std::vector<symx::Index>& conn);

		std::string _get_contact_label(const std::string physical_system, const std::string pair) const;
		std::string _get_friction_label(const std::string physical_system, const std::string pair) const;
	};

	// Name aliases parallel to those for EnergyFrictionalContact
	using ContactParamsIPC = EnergyFrictionalContactIPC::Params;
	using ContactGlobalParamsIPC = EnergyFrictionalContactIPC::GlobalParams;
	using ContactHandlerIPC = EnergyFrictionalContactIPC::Handler;
}

#endif // STARK_USE_IPC_TOOLKIT
