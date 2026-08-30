#ifdef STARK_USE_IPC_TOOLKIT
#include "EnergyFrictionalContactIPC.h"

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <fmt/format.h>

#include <ipc/distance/point_point.hpp>
#include <ipc/distance/point_edge.hpp>
#include <ipc/distance/point_triangle.hpp>
#include <ipc/distance/edge_edge.hpp>

#include "../time_integration.h"
#include "../distances.h"
#include "../rigidbodies/rigidbody_transformations.h"
#include "../../utils/include.h"
#include "friction_geometry.h"

using namespace stark;
using namespace symx;

void OGCContactData::clear()
{
	deformable.vv.clear();
	deformable.ev.clear();
	deformable.ee.clear();
	deformable.fv.clear();
	rigid_body.vv.clear();
	rigid_body.ev.clear();
	rigid_body.ee.clear();
	rigid_body.fv.clear();
	mixed.vv.clear();
	mixed.rb_vertex_d_edge.clear();
	mixed.d_vertex_rb_edge.clear();
	mixed.ee.clear();
	mixed.rb_vertex_d_face.clear();
	mixed.d_vertex_rb_face.clear();
}

/* ========================================================================================== */
/* ===================================  CONSTRUCTOR  ======================================== */
/* ========================================================================================== */

EnergyFrictionalContactIPC::EnergyFrictionalContactIPC(Stark& stark, const spPointDynamics dyn, const spRigidBodyDynamics rb)
	: dyn(dyn), rb(rb)
{
	if (!stark.settings.simulation.init_frictional_contact) { return; }
	this->is_initialized = true;

	stark.callbacks->add_before_simulation([&]() {
		this->_validate_ogc_configuration();
		this->configuration_locked = true;
		// The contact method is selected through set_global_params() after this
		// object is constructed. Register the OGC-specific filter only once that
		// selection is final: an identity filter still switches Newton into its
		// filtered line-search path, making IPC behave differently for no reason.
		if (this->global_params.contact_method == ContactMethod::OGC) {
			stark.callbacks->newton->add_step_filter(
				[&](const Eigen::VectorXd& x, Eigen::VectorXd& du) -> double {
					return this->_filter_ogc_step(stark, x, du);
				}
			);
		}
	});
	stark.callbacks->add_before_time_step([&]() { this->_before_time_step__update_friction_contacts(stark); });
	stark.callbacks->newton->add_before_step([&]() { this->_before_newton_step__update_ogc_contacts(stark); });
	stark.callbacks->newton->add_before_energy_evaluation([&]() { this->_before_energy_evaluation__update_contacts(stark); });
	//stark.callbacks->newton->add_is_intermediate_state_valid([&]() { return this->_is_intermediate_state_valid(stark, false); });
	stark.callbacks->newton->add_is_initial_state_valid([&]() { return this->_is_intermediate_state_valid(stark, true); });
	stark.callbacks->newton->add_is_converged_state_valid([&]() { return this->_is_intermediate_state_valid(stark, false); });
	stark.callbacks->newton->add_on_intermediate_state_invalid([&]() { this->_on_intermediate_state_invalid(stark); });
	stark.callbacks->add_on_time_step_accepted([&]() { this->_on_time_step_accepted(stark); });
	stark.callbacks->add_should_continue_execution([&]() { return this->_should_continue_execution(stark); });
	stark.callbacks->add_is_current_collision_state_valid([&]() { return this->_is_state_valid(stark); });

	stark.callbacks->newton->add_max_allowed_step(
		[&](const Eigen::VectorXd& x, const Eigen::VectorXd& du) -> double {
			return this->_ccd_max_allowed_step(stark, x, du);
		}
	);

	this->_energies_contact_deformables(stark);
	this->_energies_contact_rb(stark);
	this->_energies_contact_rb_deformables(stark);
	this->_energies_friction_deformables(stark);
	this->_energies_friction_rb(stark);
	this->_energies_friction_rb_deformables(stark);
	this->_energies_ogc(stark);
	this->broad_phase = std::make_unique<ipc::LBVH>();
	this->ogc_collisions.set_collision_set_type(ipc::NormalCollisions::CollisionSetType::OGC);
}

/* ========================================================================================== */
/* ===================================  PUBLIC METHODS  ===================================== */
/* ========================================================================================== */

EnergyFrictionalContactIPC::GlobalParams EnergyFrictionalContactIPC::get_global_params() const { return this->global_params; }
void EnergyFrictionalContactIPC::set_global_params(const GlobalParams& params)
{
	if (this->configuration_locked
		&& params.contact_method != this->global_params.contact_method) {
		throw std::logic_error(
			"EnergyFrictionalContactIPC: contact method cannot be changed after simulation initialization.");
	}
	if (params.contact_method == ContactMethod::OGC) {
		if (!params.triangle_point_enabled || !params.edge_edge_enabled) {
			throw std::invalid_argument(
				"EnergyFrictionalContactIPC: OGC requires both triangle-point and edge-edge contact modes.");
		}
	}
	if (params.contact_method != this->global_params.contact_method) {
		this->ogc_trust_region.reset();
		this->ogc_contacts.clear();
	} else if (this->ogc_trust_region) {
		this->ogc_trust_region->should_update_trust_region = true;
	}
	this->global_params = params;
	this->contact_stiffness = params.min_contact_stiffness;
}
double EnergyFrictionalContactIPC::get_contact_stiffness() const { return this->contact_stiffness; }

EnergyFrictionalContactIPC::Handler EnergyFrictionalContactIPC::add_triangles(const PointSetHandler& point_set, const std::vector<std::array<int, 3>>& triangles, const Params& params)
{
	point_set.exit_if_not_valid("EnergyFrictionalContactIPC::add_triangles");
	return this->add_triangles(point_set, triangles, point_set.all(), params);
}
EnergyFrictionalContactIPC::Handler EnergyFrictionalContactIPC::add_triangles(const PointSetHandler& point_set, const std::vector<std::array<int, 3>>& triangles, const std::vector<int>& point_set_map, const Params& params)
{
	point_set.exit_if_not_valid("EnergyFrictionalContactIPC::add_triangles");
	Handler handler = this->_add_triangles_edges_and_points(PhysicalSystem::Deformable, point_set.get_idx(), params, (int)point_set_map.size(), triangles);
	this->_add_deformable(point_set, handler, point_set_map);
	return handler;
}
EnergyFrictionalContactIPC::Handler EnergyFrictionalContactIPC::add_edges(const PointSetHandler& point_set, const std::vector<std::array<int, 2>>& edges, const Params& params)
{
	point_set.exit_if_not_valid("EnergyFrictionalContactIPC::add_edges");
	return this->add_edges(point_set, edges, point_set.all(), params);
}
EnergyFrictionalContactIPC::Handler EnergyFrictionalContactIPC::add_edges(const PointSetHandler& point_set, const std::vector<std::array<int, 2>>& edges, const std::vector<int>& point_set_map, const Params& params)
{
	point_set.exit_if_not_valid("EnergyFrictionalContactIPC::add_edges");
	Handler handler = this->_add_edges_and_points(PhysicalSystem::Deformable, point_set.get_idx(), params, (int)point_set_map.size(), edges);
	this->_add_deformable(point_set, handler, point_set_map);
	return handler;
}
EnergyFrictionalContactIPC::Handler EnergyFrictionalContactIPC::add_triangles(const RigidBodyHandler& rb_h, const std::vector<Eigen::Vector3d>& vertices, const std::vector<std::array<int, 3>>& triangles, const Params& params)
{
	rb_h.exit_if_not_valid("EnergyFrictionalContactIPC::add_triangles");
	Handler handler = this->_add_triangles_edges_and_points(PhysicalSystem::Rigidbody, rb_h.get_idx(), params, (int)vertices.size(), triangles);
	this->_add_rigid_body(rb_h, handler, vertices);
	return handler;
}
EnergyFrictionalContactIPC::Handler EnergyFrictionalContactIPC::add_edges(const RigidBodyHandler& rb_h, const std::vector<Eigen::Vector3d>& vertices, const std::vector<std::array<int, 2>>& edges, const Params& params)
{
	rb_h.exit_if_not_valid("EnergyFrictionalContactIPC::add_edges");
	Handler handler = this->_add_edges_and_points(PhysicalSystem::Rigidbody, rb_h.get_idx(), params, (int)vertices.size(), edges);
	this->_add_rigid_body(rb_h, handler, vertices);
	return handler;
}
void EnergyFrictionalContactIPC::set_contact_thickness(const Handler& obj, const double contact_thickness)
{
	obj.exit_if_not_valid("EnergyFrictionalContactIPC::set_contact_thickness");
	if (contact_thickness <= 0.0) { std::cout << "stark error: Contact thickness must be positive.\n"; exit(-1); }
	this->contact_thicknesses[obj.get_idx()] = contact_thickness;
	this->ogc_trust_region.reset();
}
void EnergyFrictionalContactIPC::set_friction(const Handler& obj0, const Handler& obj1, const double coulombs_coefficient)
{
	obj0.exit_if_not_valid("EnergyFrictionalContactIPC::set_friction");
	obj1.exit_if_not_valid("EnergyFrictionalContactIPC::set_friction");
	this->pair_coulombs_mu[this->_get_pair_key(obj0, obj1)] = coulombs_coefficient;
}
void EnergyFrictionalContactIPC::disable_collision(const Handler& obj0, const Handler& obj1)
{
	obj0.exit_if_not_valid("EnergyFrictionalContactIPC::disable_collision");
	obj1.exit_if_not_valid("EnergyFrictionalContactIPC::disable_collision");
	const std::array<int, 2> pair = this->_get_pair_key(obj0, obj1);
	this->disabled_collision_pairs.insert(pair);
	// If the mesh is already built, update its filter in-place rather than triggering
	// a full geometry rebuild. The geometry didn't change, only the pair exclusion list.
	if (!this->ipc_mesh_dirty)
		this->_rebuild_collision_filter();
	this->ogc_trust_region.reset();
	this->ogc_contacts.clear();
}
void EnergyFrictionalContactIPC::enable_collision(const Handler& obj0, const Handler& obj1)
{
	obj0.exit_if_not_valid("EnergyFrictionalContactIPC::enable_collision");
	obj1.exit_if_not_valid("EnergyFrictionalContactIPC::enable_collision");
	const std::array<int, 2> pair = this->_get_pair_key(obj0, obj1);
	this->disabled_collision_pairs.erase(pair);
	if (!this->ipc_mesh_dirty)
		this->_rebuild_collision_filter();
	this->ogc_trust_region.reset();
	this->ogc_contacts.clear();
}
bool EnergyFrictionalContactIPC::is_collision_enabled(const Handler& obj0, const Handler& obj1)
{
	obj0.exit_if_not_valid("EnergyFrictionalContactIPC::is_collision_enabled");
	obj1.exit_if_not_valid("EnergyFrictionalContactIPC::is_collision_enabled");
	return !this->disabled_collision_pairs.contains(this->_get_pair_key(obj0, obj1));
}
bool EnergyFrictionalContactIPC::is_empty() const { return this->meshes.empty(); }


/* ========================================================================================== */
/* ===================================  HELPERS  ============================================ */
/* ========================================================================================== */

std::array<int, 2> EnergyFrictionalContactIPC::_get_pair_key(const Handler& obj0, const Handler& obj1)
{
	return { std::min(obj0.get_idx(), obj1.get_idx()), std::max(obj0.get_idx(), obj1.get_idx()) };
}
double EnergyFrictionalContactIPC::_init_contact_thickness(double contact_thickness)
{
	if (!this->is_initialized) { return contact_thickness; }
	if (contact_thickness == 0.0) {
		if (this->global_params.default_contact_thickness > 0.0) {
			contact_thickness = this->global_params.default_contact_thickness;
		} else {
			std::cout << "stark error: Undefined contact thickness. Set per-object or global default.\n";
			exit(-1);
		}
	}
	return contact_thickness;
}

EnergyFrictionalContactIPC::Handler EnergyFrictionalContactIPC::_add_edges_and_points(const PhysicalSystem& ps, int idx, const Params& params, int n_vertices, const std::vector<std::array<int, 2>>& edges)
{
	const int group = (int)this->contact_thicknesses.size();
	this->contact_thicknesses.push_back(this->_init_contact_thickness(params.contact_thickness));

	this->meshes.push_back({});
	auto& mesh = this->meshes.back();
	mesh.ps = ps;
	mesh.idx_in_ps = idx;
	mesh.vertices.resize(n_vertices, Eigen::Vector3d::Zero());
	mesh.loc_edges.insert(mesh.loc_edges.end(), edges.begin(), edges.end());
	mesh.loc_triangles.push_back({-1, -1, -1}); // dummy

	this->ipc_mesh_dirty = true;
	return Handler(this, group);
}
EnergyFrictionalContactIPC::Handler EnergyFrictionalContactIPC::_add_triangles_edges_and_points(const PhysicalSystem& ps, int idx, const Params& params, int n_vertices, const std::vector<std::array<int, 3>>& triangles)
{
	const int group = (int)this->contact_thicknesses.size();
	this->contact_thicknesses.push_back(this->_init_contact_thickness(params.contact_thickness));

	const std::vector<std::array<int, 2>> edges = find_edges_from_simplices(triangles, n_vertices);

	this->meshes.push_back({});
	auto& mesh = this->meshes.back();
	mesh.ps = ps;
	mesh.idx_in_ps = idx;
	mesh.vertices.resize(n_vertices, Eigen::Vector3d::Zero());
	mesh.loc_edges.insert(mesh.loc_edges.end(), edges.begin(), edges.end());
	mesh.loc_triangles.insert(mesh.loc_triangles.end(), triangles.begin(), triangles.end());

	this->ipc_mesh_dirty = true;
	return Handler(this, group);
}
void EnergyFrictionalContactIPC::_add_rigid_body(const RigidBodyHandler& rb_h, const Handler& handler, const std::vector<Eigen::Vector3d>& vertices)
{
	const int group = handler.get_idx();
	const int rb_group = (int)this->group_to_rigidbody_bookkeeping.size();
	this->group_to_rigidbody_bookkeeping[group] = RigidBodyBookkeeping(group, rb_group, rb_h.get_idx());
	this->rigidbody_local_vertices.append(vertices);
	this->disable_collision(handler, handler);
}
void EnergyFrictionalContactIPC::_add_deformable(const PointSetHandler& point_set, const Handler& handler, const std::vector<int>& point_set_map)
{
	const int group = handler.get_idx();
	const int dcg = (int)this->group_to_deformable_bookkeeping.size();
	this->group_to_deformable_bookkeeping[group] = DeformableBookkeeping(group, dcg, point_set.get_idx(), point_set_map);
}

double EnergyFrictionalContactIPC::_get_friction(const int idx0, const int idx1)
{
	const std::array<int, 2> pair = { std::min(idx0, idx1), std::max(idx0, idx1) };
	auto it = this->pair_coulombs_mu.find(pair);
	return (it == this->pair_coulombs_mu.end()) ? 0.0 : it->second;
}
double EnergyFrictionalContactIPC::_get_contact_distance(int group_a, int group_b) const
{
	return this->contact_thicknesses[group_a] + this->contact_thicknesses[group_b];
}

template<std::size_t N>
std::array<int, N> EnergyFrictionalContactIPC::_local_to_ps_global_indices(int group, const std::array<int, N>& local)
{
	const PhysicalSystem ps = this->meshes[group].ps;
	std::array<int, N> global;
	if (ps == PhysicalSystem::Deformable) {
		const DeformableBookkeeping& bk = this->group_to_deformable_bookkeeping[group];
		for (int i = 0; i < (int)N; i++)
			global[i] = this->dyn->get_global_index(bk.deformable_idx, bk.point_set_map[local[i]]);
	} else if (ps == PhysicalSystem::Rigidbody) {
		const RigidBodyBookkeeping& bk = this->group_to_rigidbody_bookkeeping[group];
		for (int i = 0; i < (int)N; i++)
			global[i] = this->rigidbody_local_vertices.get_global_index(bk.rb_collision_group, local[i]);
	} else {
		std::cout << "stark error: Unknown physical system in _local_to_ps_global_indices\n"; exit(-1);
	}
	return global;
}
template std::array<int, 1> EnergyFrictionalContactIPC::_local_to_ps_global_indices<1>(int, const std::array<int, 1>&);
template std::array<int, 2> EnergyFrictionalContactIPC::_local_to_ps_global_indices<2>(int, const std::array<int, 2>&);
template std::array<int, 3> EnergyFrictionalContactIPC::_local_to_ps_global_indices<3>(int, const std::array<int, 3>&);


/* ========================================================================================== */
/* ===================================  IPC MESH BUILD  ===================================== */
/* ========================================================================================== */

void EnergyFrictionalContactIPC::_build_ipc_mesh()
{
	const int n_groups = (int)this->meshes.size();
	this->vertex_offsets.resize(n_groups + 1, 0);
	this->edge_offsets.resize(n_groups + 1, 0);
	this->tri_offsets.resize(n_groups + 1, 0);

	for (int g = 0; g < n_groups; g++) {
		this->vertex_offsets[g + 1] = this->vertex_offsets[g] + (int)this->meshes[g].vertices.size();
		this->edge_offsets[g + 1]   = this->edge_offsets[g]   + (int)this->meshes[g].loc_edges.size();
		const int n_tris = (int)this->meshes[g].loc_triangles.size();
		// If the first triangle is dummy (-1,-1,-1), this group has no real triangles
		const bool has_tris = n_tris > 0 && this->meshes[g].loc_triangles[0][0] >= 0;
		this->tri_offsets[g + 1] = this->tri_offsets[g] + (has_tris ? n_tris : 0);
	}

	const int n_verts  = this->vertex_offsets.back();
	const int n_edges  = this->edge_offsets.back();
	const int n_tris   = this->tri_offsets.back();

	Eigen::MatrixXd V_rest(n_verts, 3);
	Eigen::MatrixXi E(n_edges, 2);
	Eigen::MatrixXi F(n_tris, 3);

	for (int g = 0; g < n_groups; g++) {
		const int voff = this->vertex_offsets[g];
		const int eoff = this->edge_offsets[g];
		const int toff = this->tri_offsets[g];

		for (int i = 0; i < (int)this->meshes[g].vertices.size(); i++) {
			V_rest.row(voff + i) = this->meshes[g].vertices[i].transpose();
		}
		for (int i = 0; i < (int)this->meshes[g].loc_edges.size(); i++) {
			E(eoff + i, 0) = voff + this->meshes[g].loc_edges[i][0];
			E(eoff + i, 1) = voff + this->meshes[g].loc_edges[i][1];
		}
		const bool has_tris = this->tri_offsets[g + 1] > toff;
		if (has_tris) {
			for (int i = 0; i < (int)this->meshes[g].loc_triangles.size(); i++) {
				F(toff + i, 0) = voff + this->meshes[g].loc_triangles[i][0];
				F(toff + i, 1) = voff + this->meshes[g].loc_triangles[i][1];
				F(toff + i, 2) = voff + this->meshes[g].loc_triangles[i][2];
			}
		}
	}

	this->V_combined = V_rest; // initial size
	this->ipc_mesh = ipc::CollisionMesh(V_rest, E, F);
	this->ipc_mesh.init_adjacencies();
	this->ipc_mesh_dirty = false;
	this->_rebuild_collision_filter();
}

void EnergyFrictionalContactIPC::_rebuild_collision_filter()
{
	// Snapshot disabled pairs by value (small).
	const unordered_array_set<int, 2> disabled = this->disabled_collision_pairs;

	// O(1) adjacency lookup: vector of unordered_sets, one per vertex.
	// adj_sets[vi].count(vj) is O(1) average and avoids any pair encoding.
	// Shared pointer avoids copying the structure into the std::function.
	auto adj_sets = std::make_shared<std::vector<std::unordered_set<size_t>>>();
	{
		const auto& adj = this->ipc_mesh.vertex_vertex_adjacencies();
		adj_sets->resize(adj.size());
		for (size_t vi = 0; vi < adj.size(); vi++) {
			(*adj_sets)[vi].reserve(adj[vi].size());
			for (ipc::index_t vj : adj[vi])
				(*adj_sets)[vi].insert((size_t)vj);
		}
	}

	ipc::CollisionFilter filter([this, disabled, adj_sets](size_t vi, size_t vj) {
		// Reject adjacent pairs (IPC barrier is undefined between connected elements).
		if ((*adj_sets)[vi].contains(vj))
			return false;
		// Reject explicitly disabled group pairs.
		const int ga = this->_vertex_to_group((int)vi);
		const int gb = this->_vertex_to_group((int)vj);
		const std::array<int, 2> pair = { std::min(ga, gb), std::max(ga, gb) };
		return !disabled.contains(pair);
	});

	this->ipc_mesh.can_collide = filter;
}

void EnergyFrictionalContactIPC::_update_vertices(Stark& stark, const double dt)
{
	#pragma omp parallel for schedule(static) num_threads(stark.settings.execution.n_threads)
	for (int group = 0; group < (int)this->meshes.size(); group++) {
		auto& mesh = this->meshes[group];
		const int n = (int)mesh.vertices.size();
		if (mesh.ps == PhysicalSystem::Deformable) {
			const DeformableBookkeeping& bk = this->group_to_deformable_bookkeeping[group];
			for (int i = 0; i < n; i++)
				mesh.vertices[i] = this->dyn->get_x1(this->dyn->get_global_index(mesh.idx_in_ps, bk.point_set_map[i]), dt);
		} else if (mesh.ps == PhysicalSystem::Rigidbody) {
			const Eigen::Vector3d t1 = time_integration(this->rb->t0[mesh.idx_in_ps], this->rb->v1[mesh.idx_in_ps], dt);
			const Eigen::Quaterniond q1 = quat_time_integration(this->rb->q0[mesh.idx_in_ps], this->rb->w1[mesh.idx_in_ps], dt);
			const Eigen::Matrix3d R1 = q1.toRotationMatrix();
			const RigidBodyBookkeeping& bk = this->group_to_rigidbody_bookkeeping[group];
			for (int i = 0; i < n; i++)
				mesh.vertices[i] = local_to_global_point(this->rigidbody_local_vertices.get(bk.rb_collision_group, i), R1, t1);
		} else {
			stark.context->output->print("stark error: Unknown physical system in EnergyFrictionalContactIPC.\n", symx::Verbosity::Summary);
			exit(-1);
		}
	}
}

void EnergyFrictionalContactIPC::_update_combined_V()
{
	const int n_verts = this->vertex_offsets.back();
	if (this->V_combined.rows() != n_verts)
		this->V_combined.resize(n_verts, 3);

	for (int g = 0; g < (int)this->meshes.size(); g++) {
		const int voff = this->vertex_offsets[g];
		for (int i = 0; i < (int)this->meshes[g].vertices.size(); i++)
			this->V_combined.row(voff + i) = this->meshes[g].vertices[i].transpose();
	}
}

int EnergyFrictionalContactIPC::_vertex_to_group(int global_v) const
{
	// upper_bound gives first offset > global_v; one step back is the group
	auto it = std::upper_bound(vertex_offsets.begin(), vertex_offsets.end(), global_v);
	return (int)std::distance(vertex_offsets.begin(), it) - 1;
}
int EnergyFrictionalContactIPC::_edge_to_group(int global_e) const
{
	auto it = std::upper_bound(edge_offsets.begin(), edge_offsets.end(), global_e);
	return (int)std::distance(edge_offsets.begin(), it) - 1;
}
int EnergyFrictionalContactIPC::_tri_to_group(int global_t) const
{
	auto it = std::upper_bound(tri_offsets.begin(), tri_offsets.end(), global_t);
	return (int)std::distance(tri_offsets.begin(), it) - 1;
}
bool EnergyFrictionalContactIPC::_is_pair_disabled(int group_a, int group_b) const
{
	const std::array<int, 2> pair = { std::min(group_a, group_b), std::max(group_a, group_b) };
	return this->disabled_collision_pairs.find(pair) != this->disabled_collision_pairs.end();
}

void EnergyFrictionalContactIPC::_validate_ogc_configuration() const
{
	if (this->global_params.contact_method != ContactMethod::OGC) return;
	if (!this->global_params.triangle_point_enabled
		|| !this->global_params.edge_edge_enabled) {
		throw std::invalid_argument(
			"EnergyFrictionalContactIPC: OGC requires both triangle-point and edge-edge contact modes.");
	}
}


/* ========================================================================================== */
/* ===================================  PROXIMITY DETECTION  ================================ */
/* ========================================================================================== */

void EnergyFrictionalContactIPC::_update_vertices_and_candidates(Stark& stark, double dt)
{
	this->_update_vertices(stark, dt);
	if (this->ipc_mesh_dirty) this->_build_ipc_mesh();
	this->_update_combined_V();
	const double max_dhat = *std::max_element(this->contact_thicknesses.begin(), this->contact_thicknesses.end());
	this->ipc_candidates.build(this->ipc_mesh, this->V_combined, 2.0 * max_dhat, this->broad_phase.get());
}

void EnergyFrictionalContactIPC::_run_proximity_and_update_contacts(Stark& stark, double dt)
{
	this->_update_vertices_and_candidates(stark, dt);

	// Clear contact connectivity
	this->contacts_deformables.clear();
	this->contacts_rb.clear();
	this->contacts_rb_deformables.clear();

	// --- Face-vertex candidates (point vs. triangle) ---
	if (this->global_params.triangle_point_enabled) {
		for (const auto& fvc : this->ipc_candidates.fv_candidates) {
			const int vert_group = this->_vertex_to_group(fvc.vertex_id);
			const int face_group = this->_tri_to_group(fvc.face_id);

			if (this->_is_pair_disabled(vert_group, face_group)) continue;

			const int vert_local = fvc.vertex_id - this->vertex_offsets[vert_group];
			const int face_local = fvc.face_id - this->tri_offsets[face_group];
			const auto& tri_lv   = this->meshes[face_group].loc_triangles[face_local];

			const Eigen::Vector3d& p  = this->meshes[vert_group].vertices[vert_local];
			const Eigen::Vector3d& t0 = this->meshes[face_group].vertices[tri_lv[0]];
			const Eigen::Vector3d& t1 = this->meshes[face_group].vertices[tri_lv[1]];
			const Eigen::Vector3d& t2 = this->meshes[face_group].vertices[tri_lv[2]];

			const auto dtype = ipc::point_triangle_distance_type(p, t0, t1, t2);
			const double d_sq = ipc::point_triangle_distance(p, t0, t1, t2, dtype);
			if (d_sq > std::pow(this->_get_contact_distance(vert_group, face_group), 2)) continue;

			// Build proxies
			const PhysicalSystem ps_v = this->meshes[vert_group].ps;
			const PhysicalSystem ps_f = this->meshes[face_group].ps;
			const int idx_v = this->meshes[vert_group].idx_in_ps;
			const int idx_f = this->meshes[face_group].idx_in_ps;

			const int gv = this->_local_to_ps_global_indices<1>(vert_group, {vert_local})[0];
			const std::array<int,3> gt = this->_local_to_ps_global_indices<3>(face_group, tri_lv);

			// Dispatch based on closest feature type
			// PP sub-type: point vs one vertex of triangle
			auto push_pp = [&](int gp) {
				if (ps_v == PhysicalSystem::Deformable && ps_f == PhysicalSystem::Deformable)
					contacts_deformables.point_triangle.point_point.push_back({vert_group, face_group, gv, gp});
				else if (ps_v == PhysicalSystem::Rigidbody && ps_f == PhysicalSystem::Rigidbody)
					contacts_rb.point_triangle.point_point.push_back({vert_group, face_group, idx_v, idx_f, gv, gp});
				else if (ps_v == PhysicalSystem::Rigidbody && ps_f == PhysicalSystem::Deformable)
					contacts_rb_deformables.point_triangle.rb_d_point_point.push_back({vert_group, face_group, idx_v, gv, gp});
				else
					contacts_rb_deformables.point_triangle.rb_d_point_point.push_back({face_group, vert_group, idx_f, gp, gv});
			};

			// PE sub-type: point vs one edge of triangle
			auto push_pe = [&](int ge1, int ge2) {
				if (ps_v == PhysicalSystem::Deformable && ps_f == PhysicalSystem::Deformable)
					contacts_deformables.point_triangle.point_edge.push_back({vert_group, face_group, gv, ge1, ge2});
				else if (ps_v == PhysicalSystem::Rigidbody && ps_f == PhysicalSystem::Rigidbody)
					contacts_rb.point_triangle.point_edge.push_back({vert_group, face_group, idx_v, idx_f, gv, ge1, ge2});
				else if (ps_v == PhysicalSystem::Rigidbody && ps_f == PhysicalSystem::Deformable)
					contacts_rb_deformables.point_triangle.rb_d_point_edge.push_back({vert_group, face_group, idx_v, gv, ge1, ge2});
				else
					contacts_rb_deformables.point_triangle.rb_d_edge_point.push_back({face_group, vert_group, idx_f, ge1, ge2, gv});
			};

			// PT sub-type: point vs triangle interior
			auto push_pt = [&]() {
				if (ps_v == PhysicalSystem::Deformable && ps_f == PhysicalSystem::Deformable)
					contacts_deformables.point_triangle.point_triangle.push_back({vert_group, face_group, gv, gt[0], gt[1], gt[2]});
				else if (ps_v == PhysicalSystem::Rigidbody && ps_f == PhysicalSystem::Rigidbody)
					contacts_rb.point_triangle.point_triangle.push_back({vert_group, face_group, idx_v, idx_f, gv, gt[0], gt[1], gt[2]});
				else if (ps_v == PhysicalSystem::Rigidbody && ps_f == PhysicalSystem::Deformable)
					contacts_rb_deformables.point_triangle.rb_d_point_triangle.push_back({vert_group, face_group, idx_v, gv, gt[0], gt[1], gt[2]});
				else
					contacts_rb_deformables.point_triangle.rb_d_triangle_point.push_back({face_group, vert_group, idx_f, gt[0], gt[1], gt[2], gv});
			};

			using PT = ipc::PointTriangleDistanceType;
			switch (dtype) {
				case PT::P_T0: push_pp(gt[0]); break;
				case PT::P_T1: push_pp(gt[1]); break;
				case PT::P_T2: push_pp(gt[2]); break;
				case PT::P_E0: push_pe(gt[0], gt[1]); break;
				case PT::P_E1: push_pe(gt[1], gt[2]); break;
				case PT::P_E2: push_pe(gt[2], gt[0]); break;
				case PT::P_T:  push_pt(); break;
				default: break;
			}
		}
	}

	// --- Edge-edge candidates ---
	if (this->global_params.edge_edge_enabled) {
		for (const auto& eec : this->ipc_candidates.ee_candidates) {
			const int ga = this->_edge_to_group(eec.edge0_id);
			const int gb = this->_edge_to_group(eec.edge1_id);

			if (this->_is_pair_disabled(ga, gb)) continue;

			const int la = eec.edge0_id - this->edge_offsets[ga];
			const int lb = eec.edge1_id - this->edge_offsets[gb];
			const auto& ea_lv = this->meshes[ga].loc_edges[la]; // [ea0_loc, ea1_loc]
			const auto& eb_lv = this->meshes[gb].loc_edges[lb]; // [eb0_loc, eb1_loc]

			const Eigen::Vector3d& ea0 = this->meshes[ga].vertices[ea_lv[0]];
			const Eigen::Vector3d& ea1 = this->meshes[ga].vertices[ea_lv[1]];
			const Eigen::Vector3d& eb0 = this->meshes[gb].vertices[eb_lv[0]];
			const Eigen::Vector3d& eb1 = this->meshes[gb].vertices[eb_lv[1]];

			const auto dtype = ipc::edge_edge_distance_type(ea0, ea1, eb0, eb1);
			const double d_sq = ipc::edge_edge_distance(ea0, ea1, eb0, eb1, dtype);
			if (d_sq > std::pow(this->_get_contact_distance(ga, gb), 2)) continue;

			const PhysicalSystem ps_a = this->meshes[ga].ps;
			const PhysicalSystem ps_b = this->meshes[gb].ps;
			const int idx_a = this->meshes[ga].idx_in_ps;
			const int idx_b = this->meshes[gb].idx_in_ps;

			const std::array<int, 2> gea = this->_local_to_ps_global_indices<2>(ga, ea_lv);
			const std::array<int, 2> geb = this->_local_to_ps_global_indices<2>(gb, eb_lv);

			auto push_ee_pp = [&](int gpa, int gpb) {

				if (ps_a == PhysicalSystem::Deformable && ps_b == PhysicalSystem::Deformable)
					contacts_deformables.edge_edge.point_point.push_back({ga, gb, gea[0], gea[1], gpa, geb[0], geb[1], gpb});
				else if (ps_a == PhysicalSystem::Rigidbody && ps_b == PhysicalSystem::Rigidbody)
					contacts_rb.edge_edge.point_point.push_back({ga, gb, idx_a, idx_b, gea[0], gea[1], gpa, geb[0], geb[1], gpb});
				else if (ps_a == PhysicalSystem::Rigidbody && ps_b == PhysicalSystem::Deformable)
					contacts_rb_deformables.edge_edge.rb_d_point_point.push_back({ga, gb, idx_a, gea[0], gea[1], gpa, geb[0], geb[1], gpb});
				else
					contacts_rb_deformables.edge_edge.rb_d_point_point.push_back({gb, ga, idx_b, geb[0], geb[1], gpb, gea[0], gea[1], gpa});
			};

			auto push_ee_pe_point_on_b = [&](int gp) {

				if (ps_a == PhysicalSystem::Deformable && ps_b == PhysicalSystem::Deformable)
					contacts_deformables.edge_edge.point_edge.push_back({gb, ga, geb[0], geb[1], gp, gea[0], gea[1]});
				else if (ps_a == PhysicalSystem::Rigidbody && ps_b == PhysicalSystem::Rigidbody)
					contacts_rb.edge_edge.point_edge.push_back({gb, ga, idx_b, idx_a, geb[0], geb[1], gp, gea[0], gea[1]});
				else if (ps_a == PhysicalSystem::Rigidbody && ps_b == PhysicalSystem::Deformable)
					contacts_rb_deformables.edge_edge.rb_d_edge_point.push_back({ga, gb, idx_a, gea[0], gea[1], geb[0], geb[1], gp});
				else
					contacts_rb_deformables.edge_edge.rb_d_point_edge.push_back({gb, ga, idx_b, geb[0], geb[1], gp, gea[0], gea[1]});
			};

			auto push_ee_pe_point_on_a = [&](int gp) {

				if (ps_a == PhysicalSystem::Deformable && ps_b == PhysicalSystem::Deformable)
					contacts_deformables.edge_edge.point_edge.push_back({ga, gb, gea[0], gea[1], gp, geb[0], geb[1]});
				else if (ps_a == PhysicalSystem::Rigidbody && ps_b == PhysicalSystem::Rigidbody)
					contacts_rb.edge_edge.point_edge.push_back({ga, gb, idx_a, idx_b, gea[0], gea[1], gp, geb[0], geb[1]});
				else if (ps_a == PhysicalSystem::Rigidbody && ps_b == PhysicalSystem::Deformable)
					contacts_rb_deformables.edge_edge.rb_d_point_edge.push_back({ga, gb, idx_a, gea[0], gea[1], gp, geb[0], geb[1]});
				else
					contacts_rb_deformables.edge_edge.rb_d_edge_point.push_back({gb, ga, idx_b, geb[0], geb[1], gea[0], gea[1], gp});
			};

			// EE EE: interior of A closest to interior of B
			auto push_ee_ee = [&]() {

				if (ps_a == PhysicalSystem::Deformable && ps_b == PhysicalSystem::Deformable)
					contacts_deformables.edge_edge.edge_edge.push_back({ga, gb, gea[0], gea[1], geb[0], geb[1]});
				else if (ps_a == PhysicalSystem::Rigidbody && ps_b == PhysicalSystem::Rigidbody)
					contacts_rb.edge_edge.edge_edge.push_back({ga, gb, idx_a, idx_b, gea[0], gea[1], geb[0], geb[1]});
				else if (ps_a == PhysicalSystem::Rigidbody && ps_b == PhysicalSystem::Deformable)
					contacts_rb_deformables.edge_edge.rb_d_edge_edge.push_back({ga, gb, idx_a, gea[0], gea[1], geb[0], geb[1]});
				else
					contacts_rb_deformables.edge_edge.rb_d_edge_edge.push_back({gb, ga, idx_b, geb[0], geb[1], gea[0], gea[1]});
			};

			using EE = ipc::EdgeEdgeDistanceType;
			switch (dtype) {
				case EE::EA0_EB0: push_ee_pp(gea[0], geb[0]); break;
				case EE::EA0_EB1: push_ee_pp(gea[0], geb[1]); break;
				case EE::EA1_EB0: push_ee_pp(gea[1], geb[0]); break;
				case EE::EA1_EB1: push_ee_pp(gea[1], geb[1]); break;
				case EE::EA_EB0:  push_ee_pe_point_on_b(geb[0]); break;
				case EE::EA_EB1:  push_ee_pe_point_on_b(geb[1]); break;
				case EE::EA0_EB:  push_ee_pe_point_on_a(gea[0]); break;
				case EE::EA1_EB:  push_ee_pe_point_on_a(gea[1]); break;
				case EE::EA_EB:   push_ee_ee(); break;
				default: break;
			}
		}
	}
}

void EnergyFrictionalContactIPC::_run_proximity_and_update_friction(Stark& stark)
{
	// Vertices and ipc_candidates already built by _update_vertices_and_candidates.
	this->friction_deformables.clear();
	this->friction_rb.clear();
	this->friction_rb_deformables.clear();

	const double k = this->contact_stiffness;

	auto fn_force = [&](double d, int ga, int gb) {
		return this->_barrier_force(d, this->_get_contact_distance(ga, gb), k);
	};

	// Helper lambdas to build friction contact data (identical to EnergyFrictionalContact)
	auto point_point = [&](FrictionContact& contact, double mu, double d, int ga, int lva, int gb, int lvb) {
		const Eigen::Vector3d& P = this->meshes[ga].vertices[lva];
		const Eigen::Vector3d& Q = this->meshes[gb].vertices[lvb];
		contact.T.push_back(projection_matrix_point_point(P, Q));
		contact.mu.push_back(mu);
		contact.fn.push_back(fn_force(d, ga, gb));
	};
	auto point_edge = [&](FrictionPointEdge& friction, double mu, double d, int ga, int lva, int gb, int lve0, int lve1) {
		const Eigen::Vector3d& P  = this->meshes[ga].vertices[lva];
		const Eigen::Vector3d& E0 = this->meshes[gb].vertices[lve0];
		const Eigen::Vector3d& E1 = this->meshes[gb].vertices[lve1];
		friction.bary.push_back(barycentric_point_edge(P, E0, E1));
		friction.contact.T.push_back(projection_matrix_point_edge(P, E0, E1));
		friction.contact.mu.push_back(mu);
		friction.contact.fn.push_back(fn_force(d, ga, gb));
	};
	auto point_triangle = [&](FrictionPointTriangle& friction, double mu, double d, int ga, int lva, int gb, int lv0, int lv1, int lv2) {
		const Eigen::Vector3d& P  = this->meshes[ga].vertices[lva];
		const Eigen::Vector3d& T0 = this->meshes[gb].vertices[lv0];
		const Eigen::Vector3d& T1 = this->meshes[gb].vertices[lv1];
		const Eigen::Vector3d& T2 = this->meshes[gb].vertices[lv2];
		friction.bary.push_back(barycentric_point_triangle(P, T0, T1, T2));
		friction.contact.T.push_back(projection_matrix_triangle(T0, T1, T2));
		friction.contact.mu.push_back(mu);
		friction.contact.fn.push_back(fn_force(d, ga, gb));
	};
	auto edge_edge = [&](FrictionEdgeEdge& friction, double mu, double d, int ga, int lea0, int lea1, int gb, int leb0, int leb1) {
		const Eigen::Vector3d& EA0 = this->meshes[ga].vertices[lea0];
		const Eigen::Vector3d& EA1 = this->meshes[ga].vertices[lea1];
		const Eigen::Vector3d& EB0 = this->meshes[gb].vertices[leb0];
		const Eigen::Vector3d& EB1 = this->meshes[gb].vertices[leb1];
		friction.bary.push_back(barycentric_edge_edge(EA0, EA1, EB0, EB1));
		friction.contact.T.push_back(projection_matrix_edge_edge(EA0, EA1, EB0, EB1));
		friction.contact.mu.push_back(mu);
		friction.contact.fn.push_back(fn_force(d, ga, gb));
	};

	// FV candidates → friction
	if (this->global_params.triangle_point_enabled) {
		for (const auto& fvc : this->ipc_candidates.fv_candidates) {
			const int vert_group = this->_vertex_to_group(fvc.vertex_id);
			const int face_group = this->_tri_to_group(fvc.face_id);
			if (this->_is_pair_disabled(vert_group, face_group)) continue;
			const double mu = this->_get_friction(vert_group, face_group);
			if (mu == 0.0) continue;

			const int vert_local = fvc.vertex_id - this->vertex_offsets[vert_group];
			const int face_local = fvc.face_id - this->tri_offsets[face_group];
			const auto& tri_lv = this->meshes[face_group].loc_triangles[face_local];

			const Eigen::Vector3d& p  = this->meshes[vert_group].vertices[vert_local];
			const Eigen::Vector3d& t0 = this->meshes[face_group].vertices[tri_lv[0]];
			const Eigen::Vector3d& t1 = this->meshes[face_group].vertices[tri_lv[1]];
			const Eigen::Vector3d& t2 = this->meshes[face_group].vertices[tri_lv[2]];

			const auto dtype = ipc::point_triangle_distance_type(p, t0, t1, t2);
			const double d_sq = ipc::point_triangle_distance(p, t0, t1, t2, dtype);
			if (d_sq > std::pow(this->_get_contact_distance(vert_group, face_group), 2)) continue;
			const double d = std::sqrt(d_sq);

			const PhysicalSystem ps_v = this->meshes[vert_group].ps;
			const PhysicalSystem ps_f = this->meshes[face_group].ps;
			const int idx_v = this->meshes[vert_group].idx_in_ps;
			const int idx_f = this->meshes[face_group].idx_in_ps;

			const int gv = this->_local_to_ps_global_indices<1>(vert_group, {vert_local})[0];
			const std::array<int,3> gt = this->_local_to_ps_global_indices<3>(face_group, tri_lv);


			using PT = ipc::PointTriangleDistanceType;

			auto do_pp = [&](int index) {
				if (ps_v == PhysicalSystem::Deformable && ps_f == PhysicalSystem::Deformable) {
					friction_deformables.point_point.conn.numbered_push_back({gv, gt[index]});
					point_point(friction_deformables.point_point.contact, mu, d, vert_group, vert_local, face_group, tri_lv[index]);
				} else if (ps_v == PhysicalSystem::Rigidbody && ps_f == PhysicalSystem::Rigidbody) {
					friction_rb.point_point.conn.numbered_push_back({idx_v, idx_f, gv, gt[index]});
					point_point(friction_rb.point_point.contact, mu, d, vert_group, vert_local, face_group, tri_lv[index]);
				} else if (ps_v == PhysicalSystem::Rigidbody && ps_f == PhysicalSystem::Deformable) {
					friction_rb_deformables.point_point.conn.numbered_push_back({idx_v, gv, gt[index]});
					point_point(friction_rb_deformables.point_point.contact, mu, d, vert_group, vert_local, face_group, tri_lv[index]);
				} else {
					friction_rb_deformables.point_point.conn.numbered_push_back({idx_f, gt[index], gv});
					point_point(friction_rb_deformables.point_point.contact, mu, d, vert_group, vert_local, face_group, tri_lv[index]);
				}
			};

			auto do_pe = [&](int i1, int i2) {
				if (ps_v == PhysicalSystem::Deformable && ps_f == PhysicalSystem::Deformable) {
					friction_deformables.point_edge.conn.numbered_push_back({gv, gt[i1], gt[i2]});
					point_edge(friction_deformables.point_edge.data, mu, d, vert_group, vert_local, face_group, tri_lv[i1], tri_lv[i2]);
				} else if (ps_v == PhysicalSystem::Rigidbody && ps_f == PhysicalSystem::Rigidbody) {
					friction_rb.point_edge.conn.numbered_push_back({idx_v, idx_f, gv, gt[i1], gt[i2]});
					point_edge(friction_rb.point_edge.data, mu, d, vert_group, vert_local, face_group, tri_lv[i1], tri_lv[i2]);
				} else if (ps_v == PhysicalSystem::Rigidbody && ps_f == PhysicalSystem::Deformable) {
					friction_rb_deformables.point_edge.conn.numbered_push_back({idx_v, gv, gt[i1], gt[i2]});
					point_edge(friction_rb_deformables.point_edge.data, mu, d, vert_group, vert_local, face_group, tri_lv[i1], tri_lv[i2]);
				} else {
					friction_rb_deformables.edge_point.conn.numbered_push_back({idx_f, gt[i1], gt[i2], gv});
					point_edge(friction_rb_deformables.edge_point.data, mu, d, vert_group, vert_local, face_group, tri_lv[i1], tri_lv[i2]);
				}
			};

			auto do_pt = [&]() {
				if (ps_v == PhysicalSystem::Deformable && ps_f == PhysicalSystem::Deformable) {
					friction_deformables.point_triangle.conn.numbered_push_back({gv, gt[0], gt[1], gt[2]});
					point_triangle(friction_deformables.point_triangle.data, mu, d, vert_group, vert_local, face_group, tri_lv[0], tri_lv[1], tri_lv[2]);
				} else if (ps_v == PhysicalSystem::Rigidbody && ps_f == PhysicalSystem::Rigidbody) {
					friction_rb.point_triangle.conn.numbered_push_back({idx_v, idx_f, gv, gt[0], gt[1], gt[2]});
					point_triangle(friction_rb.point_triangle.data, mu, d, vert_group, vert_local, face_group, tri_lv[0], tri_lv[1], tri_lv[2]);
				} else if (ps_v == PhysicalSystem::Rigidbody && ps_f == PhysicalSystem::Deformable) {
					friction_rb_deformables.point_triangle.conn.numbered_push_back({idx_v, gv, gt[0], gt[1], gt[2]});
					point_triangle(friction_rb_deformables.point_triangle.data, mu, d, vert_group, vert_local, face_group, tri_lv[0], tri_lv[1], tri_lv[2]);
				} else {
					friction_rb_deformables.triangle_point.conn.numbered_push_back({idx_f, gt[0], gt[1], gt[2], gv});
					point_triangle(friction_rb_deformables.triangle_point.data, mu, d, vert_group, vert_local, face_group, tri_lv[0], tri_lv[1], tri_lv[2]);
				}
			};

			switch (dtype) {
				case PT::P_T0: do_pp(0); break;
				case PT::P_T1: do_pp(1); break;
				case PT::P_T2: do_pp(2); break;
				case PT::P_E0: do_pe(0, 1); break;
				case PT::P_E1: do_pe(1, 2); break;
				case PT::P_E2: do_pe(2, 0); break;
				case PT::P_T:  do_pt(); break;
				default: break;
			}
		}
	}

	// EE candidates → friction
	if (this->global_params.edge_edge_enabled) {
		for (const auto& eec : this->ipc_candidates.ee_candidates) {
			const int ga = this->_edge_to_group(eec.edge0_id);
			const int gb = this->_edge_to_group(eec.edge1_id);
			if (this->_is_pair_disabled(ga, gb)) continue;
			const double mu = this->_get_friction(ga, gb);
			if (mu == 0.0) continue;

			const int la = eec.edge0_id - this->edge_offsets[ga];
			const int lb = eec.edge1_id - this->edge_offsets[gb];
			const auto& ea_lv = this->meshes[ga].loc_edges[la];
			const auto& eb_lv = this->meshes[gb].loc_edges[lb];

			const Eigen::Vector3d& ea0 = this->meshes[ga].vertices[ea_lv[0]];
			const Eigen::Vector3d& ea1 = this->meshes[ga].vertices[ea_lv[1]];
			const Eigen::Vector3d& eb0 = this->meshes[gb].vertices[eb_lv[0]];
			const Eigen::Vector3d& eb1 = this->meshes[gb].vertices[eb_lv[1]];

			const auto dtype = ipc::edge_edge_distance_type(ea0, ea1, eb0, eb1);
			const double d_sq = ipc::edge_edge_distance(ea0, ea1, eb0, eb1, dtype);
			if (d_sq > std::pow(this->_get_contact_distance(ga, gb), 2)) continue;
			const double d = std::sqrt(d_sq);

			const PhysicalSystem ps_a = this->meshes[ga].ps;
			const PhysicalSystem ps_b = this->meshes[gb].ps;
			const int idx_a = this->meshes[ga].idx_in_ps;
			const int idx_b = this->meshes[gb].idx_in_ps;

			auto gal = [&](int loc) { return this->_local_to_ps_global_indices<1>(ga, {loc})[0]; };
			auto gbl = [&](int loc) { return this->_local_to_ps_global_indices<1>(gb, {loc})[0]; };

			const int gea0 = gal(ea_lv[0]), gea1 = gal(ea_lv[1]);
			const int geb0 = gbl(eb_lv[0]), geb1 = gbl(eb_lv[1]);

			if (ps_a == PhysicalSystem::Deformable && ps_b == PhysicalSystem::Deformable) {
				friction_deformables.edge_edge.conn.numbered_push_back({gea0, gea1, geb0, geb1});
				edge_edge(friction_deformables.edge_edge.data, mu, d, ga, ea_lv[0], ea_lv[1], gb, eb_lv[0], eb_lv[1]);
			} else if (ps_a == PhysicalSystem::Rigidbody && ps_b == PhysicalSystem::Rigidbody) {
				friction_rb.edge_edge.conn.numbered_push_back({idx_a, idx_b, gea0, gea1, geb0, geb1});
				edge_edge(friction_rb.edge_edge.data, mu, d, ga, ea_lv[0], ea_lv[1], gb, eb_lv[0], eb_lv[1]);
			} else if (ps_a == PhysicalSystem::Rigidbody && ps_b == PhysicalSystem::Deformable) {
				friction_rb_deformables.edge_edge.conn.numbered_push_back({idx_a, gea0, gea1, geb0, geb1});
				edge_edge(friction_rb_deformables.edge_edge.data, mu, d, ga, ea_lv[0], ea_lv[1], gb, eb_lv[0], eb_lv[1]);
			} else {
				friction_rb_deformables.edge_edge.conn.numbered_push_back({idx_b, geb0, geb1, gea0, gea1});
				edge_edge(friction_rb_deformables.edge_edge.data, mu, d, gb, eb_lv[0], eb_lv[1], ga, ea_lv[0], ea_lv[1]);
			}
		}
	}
}

void EnergyFrictionalContactIPC::_run_ogc_proximity_and_update_friction(Stark& stark)
{
	this->_update_vertices_and_candidates(stark, 0.0);

	ipc::NormalCollisions lagged_collisions;
	lagged_collisions.set_collision_set_type(ipc::NormalCollisions::CollisionSetType::OGC);
	const double max_thickness = *std::max_element(
		this->contact_thicknesses.begin(), this->contact_thicknesses.end());
	lagged_collisions.build(
		this->ipc_candidates, this->ipc_mesh, this->V_combined,
		2.0 * max_thickness, 0.0);

	auto cache_point_point = [&](FrictionContact& contact, double mu, double fn,
		int ga, int lva, int gb, int lvb) {
		const auto& p = this->meshes[ga].vertices[lva];
		const auto& q = this->meshes[gb].vertices[lvb];
		contact.T.push_back(projection_matrix_point_point(p, q));
		contact.mu.push_back(mu);
		contact.fn.push_back(fn);
	};
	auto cache_point_edge = [&](FrictionPointEdge& friction, double mu, double fn,
		int gv, int lv, int ge, int le0, int le1) {
		const auto& p = this->meshes[gv].vertices[lv];
		const auto& e0 = this->meshes[ge].vertices[le0];
		const auto& e1 = this->meshes[ge].vertices[le1];
		friction.bary.push_back(barycentric_point_edge(p, e0, e1));
		friction.contact.T.push_back(projection_matrix_point_edge(p, e0, e1));
		friction.contact.mu.push_back(mu);
		friction.contact.fn.push_back(fn);
	};
	auto cache_point_triangle = [&](FrictionPointTriangle& friction, double mu, double fn,
		int gv, int lv, int gf, int lt0, int lt1, int lt2) {
		const auto& p = this->meshes[gv].vertices[lv];
		const auto& t0 = this->meshes[gf].vertices[lt0];
		const auto& t1 = this->meshes[gf].vertices[lt1];
		const auto& t2 = this->meshes[gf].vertices[lt2];
		friction.bary.push_back(barycentric_point_triangle(p, t0, t1, t2));
		friction.contact.T.push_back(projection_matrix_triangle(t0, t1, t2));
		friction.contact.mu.push_back(mu);
		friction.contact.fn.push_back(fn);
	};
	auto cache_edge_edge = [&](FrictionEdgeEdge& friction, double mu, double fn,
		int ga, int la0, int la1, int gb, int lb0, int lb1) {
		const auto& a0 = this->meshes[ga].vertices[la0];
		const auto& a1 = this->meshes[ga].vertices[la1];
		const auto& b0 = this->meshes[gb].vertices[lb0];
		const auto& b1 = this->meshes[gb].vertices[lb1];
		friction.bary.push_back(barycentric_edge_edge(a0, a1, b0, b1));
		friction.contact.T.push_back(projection_matrix_edge_edge(a0, a1, b0, b1));
		friction.contact.mu.push_back(mu);
		friction.contact.fn.push_back(fn);
	};

	auto append_vv = [&](int ga, int lva, int gb, int lvb, double mu, double fn) {
		const int a = this->_local_to_ps_global_indices<1>(ga, {lva})[0];
		const int b = this->_local_to_ps_global_indices<1>(gb, {lvb})[0];
		const auto psa = this->meshes[ga].ps;
		const auto psb = this->meshes[gb].ps;
		if (psa == PhysicalSystem::Deformable && psb == PhysicalSystem::Deformable) {
			this->friction_deformables.point_point.conn.numbered_push_back({a, b});
			cache_point_point(this->friction_deformables.point_point.contact, mu, fn, ga, lva, gb, lvb);
		} else if (psa == PhysicalSystem::Rigidbody && psb == PhysicalSystem::Rigidbody) {
			this->friction_rb.point_point.conn.numbered_push_back(
				{this->meshes[ga].idx_in_ps, this->meshes[gb].idx_in_ps, a, b});
			cache_point_point(this->friction_rb.point_point.contact, mu, fn, ga, lva, gb, lvb);
		} else if (psa == PhysicalSystem::Rigidbody) {
			this->friction_rb_deformables.point_point.conn.numbered_push_back(
				{this->meshes[ga].idx_in_ps, a, b});
			cache_point_point(this->friction_rb_deformables.point_point.contact, mu, fn, ga, lva, gb, lvb);
		} else {
			this->friction_rb_deformables.point_point.conn.numbered_push_back(
				{this->meshes[gb].idx_in_ps, b, a});
			cache_point_point(this->friction_rb_deformables.point_point.contact, mu, fn, gb, lvb, ga, lva);
		}
	};

	auto append_ev = [&](int gv, int lv, int ge, int le0, int le1, double mu, double fn) {
		const int p = this->_local_to_ps_global_indices<1>(gv, {lv})[0];
		const auto e = this->_local_to_ps_global_indices<2>(ge, {le0, le1});
		const auto psv = this->meshes[gv].ps;
		const auto pse = this->meshes[ge].ps;
		if (psv == PhysicalSystem::Deformable && pse == PhysicalSystem::Deformable) {
			this->friction_deformables.point_edge.conn.numbered_push_back({p, e[0], e[1]});
			cache_point_edge(this->friction_deformables.point_edge.data, mu, fn, gv, lv, ge, le0, le1);
		} else if (psv == PhysicalSystem::Rigidbody && pse == PhysicalSystem::Rigidbody) {
			this->friction_rb.point_edge.conn.numbered_push_back(
				{this->meshes[gv].idx_in_ps, this->meshes[ge].idx_in_ps, p, e[0], e[1]});
			cache_point_edge(this->friction_rb.point_edge.data, mu, fn, gv, lv, ge, le0, le1);
		} else if (psv == PhysicalSystem::Rigidbody) {
			this->friction_rb_deformables.point_edge.conn.numbered_push_back(
				{this->meshes[gv].idx_in_ps, p, e[0], e[1]});
			cache_point_edge(this->friction_rb_deformables.point_edge.data, mu, fn, gv, lv, ge, le0, le1);
		} else {
			this->friction_rb_deformables.edge_point.conn.numbered_push_back(
				{this->meshes[ge].idx_in_ps, e[0], e[1], p});
			cache_point_edge(this->friction_rb_deformables.edge_point.data, mu, fn, gv, lv, ge, le0, le1);
		}
	};

	auto append_fv = [&](int gv, int lv, int gf, int lt0, int lt1, int lt2, double mu, double fn) {
		const int p = this->_local_to_ps_global_indices<1>(gv, {lv})[0];
		const auto t = this->_local_to_ps_global_indices<3>(gf, {lt0, lt1, lt2});
		const auto psv = this->meshes[gv].ps;
		const auto psf = this->meshes[gf].ps;
		if (psv == PhysicalSystem::Deformable && psf == PhysicalSystem::Deformable) {
			this->friction_deformables.point_triangle.conn.numbered_push_back({p, t[0], t[1], t[2]});
			cache_point_triangle(this->friction_deformables.point_triangle.data, mu, fn, gv, lv, gf, lt0, lt1, lt2);
		} else if (psv == PhysicalSystem::Rigidbody && psf == PhysicalSystem::Rigidbody) {
			this->friction_rb.point_triangle.conn.numbered_push_back(
				{this->meshes[gv].idx_in_ps, this->meshes[gf].idx_in_ps, p, t[0], t[1], t[2]});
			cache_point_triangle(this->friction_rb.point_triangle.data, mu, fn, gv, lv, gf, lt0, lt1, lt2);
		} else if (psv == PhysicalSystem::Rigidbody) {
			this->friction_rb_deformables.point_triangle.conn.numbered_push_back(
				{this->meshes[gv].idx_in_ps, p, t[0], t[1], t[2]});
			cache_point_triangle(this->friction_rb_deformables.point_triangle.data, mu, fn, gv, lv, gf, lt0, lt1, lt2);
		} else {
			this->friction_rb_deformables.triangle_point.conn.numbered_push_back(
				{this->meshes[gf].idx_in_ps, t[0], t[1], t[2], p});
			cache_point_triangle(this->friction_rb_deformables.triangle_point.data, mu, fn, gv, lv, gf, lt0, lt1, lt2);
		}
	};

	auto append_ee = [&](int ga, int la0, int la1, int gb, int lb0, int lb1, double mu, double fn) {
		const auto a = this->_local_to_ps_global_indices<2>(ga, {la0, la1});
		const auto b = this->_local_to_ps_global_indices<2>(gb, {lb0, lb1});
		const auto psa = this->meshes[ga].ps;
		const auto psb = this->meshes[gb].ps;
		if (psa == PhysicalSystem::Deformable && psb == PhysicalSystem::Deformable) {
			this->friction_deformables.edge_edge.conn.numbered_push_back({a[0], a[1], b[0], b[1]});
			cache_edge_edge(this->friction_deformables.edge_edge.data, mu, fn, ga, la0, la1, gb, lb0, lb1);
		} else if (psa == PhysicalSystem::Rigidbody && psb == PhysicalSystem::Rigidbody) {
			this->friction_rb.edge_edge.conn.numbered_push_back(
				{this->meshes[ga].idx_in_ps, this->meshes[gb].idx_in_ps, a[0], a[1], b[0], b[1]});
			cache_edge_edge(this->friction_rb.edge_edge.data, mu, fn, ga, la0, la1, gb, lb0, lb1);
		} else if (psa == PhysicalSystem::Rigidbody) {
			this->friction_rb_deformables.edge_edge.conn.numbered_push_back(
				{this->meshes[ga].idx_in_ps, a[0], a[1], b[0], b[1]});
			cache_edge_edge(this->friction_rb_deformables.edge_edge.data, mu, fn, ga, la0, la1, gb, lb0, lb1);
		} else {
			this->friction_rb_deformables.edge_edge.conn.numbered_push_back(
				{this->meshes[gb].idx_in_ps, b[0], b[1], a[0], a[1]});
			cache_edge_edge(this->friction_rb_deformables.edge_edge.data, mu, fn, gb, lb0, lb1, ga, la0, la1);
		}
	};

	const double k = this->contact_stiffness;
	for (const auto& collision : lagged_collisions.vv_collisions) {
		const int ga = this->_vertex_to_group(collision.vertex0_id);
		const int gb = this->_vertex_to_group(collision.vertex1_id);
		if (this->_is_pair_disabled(ga, gb)) continue;
		const double mu = this->_get_friction(ga, gb);
		if (mu == 0.0) continue;
		const int la = collision.vertex0_id - this->vertex_offsets[ga];
		const int lb = collision.vertex1_id - this->vertex_offsets[gb];
		const double dhat = this->_get_contact_distance(ga, gb);
		const double d = (this->meshes[ga].vertices[la] - this->meshes[gb].vertices[lb]).norm();
		if (d >= dhat) continue;
		append_vv(ga, la, gb, lb, mu, collision.weight * this->_barrier_force(d, dhat, k));
	}

	for (const auto& collision : lagged_collisions.ev_collisions) {
		const int gv = this->_vertex_to_group(collision.vertex_id);
		const int ge = this->_edge_to_group(collision.edge_id);
		if (this->_is_pair_disabled(gv, ge)) continue;
		const double mu = this->_get_friction(gv, ge);
		if (mu == 0.0) continue;
		const int lv = collision.vertex_id - this->vertex_offsets[gv];
		const int le = collision.edge_id - this->edge_offsets[ge];
		const auto& edge = this->meshes[ge].loc_edges[le];
		const double dhat = this->_get_contact_distance(gv, ge);
		const double d = std::sqrt(std::max(0.0, ipc::point_edge_distance(
			this->meshes[gv].vertices[lv], this->meshes[ge].vertices[edge[0]],
			this->meshes[ge].vertices[edge[1]], ipc::PointEdgeDistanceType::P_E)));
		if (d >= dhat) continue;
		append_ev(gv, lv, ge, edge[0], edge[1], mu,
			collision.weight * this->_barrier_force(d, dhat, k));
	}

	for (const auto& collision : lagged_collisions.fv_collisions) {
		const int gv = this->_vertex_to_group(collision.vertex_id);
		const int gf = this->_tri_to_group(collision.face_id);
		if (this->_is_pair_disabled(gv, gf)) continue;
		const double mu = this->_get_friction(gv, gf);
		if (mu == 0.0) continue;
		const int lv = collision.vertex_id - this->vertex_offsets[gv];
		const int lf = collision.face_id - this->tri_offsets[gf];
		const auto& face = this->meshes[gf].loc_triangles[lf];
		const double dhat = this->_get_contact_distance(gv, gf);
		const double d = std::sqrt(std::max(0.0, ipc::point_triangle_distance(
			this->meshes[gv].vertices[lv], this->meshes[gf].vertices[face[0]],
			this->meshes[gf].vertices[face[1]], this->meshes[gf].vertices[face[2]],
			ipc::PointTriangleDistanceType::P_T)));
		if (d >= dhat) continue;
		append_fv(gv, lv, gf, face[0], face[1], face[2], mu,
			collision.weight * this->_barrier_force(d, dhat, k));
	}

	for (const auto& collision : lagged_collisions.ee_collisions) {
		const int ga = this->_edge_to_group(collision.edge0_id);
		const int gb = this->_edge_to_group(collision.edge1_id);
		if (this->_is_pair_disabled(ga, gb)) continue;
		const double mu = this->_get_friction(ga, gb);
		if (mu == 0.0) continue;
		const auto& a = this->meshes[ga].loc_edges[collision.edge0_id - this->edge_offsets[ga]];
		const auto& b = this->meshes[gb].loc_edges[collision.edge1_id - this->edge_offsets[gb]];
		const auto& a0 = this->meshes[ga].vertices[a[0]];
		const auto& a1 = this->meshes[ga].vertices[a[1]];
		const auto& b0 = this->meshes[gb].vertices[b[0]];
		const auto& b1 = this->meshes[gb].vertices[b[1]];
		if ((a1 - a0).cross(b1 - b0).squaredNorm() < collision.eps_x) continue;
		const double dhat = this->_get_contact_distance(ga, gb);
		const double d = std::sqrt(std::max(0.0,
			ipc::edge_edge_distance(a0, a1, b0, b1, collision.dtype)));
		if (d >= dhat) continue;
		append_ee(ga, a[0], a[1], gb, b[0], b[1], mu,
			collision.weight * this->_barrier_force(d, dhat, k));
	}
}


/* ========================================================================================== */
/* ===================================  SYMX CALLBACKS  ===================================== */
/* ========================================================================================== */

void EnergyFrictionalContactIPC::_before_energy_evaluation__update_contacts(Stark& stark)
{
	if (this->global_params.contact_method == ContactMethod::OGC) return;
	if (!this->global_params.collisions_enabled) return;
	if (this->is_empty()) return;
	this->_run_proximity_and_update_contacts(stark, stark.dt);
}

void EnergyFrictionalContactIPC::_before_time_step__update_friction_contacts(Stark& stark)
{
	if (this->global_params.contact_method == ContactMethod::OGC) {
		if (this->ogc_trust_region) this->ogc_trust_region->should_update_trust_region = true;
		this->friction_deformables.clear();
		this->friction_rb.clear();
		this->friction_rb_deformables.clear();
		if (!this->global_params.collisions_enabled) return;
		if (!this->global_params.friction_enabled) return;
		if (this->is_empty()) return;
		this->_run_ogc_proximity_and_update_friction(stark);
		return;
	}
	if (!this->global_params.collisions_enabled) return;
	if (!this->global_params.friction_enabled) return;
	if (this->is_empty()) return;

	// Build candidates at dt=0 for lagged friction contacts; skip contact population
	// since it is immediately rebuilt on the first Newton iteration.
	this->_update_vertices_and_candidates(stark, 0.0);
	this->_run_proximity_and_update_friction(stark);
}

bool EnergyFrictionalContactIPC::_is_intermediate_state_valid(Stark& stark, bool is_initial_check)
{
	if (!this->global_params.collisions_enabled) return true;
	if (!this->global_params.intersection_test_enabled) return true;
	if (this->is_empty()) return true;

	this->_update_vertices(stark, is_initial_check ? 0.0 : stark.dt);
	if (this->ipc_mesh_dirty) this->_build_ipc_mesh();
	this->_update_combined_V();

	const bool has_intersections = ipc::has_intersections(this->ipc_mesh, this->V_combined, this->broad_phase.get());
	if (has_intersections) {
		if (is_initial_check) {
			stark.context->output->print_with_new_line("IPC-Toolkit: intersection detected.", symx::Verbosity::Summary);
		}
		return false;
	}
	return true;
}

bool EnergyFrictionalContactIPC::_is_state_valid(Stark& stark)
{
	return this->_is_intermediate_state_valid(stark, true);
}

void EnergyFrictionalContactIPC::_on_intermediate_state_invalid(Stark& stark)
{
	const double old_stiffness = this->contact_stiffness;
	this->contact_stiffness *= 2.0;
	const double new_stiffness = this->contact_stiffness;
	stark.context->output->print_with_new_line(fmt::format("Penetration couldn't be avoided. Contact stiffness hardened from {:.1e} to {:.1e}.\n", old_stiffness, new_stiffness), symx::Verbosity::Summary);
}

void EnergyFrictionalContactIPC::_on_time_step_accepted(Stark& stark)
{
	this->contact_stiffness = std::max(this->global_params.min_contact_stiffness, 0.99 * this->contact_stiffness);
}

bool EnergyFrictionalContactIPC::_should_continue_execution(Stark& stark)
{
	if (!this->global_params.collisions_enabled) return true;
	if (this->contact_stiffness > this->global_params.max_contact_stiffness) {
		stark.context->logger->set("success", 0);
		stark.context->output->print_with_new_line(fmt::format("Contact stiffness exceeded maximum value of {:.1e}.", this->global_params.max_contact_stiffness), symx::Verbosity::Summary);
		return false;
	}
	return true;
}

double EnergyFrictionalContactIPC::_ccd_max_allowed_step(Stark& stark, const Eigen::VectorXd& x, const Eigen::VectorXd& du)
{
	if (this->global_params.contact_method == ContactMethod::OGC
		&& !this->ogc_use_ccd_for_step) return 1.0;
	if (!this->global_params.collisions_enabled) return 1.0;
	if (this->is_empty()) return 1.0;
	if (this->ipc_mesh_dirty) return 1.0; // mesh not yet built

	// V0 = current positions (already in V_combined from last proximity call)
	const Eigen::MatrixXd V0 = this->V_combined;

	// Find DOF offsets for deformable v1 and rigid v1/w1 in the global DOF vector
	// We cache these on first call
	static int v1_offset = -1, rv1_offset = -1, rw1_offset = -1;
	if (v1_offset < 0) {
		const auto& offsets = stark.global_potential->get_dofs_offsets();
		const int n_sets = stark.global_potential->get_n_dof_sets();
		for (int i = 0; i < n_sets; i++) {
			const std::string label = stark.global_potential->get_dof_label(i);
			if (label == "soft.v1"  && v1_offset  < 0) v1_offset  = offsets[i];
			if (label == "rigid.v1" && rv1_offset < 0) rv1_offset = offsets[i];
			if (label == "rigid.w1" && rw1_offset < 0) rw1_offset = offsets[i];
		}
	}

	// Build V1: positions after full Newton step
	Eigen::MatrixXd V1 = V0;
	const double dt = stark.dt;

	for (int g = 0; g < (int)this->meshes.size(); g++) {
		const int voff = this->vertex_offsets[g];
		const int n = (int)this->meshes[g].vertices.size();

		if (this->meshes[g].ps == PhysicalSystem::Deformable && v1_offset >= 0) {
			const DeformableBookkeeping& bk = this->group_to_deformable_bookkeeping[g];
			for (int i = 0; i < n; i++) {
				const int gidx = this->dyn->get_global_index(this->meshes[g].idx_in_ps, bk.point_set_map[i]);
				const int dof = v1_offset + gidx * 3;
				V1(voff + i, 0) = V0(voff + i, 0) + du[dof + 0] * dt;
				V1(voff + i, 1) = V0(voff + i, 1) + du[dof + 1] * dt;
				V1(voff + i, 2) = V0(voff + i, 2) + du[dof + 2] * dt;
			}
		} else if (this->meshes[g].ps == PhysicalSystem::Rigidbody && rv1_offset >= 0 && rw1_offset >= 0) {
			const RigidBodyBookkeeping& bk = this->group_to_rigidbody_bookkeeping[g];
			const int rb_idx = this->meshes[g].idx_in_ps;
			const int dof_v = rv1_offset + rb_idx * 3;
			const int dof_w = rw1_offset + rb_idx * 3;

			const Eigen::Vector3d v1_new = this->rb->v1[rb_idx] + Eigen::Vector3d{du[dof_v], du[dof_v+1], du[dof_v+2]};
			const Eigen::Vector3d w1_new = this->rb->w1[rb_idx] + Eigen::Vector3d{du[dof_w], du[dof_w+1], du[dof_w+2]};

			const Eigen::Vector3d t1_new = time_integration(this->rb->t0[rb_idx], v1_new, dt);
			const Eigen::Quaterniond q1_new = quat_time_integration(this->rb->q0[rb_idx], w1_new, dt);
			const Eigen::Matrix3d R1_new = q1_new.toRotationMatrix();

			for (int i = 0; i < n; i++) {
				const Eigen::Vector3d p1 = local_to_global_point(this->rigidbody_local_vertices.get(bk.rb_collision_group, i), R1_new, t1_new);
				V1(voff + i, 0) = p1[0];
				V1(voff + i, 1) = p1[1];
				V1(voff + i, 2) = p1[2];
			}
		}
		// If DOFs not found, leave V1[group] = V0[group] (conservative: no CCD for this group)
	}

	const double max_step = ipc::compute_collision_free_stepsize(this->ipc_mesh, V0, V1, 0, this->broad_phase.get());
	return max_step;
}

double EnergyFrictionalContactIPC::_filter_ogc_step(
	Stark& stark,
	const Eigen::VectorXd& x,
	Eigen::VectorXd& du)
{
	(void)x;
	this->ogc_use_ccd_for_step = false;
	if (this->global_params.contact_method != ContactMethod::OGC) return 1.0;
	if (!this->global_params.collisions_enabled || this->is_empty()) return 1.0;
	if (!this->ogc_trust_region || this->ipc_mesh_dirty) return 1.0;

	int v1_offset = -1;
	int rv1_offset = -1;
	int rw1_offset = -1;
	const auto& offsets = stark.global_potential->get_dofs_offsets();
	const int n_sets = stark.global_potential->get_n_dof_sets();
	for (int i = 0; i < n_sets; ++i) {
		const std::string label = stark.global_potential->get_dof_label(i);
		if (label == "soft.v1") v1_offset = offsets[i];
		else if (label == "rigid.v1") rv1_offset = offsets[i];
		else if (label == "rigid.w1") rw1_offset = offsets[i];
	}

	const Eigen::MatrixXd V0 = this->V_combined;
	Eigen::MatrixXd V1 = V0;
	const double dt = stark.dt;

	for (int g = 0; g < (int)this->meshes.size(); ++g) {
		const int voff = this->vertex_offsets[g];
		const int n = (int)this->meshes[g].vertices.size();
		if (this->meshes[g].ps == PhysicalSystem::Deformable && v1_offset >= 0) {
			const DeformableBookkeeping& bk = this->group_to_deformable_bookkeeping[g];
			for (int i = 0; i < n; ++i) {
				const int global = this->dyn->get_global_index(
					this->meshes[g].idx_in_ps, bk.point_set_map[i]);
				V1.row(voff + i) += (dt * du.segment<3>(v1_offset + 3 * global)).transpose();
			}
		} else if (
			this->meshes[g].ps == PhysicalSystem::Rigidbody
			&& rv1_offset >= 0 && rw1_offset >= 0) {
			const int rb_idx = this->meshes[g].idx_in_ps;
			const Eigen::Vector3d v1_new =
				this->rb->v1[rb_idx] + du.segment<3>(rv1_offset + 3 * rb_idx);
			const Eigen::Vector3d w1_new =
				this->rb->w1[rb_idx] + du.segment<3>(rw1_offset + 3 * rb_idx);
			const Eigen::Vector3d t1_new = time_integration(this->rb->t0[rb_idx], v1_new, dt);
			const Eigen::Matrix3d R1_new =
				quat_time_integration(this->rb->q0[rb_idx], w1_new, dt).toRotationMatrix();
			const RigidBodyBookkeeping& bk = this->group_to_rigidbody_bookkeeping[g];
			for (int i = 0; i < n; ++i) {
				V1.row(voff + i) = local_to_global_point(
					this->rigidbody_local_vertices.get(bk.rb_collision_group, i),
					R1_new, t1_new).transpose();
			}
		}
	}

	const Eigen::MatrixXd raw_dx = V1 - V0;
	const double proposed_query_radius = raw_dx.rowwise().norm().maxCoeff();
	int max_displacement_group = -1;
	double max_group_displacement = -1.0;
	for (int g = 0; g < (int)this->meshes.size(); ++g) {
		const int begin = this->vertex_offsets[g];
		const int count = (int)this->meshes[g].vertices.size();
		const double group_displacement = raw_dx.middleRows(begin, count).rowwise().norm().maxCoeff();
		if (group_displacement > max_group_displacement) {
			max_group_displacement = group_displacement;
			max_displacement_group = g;
		}
	}
	const double max_query_radius = std::max(
		this->ogc_trust_region->trust_region_inflation_radius,
		this->global_params.ogc_max_query_radius);
	this->ogc_use_ccd_for_step = proposed_query_radius > max_query_radius;
	stark.context->logger->append(
		"ogc_ccd_fallback", this->ogc_use_ccd_for_step ? 1 : 0);
	if (this->ogc_use_ccd_for_step) {
		// A global OGC query sized from one large Newton correction is especially
		// expensive for localized motion on a dense collision mesh. Keep the
		// current OGC set and let swept CCD scale this direction instead. The
		// collision set is rebuilt around the accepted state next iteration.
		this->ogc_trust_region->should_update_trust_region = true;
		stark.context->logger->add_and_append("ogc_restricted_vertices", 0);
		stark.context->logger->add_and_append("ogc_restricted_dof_blocks", 0);
		stark.context->logger->add_and_append(
			"ogc_query_radius", this->ogc_trust_region->trust_region_inflation_radius);
		stark.context->output->print_with_new_line(fmt::format(
			"OGC filter | query: {:.3e} | proposed: {:.3e} | mode: swept CCD | max group: {}",
			this->ogc_trust_region->trust_region_inflation_radius,
			proposed_query_radius,
			max_displacement_group), symx::Verbosity::Full);
		return 1.0;
	}
	const double bounded_query_radius = std::min(proposed_query_radius, max_query_radius);
	if (bounded_query_radius > this->ogc_trust_region->trust_region_inflation_radius) {
		// IPC Toolkit normally derives the query radius from predicted motion.
		// STARK starts Newton from zero velocity, so the first direction is the
		// available prediction. Bound its global radius: using the full maximum
		// displacement on a dense mesh can create an impractically large candidate
		// set. The current step remains inside the old region; the next iteration
		// rebuilds with the larger, bounded radius.
		this->ogc_trust_region->trust_region_inflation_radius = bounded_query_radius;
		this->ogc_trust_region->should_update_trust_region = true;
	}
	Eigen::MatrixXd filtered_dx = raw_dx;
	this->ogc_trust_region->filter_step(this->ipc_mesh, V0, filtered_dx);

	std::vector<double> deformable_beta(this->dyn->size(), 1.0);
	std::vector<double> rigid_beta(this->rb->get_n_bodies(), 1.0);
	int restricted_vertices = 0;
	for (int g = 0; g < (int)this->meshes.size(); ++g) {
		const int voff = this->vertex_offsets[g];
		const int n = (int)this->meshes[g].vertices.size();
		for (int i = 0; i < n; ++i) {
			const double raw_norm = raw_dx.row(voff + i).norm();
			if (raw_norm <= 1e-15) continue;
			const double beta = std::clamp(
				filtered_dx.row(voff + i).norm() / raw_norm, 0.0, 1.0);
			if (beta < 1.0 - 1e-12) ++restricted_vertices;
			if (this->meshes[g].ps == PhysicalSystem::Deformable) {
				const DeformableBookkeeping& bk = this->group_to_deformable_bookkeeping[g];
				const int global = this->dyn->get_global_index(
					this->meshes[g].idx_in_ps, bk.point_set_map[i]);
				deformable_beta[global] = std::min(deformable_beta[global], beta);
			} else {
				const int rb_idx = this->meshes[g].idx_in_ps;
				rigid_beta[rb_idx] = std::min(rigid_beta[rb_idx], beta);
			}
		}
	}

	double min_beta = 1.0;
	int restricted_blocks = 0;
	if (v1_offset >= 0) {
		for (int i = 0; i < (int)deformable_beta.size(); ++i) {
			const double beta = deformable_beta[i];
			if (beta >= 1.0) continue;
			du.segment<3>(v1_offset + 3 * i) *= beta;
			min_beta = std::min(min_beta, beta);
			++restricted_blocks;
		}
	}

	if (rv1_offset >= 0 && rw1_offset >= 0) {
		for (int rb_idx = 0; rb_idx < (int)rigid_beta.size(); ++rb_idx) {
			double beta = rigid_beta[rb_idx];
			if (beta >= 1.0) continue;

			// Scaling angular generalized coordinates is only approximately the
			// same as scaling vertex displacement. Tighten until the exact rigid
			// transform lies inside every associated vertex trust region.
			for (int attempt = 0; attempt < 60; ++attempt) {
				const Eigen::Vector3d v1_new = this->rb->v1[rb_idx]
					+ beta * du.segment<3>(rv1_offset + 3 * rb_idx);
				const Eigen::Vector3d w1_new = this->rb->w1[rb_idx]
					+ beta * du.segment<3>(rw1_offset + 3 * rb_idx);
				const Eigen::Vector3d t1_new = time_integration(this->rb->t0[rb_idx], v1_new, dt);
				const Eigen::Matrix3d R1_new =
					quat_time_integration(this->rb->q0[rb_idx], w1_new, dt).toRotationMatrix();
				bool safe = true;
				for (int g = 0; g < (int)this->meshes.size() && safe; ++g) {
					if (this->meshes[g].ps != PhysicalSystem::Rigidbody
						|| this->meshes[g].idx_in_ps != rb_idx) continue;
					const RigidBodyBookkeeping& bk = this->group_to_rigidbody_bookkeeping[g];
					for (int i = 0; i < (int)this->meshes[g].vertices.size(); ++i) {
						const int vertex = this->vertex_offsets[g] + i;
						const Eigen::Vector3d position = local_to_global_point(
							this->rigidbody_local_vertices.get(bk.rb_collision_group, i),
							R1_new, t1_new);
						if ((position - this->ogc_trust_region->trust_region_centers.row(vertex).transpose()).norm()
							> this->ogc_trust_region->trust_region_radii[vertex] + 1e-10) {
							safe = false;
							break;
						}
					}
				}
				if (safe) break;
				beta *= 0.5;
			}

			du.segment<3>(rv1_offset + 3 * rb_idx) *= beta;
			du.segment<3>(rw1_offset + 3 * rb_idx) *= beta;
			min_beta = std::min(min_beta, beta);
			++restricted_blocks;
		}
	}

	stark.context->logger->add_and_append("ogc_restricted_vertices", restricted_vertices);
	stark.context->logger->add_and_append("ogc_restricted_dof_blocks", restricted_blocks);
	stark.context->logger->add_and_append(
		"ogc_query_radius", this->ogc_trust_region->trust_region_inflation_radius);
	std::string max_group_label = "unknown";
	if (max_displacement_group >= 0) {
		const ContactMesh& mesh = this->meshes[max_displacement_group];
		if (mesh.ps == PhysicalSystem::Deformable
			&& mesh.idx_in_ps >= 0 && mesh.idx_in_ps < (int)this->dyn->labels.size()) {
			max_group_label = this->dyn->labels[mesh.idx_in_ps];
		} else if (mesh.ps == PhysicalSystem::Rigidbody
			&& mesh.idx_in_ps >= 0 && mesh.idx_in_ps < (int)this->rb->labels.size()) {
			max_group_label = this->rb->labels[mesh.idx_in_ps];
		}
	}
	stark.context->output->print_with_new_line(fmt::format(
		"OGC filter | query: {:.3e} | proposed: {:.3e} | restricted: {}/{} vertices, {} blocks | min beta: {:.3e} | next rebuild: {} | max group: {} ({})",
		this->ogc_trust_region->trust_region_inflation_radius,
		proposed_query_radius,
		restricted_vertices,
		this->ipc_mesh.num_vertices(),
		restricted_blocks,
		min_beta,
		this->ogc_trust_region->should_update_trust_region ? "yes" : "no",
		max_displacement_group,
		max_group_label), symx::Verbosity::Full);
	return min_beta;
}

void EnergyFrictionalContactIPC::_before_newton_step__update_ogc_contacts(Stark& stark)
{
	if (this->global_params.contact_method != ContactMethod::OGC) return;
	if (!this->global_params.collisions_enabled || this->is_empty()) return;

	this->_update_vertices(stark, stark.dt);
	if (this->ipc_mesh_dirty) this->_build_ipc_mesh();
	this->_update_combined_V();

	const double max_thickness = *std::max_element(this->contact_thicknesses.begin(), this->contact_thicknesses.end());
	const double query_distance = 2.0 * max_thickness;
	const bool rebuild_requested = !this->ogc_trust_region
		|| this->ogc_trust_region->should_update_trust_region;
	if (!this->ogc_trust_region) {
		this->ogc_trust_region = std::make_unique<ipc::ogc::TrustRegion>(query_distance);
	}
	this->ogc_trust_region->relaxed_radius_scaling = this->global_params.ogc_relaxed_radius_scaling;
	this->ogc_trust_region->update_threshold = this->global_params.ogc_update_threshold;
	this->ogc_trust_region->update_if_needed(
		this->ipc_mesh, this->V_combined, this->ogc_collisions, 0.0, this->broad_phase.get());
	this->_populate_ogc_contacts();

	stark.context->logger->add_and_append("ogc_vv", (double)this->ogc_collisions.vv_collisions.size());
	stark.context->logger->add_and_append("ogc_ev", (double)this->ogc_collisions.ev_collisions.size());
	stark.context->logger->add_and_append("ogc_ee", (double)this->ogc_collisions.ee_collisions.size());
	stark.context->logger->add_and_append("ogc_fv", (double)this->ogc_collisions.fv_collisions.size());
	stark.context->output->print_with_new_line(fmt::format(
		"OGC set    | rebuilt: {} | candidates: {} | contacts (vv/ev/ee/fv): {}/{}/{}/{}",
		rebuild_requested ? "yes" : "no",
		this->ogc_trust_region->candidates.size(),
		this->ogc_collisions.vv_collisions.size(),
		this->ogc_collisions.ev_collisions.size(),
		this->ogc_collisions.ee_collisions.size(),
		this->ogc_collisions.fv_collisions.size()), symx::Verbosity::Full);
}

void EnergyFrictionalContactIPC::_populate_ogc_contacts()
{
	this->ogc_contacts.clear();

	auto vertex_info = [&](int vertex_id) {
		const int group = this->_vertex_to_group(vertex_id);
		const int local = vertex_id - this->vertex_offsets[group];
		const int global = this->_local_to_ps_global_indices<1>(group, {local})[0];
		return std::array<int, 3>{group, local, global};
	};
	auto edge_info = [&](int edge_id) {
		const int group = this->_edge_to_group(edge_id);
		const int local = edge_id - this->edge_offsets[group];
		const auto global = this->_local_to_ps_global_indices<2>(group, this->meshes[group].loc_edges[local]);
		return std::tuple<int, int, int>{group, global[0], global[1]};
	};
	auto face_info = [&](int face_id) {
		const int group = this->_tri_to_group(face_id);
		const int local = face_id - this->tri_offsets[group];
		const auto global = this->_local_to_ps_global_indices<3>(group, this->meshes[group].loc_triangles[local]);
		return std::tuple<int, int, int, int>{group, global[0], global[1], global[2]};
	};

	for (const auto& collision : this->ogc_collisions.vv_collisions) {
		auto a = vertex_info(collision.vertex0_id);
		auto b = vertex_info(collision.vertex1_id);
		const auto psa = this->meshes[a[0]].ps;
		const auto psb = this->meshes[b[0]].ps;
		if (psa == PhysicalSystem::Deformable && psb == PhysicalSystem::Deformable) {
			this->ogc_contacts.deformable.vv.push_back({a[0], b[0], a[2], b[2]}, collision.weight);
		} else if (psa == PhysicalSystem::Rigidbody && psb == PhysicalSystem::Rigidbody) {
			this->ogc_contacts.rigid_body.vv.push_back({a[0], b[0], this->meshes[a[0]].idx_in_ps, this->meshes[b[0]].idx_in_ps, a[2], b[2]}, collision.weight);
		} else if (psa == PhysicalSystem::Rigidbody) {
			this->ogc_contacts.mixed.vv.push_back({a[0], b[0], this->meshes[a[0]].idx_in_ps, a[2], b[2]}, collision.weight);
		} else {
			this->ogc_contacts.mixed.vv.push_back({b[0], a[0], this->meshes[b[0]].idx_in_ps, b[2], a[2]}, collision.weight);
		}
	}

	for (const auto& collision : this->ogc_collisions.ev_collisions) {
		auto v = vertex_info(collision.vertex_id);
		auto [eg, e0, e1] = edge_info(collision.edge_id);
		const auto psv = this->meshes[v[0]].ps;
		const auto pse = this->meshes[eg].ps;
		if (psv == PhysicalSystem::Deformable && pse == PhysicalSystem::Deformable) {
			this->ogc_contacts.deformable.ev.push_back({v[0], eg, v[2], e0, e1}, collision.weight);
		} else if (psv == PhysicalSystem::Rigidbody && pse == PhysicalSystem::Rigidbody) {
			this->ogc_contacts.rigid_body.ev.push_back({v[0], eg, this->meshes[v[0]].idx_in_ps, this->meshes[eg].idx_in_ps, v[2], e0, e1}, collision.weight);
		} else if (psv == PhysicalSystem::Rigidbody) {
			this->ogc_contacts.mixed.rb_vertex_d_edge.push_back({v[0], eg, this->meshes[v[0]].idx_in_ps, v[2], e0, e1}, collision.weight);
		} else {
			this->ogc_contacts.mixed.d_vertex_rb_edge.push_back({v[0], eg, this->meshes[eg].idx_in_ps, v[2], e0, e1}, collision.weight);
		}
	}

	for (const auto& collision : this->ogc_collisions.ee_collisions) {
		auto [ga, a0, a1] = edge_info(collision.edge0_id);
		auto [gb, b0, b1] = edge_info(collision.edge1_id);
		const auto psa = this->meshes[ga].ps;
		const auto psb = this->meshes[gb].ps;
		if (psa == PhysicalSystem::Deformable && psb == PhysicalSystem::Deformable) {
			this->ogc_contacts.deformable.ee.get(collision.dtype).push_back({ga, gb, a0, a1, b0, b1}, collision.weight);
		} else if (psa == PhysicalSystem::Rigidbody && psb == PhysicalSystem::Rigidbody) {
			this->ogc_contacts.rigid_body.ee.get(collision.dtype).push_back({ga, gb, this->meshes[ga].idx_in_ps, this->meshes[gb].idx_in_ps, a0, a1, b0, b1}, collision.weight);
		} else if (psa == PhysicalSystem::Rigidbody) {
			this->ogc_contacts.mixed.ee.get(collision.dtype).push_back({ga, gb, this->meshes[ga].idx_in_ps, a0, a1, b0, b1}, collision.weight);
		} else {
			using EE = ipc::EdgeEdgeDistanceType;
			EE swapped_type = collision.dtype;
			switch (collision.dtype) {
				case EE::EA_EB0: swapped_type = EE::EA0_EB; break;
				case EE::EA_EB1: swapped_type = EE::EA1_EB; break;
				case EE::EA0_EB: swapped_type = EE::EA_EB0; break;
				case EE::EA1_EB: swapped_type = EE::EA_EB1; break;
				case EE::EA0_EB1: swapped_type = EE::EA1_EB0; break;
				case EE::EA1_EB0: swapped_type = EE::EA0_EB1; break;
				default: break;
			}
			this->ogc_contacts.mixed.ee.get(swapped_type).push_back({gb, ga, this->meshes[gb].idx_in_ps, b0, b1, a0, a1}, collision.weight);
		}
	}

	for (const auto& collision : this->ogc_collisions.fv_collisions) {
		auto v = vertex_info(collision.vertex_id);
		auto [fg, t0, t1, t2] = face_info(collision.face_id);
		const auto psv = this->meshes[v[0]].ps;
		const auto psf = this->meshes[fg].ps;
		if (psv == PhysicalSystem::Deformable && psf == PhysicalSystem::Deformable) {
			this->ogc_contacts.deformable.fv.push_back({v[0], fg, v[2], t0, t1, t2}, collision.weight);
		} else if (psv == PhysicalSystem::Rigidbody && psf == PhysicalSystem::Rigidbody) {
			this->ogc_contacts.rigid_body.fv.push_back({v[0], fg, this->meshes[v[0]].idx_in_ps, this->meshes[fg].idx_in_ps, v[2], t0, t1, t2}, collision.weight);
		} else if (psv == PhysicalSystem::Rigidbody) {
			this->ogc_contacts.mixed.rb_vertex_d_face.push_back({v[0], fg, this->meshes[v[0]].idx_in_ps, v[2], t0, t1, t2}, collision.weight);
		} else {
			this->ogc_contacts.mixed.d_vertex_rb_face.push_back({v[0], fg, this->meshes[fg].idx_in_ps, v[2], t0, t1, t2}, collision.weight);
		}
	}
}


/* ========================================================================================== */
/* ===================================  SYMX DEFINITIONS  =================================== */
/* ========================================================================================== */

void EnergyFrictionalContactIPC::_energies_contact_deformables(Stark& stark)
{
	stark.global_potential->add_potential(this->_get_contact_label("d_d", "pt_pp"), this->contacts_deformables.point_triangle.point_point,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			std::vector<Vector> p = this->_get_d_x1(mws, stark, { conn["p"] });
			std::vector<Vector> q = this->_get_d_x1(mws, stark, { conn["q"] });
			Scalar d = distance_point_point(p[0], q[0]);
			return this->_barrier_potential(mws, stark, d, conn["group_a"], conn["group_b"]);
		});

	stark.global_potential->add_potential(this->_get_contact_label("d_d", "pt_pe"), this->contacts_deformables.point_triangle.point_edge,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			std::vector<Vector> p = this->_get_d_x1(mws, stark, { conn["p"] });
			std::vector<Vector> e = this->_get_d_x1(mws, stark, { conn["e0"], conn["e1"] });
			Scalar d = distance_point_line(p[0], e[0], e[1]);
			return this->_barrier_potential(mws, stark, d, conn["group_a"], conn["group_b"]);
		});

	stark.global_potential->add_potential(this->_get_contact_label("d_d", "pt_pt"), this->contacts_deformables.point_triangle.point_triangle,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			std::vector<Vector> p = this->_get_d_x1(mws, stark, { conn["p"] });
			std::vector<Vector> t = this->_get_d_x1(mws, stark, { conn["t0"], conn["t1"], conn["t2"] });
			Scalar d = distance_point_plane(p[0], t[0], t[1], t[2]);
			return this->_barrier_potential(mws, stark, d, conn["group_a"], conn["group_b"]);
		});

	stark.global_potential->add_potential(this->_get_contact_label("d_d", "ee_pp"), this->contacts_deformables.edge_edge.point_point,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			auto [ea, ea_rest, p] = this->_get_d_edge_point(mws, stark, { conn["ea0"], conn["ea1"], conn["p"] });
			auto [eb, eb_rest, q] = this->_get_d_edge_point(mws, stark, { conn["eb0"], conn["eb1"], conn["q"] });
			Scalar d = distance_point_point(p[0], q[0]);
			return this->_edge_edge_mollified_barrier_potential(mws, stark, d, ea, eb, ea_rest, eb_rest, conn["group_a"], conn["group_b"]);
		});

	stark.global_potential->add_potential(this->_get_contact_label("d_d", "ee_pe"), this->contacts_deformables.edge_edge.point_edge,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			auto [ea, ea_rest, p] = this->_get_d_edge_point(mws, stark, { conn["ea0"], conn["ea1"], conn["p"] });
			auto [eb, eb_rest] = this->_get_d_edge(mws, stark, { conn["eb0"], conn["eb1"] });
			Scalar d = distance_point_line(p[0], eb[0], eb[1]);
			return this->_edge_edge_mollified_barrier_potential(mws, stark, d, ea, eb, ea_rest, eb_rest, conn["group_a"], conn["group_b"]);
		});

	stark.global_potential->add_potential(this->_get_contact_label("d_d", "ee_ee"), this->contacts_deformables.edge_edge.edge_edge,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			auto [ea, ea_rest] = this->_get_d_edge(mws, stark, { conn["ea0"], conn["ea1"] });
			auto [eb, eb_rest] = this->_get_d_edge(mws, stark, { conn["eb0"], conn["eb1"] });
			Scalar d = distance_line_line(ea[0], ea[1], eb[0], eb[1]);
			return this->_edge_edge_mollified_barrier_potential(mws, stark, d, ea, eb, ea_rest, eb_rest, conn["group_a"], conn["group_b"]);
		});
}

void EnergyFrictionalContactIPC::_energies_contact_rb(Stark& stark)
{
	stark.global_potential->add_potential(this->_get_contact_label("rb_rb", "pt_pp"), this->contacts_rb.point_triangle.point_point,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			std::vector<Vector> p = this->_get_rb_x1(mws, stark, conn["a"], { conn["p"] });
			std::vector<Vector> q = this->_get_rb_x1(mws, stark, conn["b"], { conn["q"] });
			Scalar d = distance_point_point(p[0], q[0]);
			return this->_barrier_potential(mws, stark, d, conn["group_a"], conn["group_b"]);
		});

	stark.global_potential->add_potential(this->_get_contact_label("rb_rb", "pt_pe"), this->contacts_rb.point_triangle.point_edge,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			std::vector<Vector> p = this->_get_rb_x1(mws, stark, conn["a"], { conn["p"] });
			std::vector<Vector> e = this->_get_rb_x1(mws, stark, conn["b"], { conn["e0"], conn["e1"] });
			Scalar d = distance_point_line(p[0], e[0], e[1]);
			return this->_barrier_potential(mws, stark, d, conn["group_a"], conn["group_b"]);
		});

	stark.global_potential->add_potential(this->_get_contact_label("rb_rb", "pt_pt"), this->contacts_rb.point_triangle.point_triangle,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			std::vector<Vector> p = this->_get_rb_x1(mws, stark, conn["a"], { conn["p"] });
			std::vector<Vector> t = this->_get_rb_x1(mws, stark, conn["b"], { conn["t0"], conn["t1"], conn["t2"] });
			Scalar d = distance_point_plane(p[0], t[0], t[1], t[2]);
			return this->_barrier_potential(mws, stark, d, conn["group_a"], conn["group_b"]);
		});

	stark.global_potential->add_potential(this->_get_contact_label("rb_rb", "ee_pp"), this->contacts_rb.edge_edge.point_point,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			auto [ea, ea_rest, p] = this->_get_rb_edge_point(mws, stark, conn["a"], { conn["ea0"], conn["ea1"], conn["p"] });
			auto [eb, eb_rest, q] = this->_get_rb_edge_point(mws, stark, conn["b"], { conn["eb0"], conn["eb1"], conn["q"] });
			Scalar d = distance_point_point(p[0], q[0]);
			return this->_edge_edge_mollified_barrier_potential(mws, stark, d, ea, eb, ea_rest, eb_rest, conn["group_a"], conn["group_b"]);
		});

	stark.global_potential->add_potential(this->_get_contact_label("rb_rb", "ee_pe"), this->contacts_rb.edge_edge.point_edge,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			auto [ea, ea_rest, p] = this->_get_rb_edge_point(mws, stark, conn["a"], { conn["ea0"], conn["ea1"], conn["p"] });
			auto [eb, eb_rest] = this->_get_rb_edge(mws, stark, conn["b"], { conn["eb0"], conn["eb1"] });
			Scalar d = distance_point_line(p[0], eb[0], eb[1]);
			return this->_edge_edge_mollified_barrier_potential(mws, stark, d, ea, eb, ea_rest, eb_rest, conn["group_a"], conn["group_b"]);
		});

	stark.global_potential->add_potential(this->_get_contact_label("rb_rb", "ee_ee"), this->contacts_rb.edge_edge.edge_edge,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			auto [ea, ea_rest] = this->_get_rb_edge(mws, stark, conn["a"], { conn["ea0"], conn["ea1"] });
			auto [eb, eb_rest] = this->_get_rb_edge(mws, stark, conn["b"], { conn["eb0"], conn["eb1"] });
			Scalar d = distance_line_line(ea[0], ea[1], eb[0], eb[1]);
			return this->_edge_edge_mollified_barrier_potential(mws, stark, d, ea, eb, ea_rest, eb_rest, conn["group_a"], conn["group_b"]);
		});
}

void EnergyFrictionalContactIPC::_energies_contact_rb_deformables(Stark& stark)
{
	stark.global_potential->add_potential(this->_get_contact_label("rb_d", "pt_pp"), this->contacts_rb_deformables.point_triangle.rb_d_point_point,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			std::vector<Vector> p = this->_get_rb_x1(mws, stark, conn["rb"], { conn["p"] });
			std::vector<Vector> q = this->_get_d_x1(mws, stark, { conn["q"] });
			Scalar d = distance_point_point(p[0], q[0]);
			return this->_barrier_potential(mws, stark, d, conn["group_a"], conn["group_b"]);
		});

	stark.global_potential->add_potential(this->_get_contact_label("rb_d", "pt_pe"), this->contacts_rb_deformables.point_triangle.rb_d_point_edge,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			std::vector<Vector> p = this->_get_rb_x1(mws, stark, conn["rb"], { conn["p"] });
			std::vector<Vector> e = this->_get_d_x1(mws, stark, { conn["e0"], conn["e1"] });
			Scalar d = distance_point_line(p[0], e[0], e[1]);
			return this->_barrier_potential(mws, stark, d, conn["group_a"], conn["group_b"]);
		});

	stark.global_potential->add_potential(this->_get_contact_label("rb_d", "pt_pt"), this->contacts_rb_deformables.point_triangle.rb_d_point_triangle,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			std::vector<Vector> p = this->_get_rb_x1(mws, stark, conn["rb"], { conn["p"] });
			std::vector<Vector> t = this->_get_d_x1(mws, stark, { conn["t0"], conn["t1"], conn["t2"] });
			Scalar d = distance_point_plane(p[0], t[0], t[1], t[2]);
			return this->_barrier_potential(mws, stark, d, conn["group_a"], conn["group_b"]);
		});

	stark.global_potential->add_potential(this->_get_contact_label("rb_d", "pt_ep"), this->contacts_rb_deformables.point_triangle.rb_d_edge_point,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			std::vector<Vector> e = this->_get_rb_x1(mws, stark, conn["rb"], { conn["e0"], conn["e1"] });
			std::vector<Vector> p = this->_get_d_x1(mws, stark, { conn["p"] });
			Scalar d = distance_point_line(p[0], e[0], e[1]);
			return this->_barrier_potential(mws, stark, d, conn["group_a"], conn["group_b"]);
		});

	stark.global_potential->add_potential(this->_get_contact_label("rb_d", "pt_tp"), this->contacts_rb_deformables.point_triangle.rb_d_triangle_point,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			std::vector<Vector> t = this->_get_rb_x1(mws, stark, conn["rb"], { conn["t0"], conn["t1"], conn["t2"] });
			std::vector<Vector> p = this->_get_d_x1(mws, stark, { conn["p"] });
			Scalar d = distance_point_plane(p[0], t[0], t[1], t[2]);
			return this->_barrier_potential(mws, stark, d, conn["group_a"], conn["group_b"]);
		});

	stark.global_potential->add_potential(this->_get_contact_label("rb_d", "ee_pp"), this->contacts_rb_deformables.edge_edge.rb_d_point_point,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			auto [ea, ea_rest, p] = this->_get_rb_edge_point(mws, stark, conn["rb"], { conn["ea0"], conn["ea1"], conn["p"] });
			auto [eb, eb_rest, q] = this->_get_d_edge_point(mws, stark, { conn["eb0"], conn["eb1"], conn["q"] });
			Scalar d = distance_point_point(p[0], q[0]);
			return this->_edge_edge_mollified_barrier_potential(mws, stark, d, ea, eb, ea_rest, eb_rest, conn["group_a"], conn["group_b"]);
		});

	stark.global_potential->add_potential(this->_get_contact_label("rb_d", "ee_pe"), this->contacts_rb_deformables.edge_edge.rb_d_point_edge,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			auto [ea, ea_rest, p] = this->_get_rb_edge_point(mws, stark, conn["rb"], { conn["ea0"], conn["ea1"], conn["p"] });
			auto [eb, eb_rest] = this->_get_d_edge(mws, stark, { conn["eb0"], conn["eb1"] });
			Scalar d = distance_point_line(p[0], eb[0], eb[1]);
			return this->_edge_edge_mollified_barrier_potential(mws, stark, d, ea, eb, ea_rest, eb_rest, conn["group_a"], conn["group_b"]);
		});

	stark.global_potential->add_potential(this->_get_contact_label("rb_d", "ee_ee"), this->contacts_rb_deformables.edge_edge.rb_d_edge_edge,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			auto [ea, ea_rest] = this->_get_rb_edge(mws, stark, conn["rb"], { conn["ea0"], conn["ea1"] });
			auto [eb, eb_rest] = this->_get_d_edge(mws, stark, { conn["eb0"], conn["eb1"] });
			Scalar d = distance_line_line(ea[0], ea[1], eb[0], eb[1]);
			return this->_edge_edge_mollified_barrier_potential(mws, stark, d, ea, eb, ea_rest, eb_rest, conn["group_a"], conn["group_b"]);
		});

	stark.global_potential->add_potential(this->_get_contact_label("rb_d", "ee_ep"), this->contacts_rb_deformables.edge_edge.rb_d_edge_point,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			auto [ea, ea_rest] = this->_get_rb_edge(mws, stark, conn["rb"], { conn["ea0"], conn["ea1"] });
			auto [eb, eb_rest, q] = this->_get_d_edge_point(mws, stark, { conn["eb0"], conn["eb1"], conn["q"] });
			Scalar d = distance_point_line(q[0], ea[0], ea[1]);
			return this->_edge_edge_mollified_barrier_potential(mws, stark, d, ea, eb, ea_rest, eb_rest, conn["group_a"], conn["group_b"]);
		});
}

void EnergyFrictionalContactIPC::_energies_friction_deformables(Stark& stark)
{
	stark.global_potential->add_potential(this->_get_friction_label("d_d", "pp"), this->friction_deformables.point_point.conn,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			std::vector<Vector> vp = this->_get_d_v1(mws, { conn["p"] });
			std::vector<Vector> vq = this->_get_d_v1(mws, { conn["q"] });
			Vector v = vq[0] - vp[0];
			return this->_friction_potential(mws, stark, v, conn["idx"], this->friction_deformables.point_point.contact);
		});

	stark.global_potential->add_potential(this->_get_friction_label("d_d", "pe"), this->friction_deformables.point_edge.conn,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			std::vector<Vector> vp = this->_get_d_v1(mws, { conn["p"] });
			std::vector<Vector> ve = this->_get_d_v1(mws, { conn["e0"], conn["e1"] });
			return this->_friction_point_edge(mws, stark, vp[0], ve, conn["idx"], this->friction_deformables.point_edge.data);
		});

	stark.global_potential->add_potential(this->_get_friction_label("d_d", "pt"), this->friction_deformables.point_triangle.conn,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			std::vector<Vector> vp = this->_get_d_v1(mws, { conn["p"] });
			std::vector<Vector> vt = this->_get_d_v1(mws, { conn["t0"], conn["t1"], conn["t2"] });
			return this->_friction_point_triangle(mws, stark, vp[0], vt, conn["idx"], this->friction_deformables.point_triangle.data);
		});

	stark.global_potential->add_potential(this->_get_friction_label("d_d", "ee"), this->friction_deformables.edge_edge.conn,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			std::vector<Vector> vea = this->_get_d_v1(mws, { conn["ea0"], conn["ea1"] });
			std::vector<Vector> veb = this->_get_d_v1(mws, { conn["eb0"], conn["eb1"] });
			return this->_friction_edge_edge(mws, stark, vea, veb, conn["idx"], this->friction_deformables.edge_edge.data);
		});
}

void EnergyFrictionalContactIPC::_energies_friction_rb(Stark& stark)
{
	stark.global_potential->add_potential(this->_get_friction_label("rb_rb", "pp"), this->friction_rb.point_point.conn,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			std::vector<Vector> vp = this->_get_rb_v1(mws, stark, conn["a"], { conn["p"] });
			std::vector<Vector> vq = this->_get_rb_v1(mws, stark, conn["b"], { conn["q"] });
			Vector v = vq[0] - vp[0];
			return this->_friction_potential(mws, stark, v, conn["idx"], this->friction_rb.point_point.contact);
		});

	stark.global_potential->add_potential(this->_get_friction_label("rb_rb", "pe"), this->friction_rb.point_edge.conn,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			std::vector<Vector> vp = this->_get_rb_v1(mws, stark, conn["a"], { conn["p"] });
			std::vector<Vector> ve = this->_get_rb_v1(mws, stark, conn["b"], { conn["e0"], conn["e1"] });
			return this->_friction_point_edge(mws, stark, vp[0], ve, conn["idx"], this->friction_rb.point_edge.data);
		});

	stark.global_potential->add_potential(this->_get_friction_label("rb_rb", "pt"), this->friction_rb.point_triangle.conn,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			std::vector<Vector> vp = this->_get_rb_v1(mws, stark, conn["a"], { conn["p"] });
			std::vector<Vector> vt = this->_get_rb_v1(mws, stark, conn["b"], { conn["t0"], conn["t1"], conn["t2"] });
			return this->_friction_point_triangle(mws, stark, vp[0], vt, conn["idx"], this->friction_rb.point_triangle.data);
		});

	stark.global_potential->add_potential(this->_get_friction_label("rb_rb", "ee"), this->friction_rb.edge_edge.conn,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			std::vector<Vector> vea = this->_get_rb_v1(mws, stark, conn["a"], { conn["ea0"], conn["ea1"] });
			std::vector<Vector> veb = this->_get_rb_v1(mws, stark, conn["b"], { conn["eb0"], conn["eb1"] });
			return this->_friction_edge_edge(mws, stark, vea, veb, conn["idx"], this->friction_rb.edge_edge.data);
		});
}

void EnergyFrictionalContactIPC::_energies_friction_rb_deformables(Stark& stark)
{
	stark.global_potential->add_potential(this->_get_friction_label("rb_d", "pp"), this->friction_rb_deformables.point_point.conn,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			std::vector<Vector> vp = this->_get_rb_v1(mws, stark, conn["rb"], { conn["p"] });
			std::vector<Vector> vq = this->_get_d_v1(mws, { conn["q"] });
			Vector v = vq[0] - vp[0];
			return this->_friction_potential(mws, stark, v, conn["idx"], this->friction_rb_deformables.point_point.contact);
		});

	stark.global_potential->add_potential(this->_get_friction_label("rb_d", "pe"), this->friction_rb_deformables.point_edge.conn,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			std::vector<Vector> vp = this->_get_rb_v1(mws, stark, conn["rb"], { conn["p"] });
			std::vector<Vector> ve = this->_get_d_v1(mws, { conn["e0"], conn["e1"] });
			return this->_friction_point_edge(mws, stark, vp[0], ve, conn["idx"], this->friction_rb_deformables.point_edge.data);
		});

	stark.global_potential->add_potential(this->_get_friction_label("rb_d", "pt"), this->friction_rb_deformables.point_triangle.conn,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			std::vector<Vector> vp = this->_get_rb_v1(mws, stark, conn["rb"], { conn["p"] });
			std::vector<Vector> vt = this->_get_d_v1(mws, { conn["t0"], conn["t1"], conn["t2"] });
			return this->_friction_point_triangle(mws, stark, vp[0], vt, conn["idx"], this->friction_rb_deformables.point_triangle.data);
		});

	stark.global_potential->add_potential(this->_get_friction_label("rb_d", "ee"), this->friction_rb_deformables.edge_edge.conn,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			std::vector<Vector> vea = this->_get_rb_v1(mws, stark, conn["rb"], { conn["ea0"], conn["ea1"] });
			std::vector<Vector> veb = this->_get_d_v1(mws, { conn["eb0"], conn["eb1"] });
			return this->_friction_edge_edge(mws, stark, vea, veb, conn["idx"], this->friction_rb_deformables.edge_edge.data);
		});

	stark.global_potential->add_potential(this->_get_friction_label("rb_d", "ep"), this->friction_rb_deformables.edge_point.conn,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			std::vector<Vector> ve = this->_get_rb_v1(mws, stark, conn["rb"], { conn["e0"], conn["e1"] });
			std::vector<Vector> vp = this->_get_d_v1(mws, { conn["p"] });
			return this->_friction_point_edge(mws, stark, vp[0], ve, conn["idx"], this->friction_rb_deformables.edge_point.data);
		});

	stark.global_potential->add_potential(this->_get_friction_label("rb_d", "tp"), this->friction_rb_deformables.triangle_point.conn,
		[&](MappedWorkspace<double>& mws, Element& conn)
		{
			std::vector<Vector> vt = this->_get_rb_v1(mws, stark, conn["rb"], { conn["t0"], conn["t1"], conn["t2"] });
			std::vector<Vector> vp = this->_get_d_v1(mws, { conn["p"] });
			return this->_friction_point_triangle(mws, stark, vp[0], vt, conn["idx"], this->friction_rb_deformables.triangle_point.data);
		});
}

void EnergyFrictionalContactIPC::_energies_ogc(Stark& stark)
{
	auto add_deformable = [&](const std::string& label, auto& weighted_conn, auto distance) {
		auto* weights = &weighted_conn.weights;
		stark.global_potential->add_potential(label, weighted_conn.conn,
			[&, distance, weights](MappedWorkspace<double>& mws, Element& conn) {
				return this->_ogc_barrier_potential(
					mws, distance(mws, conn), conn["idx"], conn["group_a"], conn["group_b"], *weights);
			});
		};
	auto add_edge_edge = [&](const std::string& prefix, auto& edge_edge, auto get_edges) {
		auto add = [&](const std::string& suffix, auto& weighted_conn, auto distance) {
			auto* weights = &weighted_conn.weights;
			stark.global_potential->add_potential(prefix + suffix, weighted_conn.conn,
				[&, get_edges, distance, weights](MappedWorkspace<double>& mws, Element& conn) {
					auto [ea, eb, ea_rest, eb_rest] = get_edges(mws, conn);
					auto out = this->_ogc_barrier_potential(
						mws, distance(ea, eb), conn["idx"], conn["group_a"], conn["group_b"], *weights);
					out.first *= this->_edge_edge_mollifier(ea, eb, ea_rest, eb_rest);
					return out;
				});
		};
		add("ea0_eb0", edge_edge.ea0_eb0, [](const auto& ea, const auto& eb) { return distance_point_point(ea[0], eb[0]); });
		add("ea0_eb1", edge_edge.ea0_eb1, [](const auto& ea, const auto& eb) { return distance_point_point(ea[0], eb[1]); });
		add("ea1_eb0", edge_edge.ea1_eb0, [](const auto& ea, const auto& eb) { return distance_point_point(ea[1], eb[0]); });
		add("ea1_eb1", edge_edge.ea1_eb1, [](const auto& ea, const auto& eb) { return distance_point_point(ea[1], eb[1]); });
		add("ea_eb0", edge_edge.ea_eb0, [](const auto& ea, const auto& eb) { return distance_point_line(eb[0], ea[0], ea[1]); });
		add("ea_eb1", edge_edge.ea_eb1, [](const auto& ea, const auto& eb) { return distance_point_line(eb[1], ea[0], ea[1]); });
		add("ea0_eb", edge_edge.ea0_eb, [](const auto& ea, const auto& eb) { return distance_point_line(ea[0], eb[0], eb[1]); });
		add("ea1_eb", edge_edge.ea1_eb, [](const auto& ea, const auto& eb) { return distance_point_line(ea[1], eb[0], eb[1]); });
		add("ea_eb", edge_edge.ea_eb, [](const auto& ea, const auto& eb) { return distance_line_line(ea[0], ea[1], eb[0], eb[1]); });
	};

	add_deformable("ogc_contact_d_d_vv", this->ogc_contacts.deformable.vv,
		[&](MappedWorkspace<double>& mws, Element& conn) {
			auto x = this->_get_d_x1(mws, stark, {conn["p"], conn["q"]});
			return distance_point_point(x[0], x[1]);
		});
	add_deformable("ogc_contact_d_d_ev", this->ogc_contacts.deformable.ev,
		[&](MappedWorkspace<double>& mws, Element& conn) {
			auto x = this->_get_d_x1(mws, stark, {conn["p"], conn["e0"], conn["e1"]});
			return distance_point_line(x[0], x[1], x[2]);
		});
	add_deformable("ogc_contact_d_d_fv", this->ogc_contacts.deformable.fv,
		[&](MappedWorkspace<double>& mws, Element& conn) {
			auto x = this->_get_d_x1(mws, stark, {conn["p"], conn["t0"], conn["t1"], conn["t2"]});
			return distance_point_plane(x[0], x[1], x[2], x[3]);
		});
	add_edge_edge("ogc_contact_d_d_ee_", this->ogc_contacts.deformable.ee,
		[&](MappedWorkspace<double>& mws, Element& conn) {
			auto [ea, ea_rest] = this->_get_d_edge(mws, stark, {conn["ea0"], conn["ea1"]});
			auto [eb, eb_rest] = this->_get_d_edge(mws, stark, {conn["eb0"], conn["eb1"]});
			return std::make_tuple(ea, eb, ea_rest, eb_rest);
		});

	auto add_rigid = [&](const std::string& label, auto& weighted_conn, auto distance) {
		auto* weights = &weighted_conn.weights;
		stark.global_potential->add_potential(label, weighted_conn.conn,
			[&, distance, weights](MappedWorkspace<double>& mws, Element& conn) {
				return this->_ogc_barrier_potential(
					mws, distance(mws, conn), conn["idx"], conn["group_a"], conn["group_b"], *weights);
			});
	};
	add_rigid("ogc_contact_rb_rb_vv", this->ogc_contacts.rigid_body.vv,
		[&](MappedWorkspace<double>& mws, Element& conn) {
			auto p = this->_get_rb_x1(mws, stark, conn["a"], {conn["p"]});
			auto q = this->_get_rb_x1(mws, stark, conn["b"], {conn["q"]});
			return distance_point_point(p[0], q[0]);
		});
	add_rigid("ogc_contact_rb_rb_ev", this->ogc_contacts.rigid_body.ev,
		[&](MappedWorkspace<double>& mws, Element& conn) {
			auto p = this->_get_rb_x1(mws, stark, conn["a"], {conn["p"]});
			auto e = this->_get_rb_x1(mws, stark, conn["b"], {conn["e0"], conn["e1"]});
			return distance_point_line(p[0], e[0], e[1]);
		});
	add_rigid("ogc_contact_rb_rb_fv", this->ogc_contacts.rigid_body.fv,
		[&](MappedWorkspace<double>& mws, Element& conn) {
			auto p = this->_get_rb_x1(mws, stark, conn["a"], {conn["p"]});
			auto t = this->_get_rb_x1(mws, stark, conn["b"], {conn["t0"], conn["t1"], conn["t2"]});
			return distance_point_plane(p[0], t[0], t[1], t[2]);
		});
	add_edge_edge("ogc_contact_rb_rb_ee_", this->ogc_contacts.rigid_body.ee,
		[&](MappedWorkspace<double>& mws, Element& conn) {
			auto [ea, ea_rest] = this->_get_rb_edge(mws, stark, conn["a"], {conn["ea0"], conn["ea1"]});
			auto [eb, eb_rest] = this->_get_rb_edge(mws, stark, conn["b"], {conn["eb0"], conn["eb1"]});
			return std::make_tuple(ea, eb, ea_rest, eb_rest);
		});

	auto add_mixed = [&](const std::string& label, auto& weighted_conn, auto distance) {
		auto* weights = &weighted_conn.weights;
		stark.global_potential->add_potential(label, weighted_conn.conn,
			[&, distance, weights](MappedWorkspace<double>& mws, Element& conn) {
				return this->_ogc_barrier_potential(
					mws, distance(mws, conn), conn["idx"], conn["group_a"], conn["group_b"], *weights);
			});
	};
	add_mixed("ogc_contact_rb_d_vv", this->ogc_contacts.mixed.vv,
		[&](MappedWorkspace<double>& mws, Element& conn) {
			auto p = this->_get_rb_x1(mws, stark, conn["rb"], {conn["p"]});
			auto q = this->_get_d_x1(mws, stark, {conn["q"]});
			return distance_point_point(p[0], q[0]);
		});
	add_mixed("ogc_contact_rb_d_ev", this->ogc_contacts.mixed.rb_vertex_d_edge,
		[&](MappedWorkspace<double>& mws, Element& conn) {
			auto p = this->_get_rb_x1(mws, stark, conn["rb"], {conn["p"]});
			auto e = this->_get_d_x1(mws, stark, {conn["e0"], conn["e1"]});
			return distance_point_line(p[0], e[0], e[1]);
		});
	add_mixed("ogc_contact_d_rb_ev", this->ogc_contacts.mixed.d_vertex_rb_edge,
		[&](MappedWorkspace<double>& mws, Element& conn) {
			auto p = this->_get_d_x1(mws, stark, {conn["p"]});
			auto e = this->_get_rb_x1(mws, stark, conn["rb"], {conn["e0"], conn["e1"]});
			return distance_point_line(p[0], e[0], e[1]);
		});
	add_mixed("ogc_contact_rb_d_fv", this->ogc_contacts.mixed.rb_vertex_d_face,
		[&](MappedWorkspace<double>& mws, Element& conn) {
			auto p = this->_get_rb_x1(mws, stark, conn["rb"], {conn["p"]});
			auto t = this->_get_d_x1(mws, stark, {conn["t0"], conn["t1"], conn["t2"]});
			return distance_point_plane(p[0], t[0], t[1], t[2]);
		});
	add_mixed("ogc_contact_d_rb_fv", this->ogc_contacts.mixed.d_vertex_rb_face,
		[&](MappedWorkspace<double>& mws, Element& conn) {
			auto p = this->_get_d_x1(mws, stark, {conn["p"]});
			auto t = this->_get_rb_x1(mws, stark, conn["rb"], {conn["t0"], conn["t1"], conn["t2"]});
			return distance_point_plane(p[0], t[0], t[1], t[2]);
		});
	add_edge_edge("ogc_contact_rb_d_ee_", this->ogc_contacts.mixed.ee,
		[&](MappedWorkspace<double>& mws, Element& conn) {
			auto [ea, ea_rest] = this->_get_rb_edge(mws, stark, conn["rb"], {conn["ea0"], conn["ea1"]});
			auto [eb, eb_rest] = this->_get_d_edge(mws, stark, {conn["eb0"], conn["eb1"]});
			return std::make_tuple(ea, eb, ea_rest, eb_rest);
		});
}


/* ========================================================================================== */
/* ===================================  IPC POTENTIAL HELPERS  ============================== */
/* ========================================================================================== */

Scalar EnergyFrictionalContactIPC::_barrier_potential(const Scalar& d, const Scalar& dhat, const Scalar& k)
{
	if (this->ipc_barrier_type == IPCBarrierTypeIPC::Cubic) {
		return k * (dhat - d).powN(3) / 3.0;
	} else {
		return -k * (dhat - d).powN(2) * log(d / dhat);
	}
}

std::pair<Scalar, Scalar> EnergyFrictionalContactIPC::_ogc_barrier_potential(
	MappedWorkspace<double>& mws,
	const Scalar& d,
	const Index& contact_idx,
	const Index& group_a,
	const Index& group_b,
	const std::vector<double>& weights)
{
	Scalar dhat = this->_get_contact_distance(mws, group_a, group_b);
	Scalar stiffness = mws.make_scalar(this->contact_stiffness);
	Scalar weight = mws.make_scalar(weights, contact_idx);
	return {weight * this->_barrier_potential(d, dhat, stiffness), d < dhat};
}

double EnergyFrictionalContactIPC::_barrier_force(const double d, const double dhat, const double k)
{
	if (this->ipc_barrier_type == IPCBarrierTypeIPC::Cubic) {
		return k * std::pow(dhat - d, 2);
	} else {
		return -(k * (dhat - d) * (2.0 * d * std::log(d / dhat) + d - dhat)) / d;
	}
}
Scalar EnergyFrictionalContactIPC::_edge_edge_mollifier(const std::vector<Vector>& ea, const std::vector<Vector>& eb, const std::vector<Vector>& ea_rest, const std::vector<Vector>& eb_rest)
{
	Scalar eps_x = 1e-3 * (ea_rest[0] - ea_rest[1]).squared_norm() * (eb_rest[0] - eb_rest[1]).squared_norm();
	Scalar x = (ea[1] - ea[0]).cross3(eb[1] - eb[0]).squared_norm();
	Scalar x_div_eps_x = x / eps_x;
	Scalar f = (-x_div_eps_x + 2.0) * x_div_eps_x;
	return branch(x > eps_x, 1.0, f);
}
Scalar EnergyFrictionalContactIPC::_friction_potential(const Vector& v, const Scalar& fn, const Scalar& mu, const Matrix& T, const Scalar& epsv, const Scalar& dt)
{
	constexpr double PERTURBATION = 1e-9;
	Vector vt = T * v;
	Vector ut = vt * dt;
	ut[0] += 1.13 * PERTURBATION;
	ut[1] -= 1.07 * PERTURBATION;
	Scalar u = ut.norm();
	Scalar epsu = dt * epsv;
	if (this->ipc_friction_type == IPCFrictionTypeIPC::C0) {
		Scalar k = mu * fn / epsu;
		Scalar eps = mu * fn / (2.0 * k);
		Scalar E_stick = 0.5 * k * u.powN(2);
		Scalar E_slide = mu * fn * (u - eps);
		return branch(u < epsu, E_stick, E_slide);
	} else {
		Scalar E_stick = mu * fn * (-u * u * u / (3.0 * epsu.powN(2)) + u * u / epsu + epsu / 3.0);
		Scalar E_slide = mu * fn * u;
		return branch(u < epsu, E_stick, E_slide);
	}
}

Scalar EnergyFrictionalContactIPC::_get_contact_distance(MappedWorkspace<double>& mws, const Index& group_a, const Index& group_b)
{
	Scalar dhat_a = mws.make_scalar(this->contact_thicknesses, group_a);
	Scalar dhat_b = mws.make_scalar(this->contact_thicknesses, group_b);
	return dhat_a + dhat_b;
}
Scalar EnergyFrictionalContactIPC::_barrier_potential(MappedWorkspace<double>& mws, const Stark& stark, const Scalar& d, const Index& group_a, const Index& group_b)
{
	Scalar dhat = this->_get_contact_distance(mws, group_a, group_b);
	Scalar k = mws.make_scalar(this->contact_stiffness);
	return this->_barrier_potential(d, dhat, k);
}
Scalar EnergyFrictionalContactIPC::_edge_edge_mollified_barrier_potential(MappedWorkspace<double>& mws, const Stark& stark, const Scalar& d, const std::vector<Vector>& ea, const std::vector<Vector>& eb, const std::vector<Vector>& ea_rest, const std::vector<Vector>& eb_rest, const Index& group_a, const Index& group_b)
{
	Scalar k = mws.make_scalar(this->contact_stiffness);
	Scalar dhat = this->_get_contact_distance(mws, group_a, group_b);
	return this->_edge_edge_mollifier(ea, eb, ea_rest, eb_rest) * this->_barrier_potential(d, dhat, k);
}
Scalar EnergyFrictionalContactIPC::_friction_potential(MappedWorkspace<double>& mws, const Stark& stark, const Vector& v, const Index& contact_idx, const FrictionContact& contact)
{
	Matrix T = mws.make_matrix(contact.T, { 2, 3 }, contact_idx);
	Scalar mu = mws.make_scalar(contact.mu, contact_idx);
	Scalar fn = mws.make_scalar(contact.fn, contact_idx);
	Scalar epsv = mws.make_scalar(this->global_params.friction_stick_slide_threshold);
	Scalar dt = mws.make_scalar(stark.dt);
	return this->_friction_potential(v, fn, mu, T, epsv, dt);
}
Scalar EnergyFrictionalContactIPC::_friction_point_edge(MappedWorkspace<double>& mws, const Stark& stark, const Vector& vp, const std::vector<Vector>& ve, const Index& contact_idx, FrictionPointEdge& data)
{
	Vector bary = mws.make_vector(data.bary, contact_idx);
	Vector va = vp;
	Vector vb = bary[0] * ve[0] + bary[1] * ve[1];
	return this->_friction_potential(mws, stark, vb - va, contact_idx, data.contact);
}
Scalar EnergyFrictionalContactIPC::_friction_point_triangle(MappedWorkspace<double>& mws, const Stark& stark, const Vector& vp, const std::vector<Vector>& vt, const Index& contact_idx, FrictionPointTriangle& data)
{
	Vector bary = mws.make_vector(data.bary, contact_idx);
	Vector va = vp;
	Vector vb = bary[0] * vt[0] + bary[1] * vt[1] + bary[2] * vt[2];
	return this->_friction_potential(mws, stark, vb - va, contact_idx, data.contact);
}
Scalar EnergyFrictionalContactIPC::_friction_edge_edge(MappedWorkspace<double>& mws, const Stark& stark, const std::vector<Vector>& vea, const std::vector<Vector>& veb, const Index& contact_idx, FrictionEdgeEdge& data)
{
	Vector bary = mws.make_vector(data.bary, contact_idx);
	Vector va = vea[0] + bary[0] * (vea[1] - vea[0]);
	Vector vb = veb[0] + bary[1] * (veb[1] - veb[0]);
	return this->_friction_potential(mws, stark, vb - va, contact_idx, data.contact);
}


/* ========================================================================================== */
/* ===================================  SYMX DATA GETTERS  ================================== */
/* ========================================================================================== */

std::vector<Vector> EnergyFrictionalContactIPC::_get_rb_v1(MappedWorkspace<double>& mws, const Stark& stark, const Index& rb_idx, const std::vector<Index>& conn)
{
	Scalar dt = mws.make_scalar(stark.dt);
	std::vector<Vector> x_loc = mws.make_vectors(this->rigidbody_local_vertices.data, conn);
	return this->rb->get_v1(mws, rb_idx, x_loc, dt);
}
std::vector<Vector> EnergyFrictionalContactIPC::_get_rb_x1(MappedWorkspace<double>& mws, const Stark& stark, const Index& rb_idx, const std::vector<Index>& conn)
{
	Scalar dt = mws.make_scalar(stark.dt);
	std::vector<Vector> x_loc = mws.make_vectors(this->rigidbody_local_vertices.data, conn);
	return this->rb->get_x1(mws, rb_idx, x_loc, dt);
}
std::vector<Vector> EnergyFrictionalContactIPC::_get_rb_X(MappedWorkspace<double>& mws, const std::vector<Index>& conn)
{
	return mws.make_vectors(this->rigidbody_local_vertices.data, conn);
}
std::array<std::vector<Vector>, 2> EnergyFrictionalContactIPC::_get_rb_edge(MappedWorkspace<double>& mws, const Stark& stark, const Index& rb_idx, const std::vector<Index>& conn)
{
	std::vector<Vector> ea = this->_get_rb_x1(mws, stark, rb_idx, { conn[0], conn[1] });
	std::vector<Vector> ea_rest = this->_get_rb_X(mws, { conn[0], conn[1] });
	return { ea, ea_rest };
}
std::array<std::vector<Vector>, 3> EnergyFrictionalContactIPC::_get_rb_edge_point(MappedWorkspace<double>& mws, const Stark& stark, const Index& rb_idx, const std::vector<Index>& conn)
{
	std::vector<Vector> ea = this->_get_rb_x1(mws, stark, rb_idx, { conn[0], conn[1] });
	std::vector<Vector> ea_rest = this->_get_rb_X(mws, { conn[0], conn[1] });
	std::vector<Vector> p = this->_get_rb_x1(mws, stark, rb_idx, { conn[2] });
	return { ea, ea_rest, p };
}
std::vector<Vector> EnergyFrictionalContactIPC::_get_d_v1(MappedWorkspace<double>& mws, const std::vector<Index>& conn)
{
	return mws.make_vectors(this->dyn->v1.data, conn);
}
std::vector<Vector> EnergyFrictionalContactIPC::_get_d_x1(MappedWorkspace<double>& mws, const Stark& stark, const std::vector<Index>& conn)
{
	std::vector<Vector> v1 = mws.make_vectors(this->dyn->v1.data, conn);
	std::vector<Vector> x0 = mws.make_vectors(this->dyn->x0.data, conn);
	Scalar dt = mws.make_scalar(stark.dt);
	return time_integration(x0, v1, dt);
}
std::vector<Vector> EnergyFrictionalContactIPC::_get_d_X(MappedWorkspace<double>& mws, const std::vector<Index>& conn)
{
	return mws.make_vectors(this->dyn->X.data, conn);
}
std::array<std::vector<Vector>, 2> EnergyFrictionalContactIPC::_get_d_edge(MappedWorkspace<double>& mws, const Stark& stark, const std::vector<Index>& conn)
{
	std::vector<Vector> ea = this->_get_d_x1(mws, stark, { conn[0], conn[1] });
	std::vector<Vector> ea_rest = this->_get_d_X(mws, { conn[0], conn[1] });
	return { ea, ea_rest };
}
std::array<std::vector<Vector>, 3> EnergyFrictionalContactIPC::_get_d_edge_point(MappedWorkspace<double>& mws, const Stark& stark, const std::vector<Index>& conn)
{
	std::vector<Vector> ea = this->_get_d_x1(mws, stark, { conn[0], conn[1] });
	std::vector<Vector> ea_rest = this->_get_d_X(mws, { conn[0], conn[1] });
	std::vector<Vector> p = this->_get_d_x1(mws, stark, { conn[2] });
	return { ea, ea_rest, p };
}


/* ========================================================================================== */
/* ===================================  MISC  =============================================== */
/* ========================================================================================== */

std::string EnergyFrictionalContactIPC::_get_contact_label(const std::string physical_system, const std::string pair) const
{
	std::string out = "ipc_contact_" + physical_system + "_" + pair + "_";
	out += (this->ipc_barrier_type == IPCBarrierTypeIPC::Cubic) ? "cubic" : "log";
	return out;
}
std::string EnergyFrictionalContactIPC::_get_friction_label(const std::string physical_system, const std::string pair) const
{
	std::string out = "ipc_friction_" + physical_system + "_" + pair + "_";
	out += (this->ipc_friction_type == IPCFrictionTypeIPC::C0) ? "C0" : "C1";
	return out;
}

#endif // STARK_USE_IPC_TOOLKIT
