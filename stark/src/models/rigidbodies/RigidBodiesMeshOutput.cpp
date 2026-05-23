#include "RigidBodiesMeshOutput.h"

#include "../../utils/include.h"

using namespace stark;


RigidBodiesMeshOutput::RigidBodiesMeshOutput(Stark& stark, spRigidBodyDynamics rb)
	: rb(rb)
{
	stark.callbacks->add_write_frame([&]() { this->_write_frame(stark); });
}
void RigidBodiesMeshOutput::add_point_mesh(const std::string& label, const RigidBodyHandler& rb, const std::vector<Eigen::Vector3d>& vertices)
{
	rb.exit_if_not_valid("RigidBodiesMeshOutput::add_point_set");
	std::vector<std::array<int, 1>> conn(vertices.size());
	for (int i = 0; i < (int)vertices.size(); i++) {
		conn[i] = { i };
	}

	this->points.push_back({ rb, vertices });
	this->point_groups.add_to_group(label, (int)this->points.size() - 1);
}
void RigidBodiesMeshOutput::add_segment_mesh(const std::string& label, const RigidBodyHandler& rb, const std::vector<Eigen::Vector3d>& vertices, const std::vector<std::array<int, 2>>& conn)
{
	rb.exit_if_not_valid("RigidBodiesMeshOutput::add_segment_mesh");
	this->segments.push_back({ rb, vertices, conn });
	this->segment_groups.add_to_group(label, (int)this->segments.size() - 1);
}
void RigidBodiesMeshOutput::add_triangle_mesh(const std::string& label, const RigidBodyHandler& rb, const std::vector<Eigen::Vector3d>& vertices, const std::vector<std::array<int, 3>>& conn)
{
	rb.exit_if_not_valid("RigidBodiesMeshOutput::add_triangle_mesh");
	this->triangles.push_back({ rb, vertices, conn });
	this->triangle_groups.add_to_group(label, (int)this->triangles.size() - 1);
}
void RigidBodiesMeshOutput::add_tet_mesh(const std::string& label, const RigidBodyHandler& rb, const std::vector<Eigen::Vector3d>& vertices, const std::vector<std::array<int, 4>>& conn)
{
	rb.exit_if_not_valid("RigidBodiesMeshOutput::add_tet_mesh");
	this->tets.push_back({ rb, vertices, conn });
	this->tet_groups.add_to_group(label, (int)this->tets.size() - 1);
}

void RigidBodiesMeshOutput::_write_frame(Stark& stark)
{
	this->_write(stark, this->point_groups, this->points, conn_points);
	this->_write(stark, this->segment_groups, this->segments, conn_segments);
	this->_write(stark, this->triangle_groups, this->triangles, conn_triangles);
	this->_write(stark, this->tet_groups, this->tets, conn_tets);
}

template<std::size_t N>
inline void RigidBodiesMeshOutput::_write(Stark& stark, const MeshOutputGroups& groups, const std::vector<Mesh<N>>& meshes, std::vector<std::array<int, N>>& conn_buffer)
{
	for (const auto& it : groups.groups) {
		const std::string& label = it.first;
		const std::unordered_set<int>& group = it.second;

		this->vertices.clear();
		conn_buffer.clear();
		for (const int local_idx : group) {
			const auto& mesh = meshes[local_idx];
			const auto rb = mesh.rb;

			const int out_vertex_offset = (int)this->vertices.size();
			for (const Eigen::Vector3d& vertex : mesh.vertices) {
				this->vertices.push_back(rb.transform_local_to_global_point(vertex));
			}

			for (const auto& elem : mesh.conn) {
				conn_buffer.push_back(elem);
				for (int i = 0; i < N; i++) {
					conn_buffer.back()[i] += out_vertex_offset;
				}
			}
		}
		write_VTK(stark.get_frame_path(label) + ".vtk", this->vertices, conn_buffer);
	}
}
// Forward declaration
template void RigidBodiesMeshOutput::_write<1>(Stark& stark, const MeshOutputGroups& groups, const std::vector<RigidBodiesMeshOutput::Mesh<1>>& meshes, std::vector<std::array<int, 1>>& conn_buffer);
template void RigidBodiesMeshOutput::_write<2>(Stark& stark, const MeshOutputGroups& groups, const std::vector<RigidBodiesMeshOutput::Mesh<2>>& meshes, std::vector<std::array<int, 2>>& conn_buffer);
template void RigidBodiesMeshOutput::_write<3>(Stark& stark, const MeshOutputGroups& groups, const std::vector<RigidBodiesMeshOutput::Mesh<3>>& meshes, std::vector<std::array<int, 3>>& conn_buffer);
template void RigidBodiesMeshOutput::_write<4>(Stark& stark, const MeshOutputGroups& groups, const std::vector<RigidBodiesMeshOutput::Mesh<4>>& meshes, std::vector<std::array<int, 4>>& conn_buffer);
