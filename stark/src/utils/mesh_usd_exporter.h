#pragma once

#include "Mesh.h"
#include "pxr/usd/sdf/path.h"


// If the file exists, this function will simply return (safe to call once).
bool InitPrim(const std::string& usdPath,
              const pxr::SdfPath& primPath,
              const std::vector<std::array<int, 3>>& face_conn);
bool AppendFrame(const std::string& usdPath,
                 const pxr::SdfPath& primPath,
                 double frameTime,
                 const std::vector<Eigen::Vector3d>& points,
                 const std::vector<Eigen::Vector2d>& uvs);
bool writePrimitiveFrame(const std::string& usdPath, const pxr::SdfPath& primPath, const stark::Mesh<3>& mesh,
                         const std::vector<Eigen::Vector2d>& uvs,
                         int frame,
                         bool init = false);
bool InitStage(const std::string& usdPath, bool clean = false,
               double startTime = 0, double endTime = 0);
