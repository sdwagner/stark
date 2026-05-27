#include "mesh_usd_exporter.h"
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>

#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/attribute.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/vt/array.h>
#include <pxr/usd/usd/timeCode.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>


using namespace pxr;

bool writePrimitiveFrame(const std::string& usdPath, const SdfPath& primPath, const stark::Mesh<3>& mesh, const std::vector<Eigen::Vector2d>& uvs, int frame,
                   bool init)
{
    if (init && !InitPrim(usdPath, primPath, mesh.conn))
    {
        std::cerr << "Failed init of Primitive" << "\n";
        return false;
    }

    if (!AppendFrame(usdPath, primPath, frame, mesh.vertices, uvs))
    {
        std::cerr << "Failed writing frame: " << frame << "\n";
        return false;
    }
    return true;
}

// Simple helper: turn a vector-of-3-floats into VtArray<GfVec3f>
static VtArray<GfVec3f> VecToVt(const std::vector<Eigen::Vector3d>& pts)
{
    VtArray<GfVec3f> arr(pts.size());
    for (size_t i = 0; i < pts.size(); ++i)
    {
        arr[i] = GfVec3f(pts[i][0], pts[i][1], pts[i][2]);
    }
    return arr;
}

static VtArray<GfVec2f> VecToVt(const std::vector<Eigen::Vector2d>& pts)
{
    VtArray<GfVec2f> arr(pts.size());
    for (size_t i = 0; i < pts.size(); ++i)
    {
        arr[i] = GfVec2f(pts[i][0], pts[i][1]);
    }
    return arr;
}

bool InitStage(const std::string& usdPath, bool clean,
                   double startTime, double endTime)
{
    // If layer exists already, don't clobber it.
    if (SdfLayer::FindOrOpen(usdPath) && !clean)
    {
        //std::cout << "[InitializeUsd] USD file already exists: " << usdPath << "\n";
        return true;
    }

    auto stage = UsdStage::CreateNew(usdPath);
    if (!stage)
    {
        std::cerr << "Failed to create stage: " << usdPath << "\n";
        return false;
    }

    // Optionally set extent (left unset here). Set time range.
    stage->SetStartTimeCode(startTime);
    stage->SetEndTimeCode(endTime);
    stage->GetRootLayer()->Save();
    return true;
}

// Initialize a USD file with the mesh topology and an empty points attribute.
// If the file exists, this function will simply return (safe to call once).
bool InitPrim(const std::string& usdPath,
                   const SdfPath& primPath,
                   const std::vector<std::array<int, 3>>& face_conn)
{

    auto stage = UsdStage::Open(usdPath);
    if (!stage)
    {
        std::cerr << "Failed to open USD stage: " << usdPath << "\n";
        return false;
    }

    // Define a Mesh prim at primPath
    UsdGeomMesh mesh = UsdGeomMesh::Define(stage, primPath);

    // Set topology (non-time-sampled; constant)
    VtArray<int> vtCounts(face_conn.size());
    for (size_t i = 0; i < face_conn.size(); ++i) vtCounts[i] = 3;

    VtArray<int> vtIndices(face_conn.size() * 3);
    for (size_t i = 0; i < face_conn.size(); ++i)
    {
        vtIndices[3 * i] = face_conn[i][0];
        vtIndices[3 * i + 1] = face_conn[i][1];
        vtIndices[3 * i + 2] = face_conn[i][2];
    }

    mesh.CreateFaceVertexCountsAttr().Set(vtCounts);
    mesh.CreateFaceVertexIndicesAttr().Set(vtIndices);

    UsdGeomPrimvarsAPI primvarsAPI(mesh);
    UsdGeomPrimvar st = primvarsAPI.CreatePrimvar(
        TfToken("st"),
        SdfValueTypeNames->TexCoord2fArray,
        UsdGeomTokens->faceVarying
    );

    // Create empty points attr (will hold time-samples later)
    mesh.CreatePointsAttr();

    // Save to disk
    stage->GetRootLayer()->Save();

    //std::cout << "[InitializeUsd] Created USD: " << usdPath << " with prim " << primPath.GetString() << "\n";
    return true;
}

// Append a single time-sample for vertex positions at the given frame time.
// `points` must match the mesh's vertex count (topology is constant).
bool AppendFrame(const std::string& usdPath,
                 const SdfPath& primPath,
                 double frameTime,
                 const std::vector<Eigen::Vector3d>& points,
                 const std::vector<Eigen::Vector2d>& uvs)
{
    // Open stage for editing
    auto stage = UsdStage::Open(usdPath);
    if (!stage)
    {
        std::cerr << "Failed to open USD stage: " << usdPath << "\n";
        return false;
    }

    UsdPrim prim = stage->GetPrimAtPath(primPath);
    if (!prim)
    {
        std::cerr << "Prim not found at path: " << primPath.GetString() << "\n";
        return false;
    }

    UsdGeomMesh mesh(prim);
    if (!mesh)
    {
        std::cerr << "Prim is not a UsdGeomMesh at: " << primPath.GetString() << "\n";
        return false;
    }

    // Convert to VtArray and set the time-sample on the points attribute
    VtArray<GfVec3f> vtPoints = VecToVt(points);
    UsdAttribute pointsAttr = mesh.GetPointsAttr();
    if (!pointsAttr)
    {
        std::cerr << "Mesh does not have a points attribute (unexpected).\n";
        return false;
    }

    UsdTimeCode timeCode(frameTime);
    pointsAttr.ClearAtTime(timeCode);
    if (!pointsAttr.Set(vtPoints, timeCode))
    {
        std::cerr << "Failed to set points at time " << frameTime << "\n";
        // continue to try saving? here we fail.
        return false;
    }

    if (!uvs.empty())
    {
        VtArray<GfVec2f> vtUV = VecToVt(uvs);
        UsdGeomPrimvarsAPI primvarsAPI(mesh);
        UsdGeomPrimvar st = primvarsAPI.GetPrimvar(TfToken("st"));
        if (!st)
        {
            std::cerr << "Mesh does not have UV coordinates (unexpected).\n";
            return false;
        }
        st.Set(vtUV, timeCode);
    }


    // Update stage time-range if necessary
    double start = stage->GetStartTimeCode();
    double end = stage->GetEndTimeCode();
    if (frameTime < start)
    {
        stage->SetStartTimeCode(frameTime);
    }
    if (frameTime > end)
    {
        stage->SetEndTimeCode(frameTime);
    }

    // Save
    stage->GetRootLayer()->Save();

    //std::cout << "[AppendFrame] Wrote frame " << frameTime << " to " << usdPath
    //          << (changedRange ? " (extended time range)" : "") << "\n";
    return true;
}
