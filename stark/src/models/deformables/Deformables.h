#pragma once
#include "DeformablesMeshOutput.h"
#include "deformables_energies_include.h"
#include "surface/EnergyQuadraticBending.h"
#include "surface/EnergyTriangleArea.h"
#include "surface/EnergyTriangleStretching.h"
#include "volume/EnergyPrescribedPositionsTetBarycentric.h"

namespace stark
{
	class Deformables
	{
	public:
		/* Methods */
		Deformables(Stark& stark, spPointDynamics dyn);

		/* Fields */
		std::shared_ptr<DeformablesMeshOutput> output;

		// Models
		spPointDynamics point_sets;
		std::shared_ptr<EnergyLumpedInertia> lumped_inertia;
		std::shared_ptr<EnergyPrescribedPositions> prescribed_positions;
		std::shared_ptr<EnergyPrescribedPositionsTetBarycentric> tet_prescribed_positions;
		std::shared_ptr<EnergySegmentStrain> segment_strain;
		std::shared_ptr<EnergyTriangleStrain> triangle_strain;
		std::shared_ptr<EnergyTriangleStretching> triangle_stretching;
		std::shared_ptr<EnergyDiscreteShells> discrete_shells;
		std::shared_ptr<EnergyQuadraticBending> triangle_quadratic_bending;
		std::shared_ptr<EnergyTetStrain> tet_strain;
		std::shared_ptr<EnergyTriangleArea> triangle_area;
	};
}
