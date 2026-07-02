#pragma once

#ifdef STARK_USE_IPC_TOOLKIT
    #include "EnergyFrictionalContactIPC.h"
#else
    #include "EnergyFrictionalContact.h"
#endif

#include "EnergyAttachments.h"
#include "../rigidbodies/RigidBodyHandler.h"
#include "../deformables/PointSetHandler.h"

namespace stark
{
#ifdef STARK_USE_IPC_TOOLKIT
	using ContactEnergy = EnergyFrictionalContactIPC;
#else
	using ContactEnergy = EnergyFrictionalContact;
#endif

	// Stable type aliases used throughout presets and user code.
	// These point to the active backend's types regardless of which backend is compiled.
	using ContactParams       = ContactEnergy::Params;
	using ContactGlobalParams = ContactEnergy::GlobalParams;
	using ContactHandler      = ContactEnergy::Handler;

	class Interactions
	{
	public:
		/* Methods */
		Interactions(Stark& stark, spPointDynamics dyn, spRigidBodyDynamics rb);

		/* Fields */
		std::shared_ptr<EnergyAttachments> attachments;
		std::shared_ptr<ContactEnergy> contact;

	private:
		/* Fields */
		spPointDynamics dyn;
		spRigidBodyDynamics rb;
	};
}
