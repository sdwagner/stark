# Frictional Contact

STARK uses an **IPC (Incremental Potential Contact)** based formulation for frictional contact.
It is guaranteed to be intersection-free and handles all contact pairs (deformable–deformable, rigid–deformable, rigid–rigid) in a unified framework.

Contact is handled by `simulation.interactions->contact` (C++) or `simulation.interactions().contact()` (Python).

## Contact Handler
Every mesh that should collide is registered as a contact object and receives a `ContactHandler`, used to configure friction, disable pairs, or update contact thickness.

The actual motion comes from the underlying `PointSetHandler` or `RigidBodyHandler`; the contact object holds only the collision geometry.

Contact geometry is registered either through presets (recommended) or manually. Details at the end of this page.

## Global Parameters

Set once before the simulation starts:

```cpp
// C++
stark::EnergyFrictionalContact::GlobalParams params;

// Contact thickness (d_hat): the distance at which contact forces activate
// MUST be set; no sensible default exists — it depends on your scene scale
params.default_contact_thickness = 0.001;  // 1 mm

// Stiffness range (automatically hardened when penetration is detected)
params.min_contact_stiffness = 1e4;
params.max_contact_stiffness = 1e20;

// Friction: stick-to-slide threshold (relative tangential displacement)
params.friction_stick_slide_threshold = 0.01;

// Enable/disable specific contact modes
params.collisions_enabled        = true;
params.friction_enabled          = true;
params.triangle_point_enabled    = true;
params.edge_edge_enabled         = true;
params.intersection_test_enabled = true;

simulation.interactions->contact->set_global_params(params);
```

### Offset Geometric Contact (OGC)

When STARK is built with `STARK_IPC_TOOLKIT_CONTACT=ON`, normal contact can use
IPC Toolkit's Offset Geometric Contact collision set and trust-region step
filter. The default remains IPC.

```cpp
stark::ContactGlobalParams params;
params.set_contact_method(stark::ContactMethod::OGC)
    .set_ogc_relaxed_radius_scaling(0.9)
    .set_ogc_update_threshold(0.01)
    .set_ogc_max_query_radius(0.01);
simulation.interactions->contact->set_global_params(params);
```

Set the contact method before the simulation is initialized; changing between
IPC and OGC afterward is rejected. OGC currently supports normal contact only;
stored friction coefficients are ignored while OGC is active.
Both triangle-point and edge-edge contact modes must remain enabled because the
OGC feasible-region construction uses the complete collision stencil set.
The OGC implementation keeps the collision set fixed during each Newton line
search and filters the Newton direction before Armijo backtracking. Deformable
steps are filtered per velocity block; rigid translation and rotation are
filtered together and checked against the vertex trust regions after applying
the exact rigid transform. The query radius grows from IPC Toolkit's
contact-distance-derived minimum toward ordinary Newton motion, but is capped
by `ogc_max_query_radius` (in scene length units). If a proposed correction
exceeds that budget, STARK keeps the current local OGC query and uses swept CCD
to scale the whole Newton direction for that iteration. This is important for
localized fast motion: inflating the combined collision mesh from one fast
vertex can create a prohibitively large global broad-phase query. Increase the
cap only when the extra candidate memory is acceptable.


## Setting up frictional contact
### Setting Friction Between Pairs

Friction must be explicitly enabled between each pair of contact objects.
The Coulomb friction coefficient $\mu$ is set per pair:

```cpp
// C++
ch_cloth.set_friction(ch_box, 0.5);   // μ = 0.5 between cloth and box
ch_cloth.set_friction(ch_cloth, 0.1); // self-contact with μ = 0.1
```

Pairs without an explicit `set_friction` call have no friction forces between them.

### Per-Object Contact Thickness

You can override the global default thickness per object:

```cpp
ch_cloth.set_contact_thickness(0.0005);  // 0.5 mm
```

### Disabling Collisions Between Pairs

To exclude specific pairs from collision detection entirely (e.g. objects that start overlapping):

```cpp
ch_a.disable_collision(ch_b);
```

## Stiffness Hardening

When Newton's Method cannot resolve an intersection, contact stiffness is doubled and the step is retried.
This repeats until penetration resolves or `max_contact_stiffness` is reached.
You will see messages like:

```
Penetration couldn't be avoided. Contact stiffness hardened from 1.0e+04 to 2.0e+04.
```

This is normal in contact-heavy scenes.


## Registering Contact Objects

### Contact through presets

Presets are the recommended path for standard objects. Deformable and rigid-body presets automatically register suitable collision geometry and return a contact handle in the preset handler.

```cpp
// Deformable object with contact
stark::Volume::Params rubber = stark::Volume::Params::Soft_Rubber();
rubber.contact.contact_thickness = 0.001;

auto [V, T, body] = simulation.presets->deformables->add_volume(
    "body",
    vertices,
    tets,
    rubber
);

stark::ContactHandler body_contact = body.contact;

// Rigid body with contact
auto [floor_V, floor_T, floor] = simulation.presets->rigidbodies->add_box(
    "floor",
    1.0,
    Eigen::Vector3d(2.0, 2.0, 0.05),
    stark::ContactParams().set_contact_thickness(0.001)
);

stark::ContactHandler floor_contact = floor.contact;
```

### Manual contact registration

For custom objects, register the contact geometry directly.
The collision mesh indices refer to vertices of the `PointSetHandler`.

```cpp
stark::PointSetHandler point_set = /* your deformable point set */;
std::vector<std::array<int, 3>> surface_triangles = /* collision surface */;

stark::ContactHandler cloth_contact =
    simulation.interactions->contact->add_triangles(
        point_set,
        surface_triangles,
        stark::ContactParams().set_contact_thickness(0.001)
    );
```

For segment-like objects, use edges instead of triangles:

```cpp
std::vector<std::array<int, 2>> edges = /* collision edges */;

stark::ContactHandler rod_contact =
    simulation.interactions->contact->add_edges(
        point_set,
        edges,
        stark::ContactParams().set_contact_thickness(0.001)
    );
```

#### Contact on a subset of a point set

Sometimes the simulated point set contains more vertices than the collision mesh. For example, a tetrahedral volume may use only its boundary triangles for contact.

Use the overload with `point_set_map` in that case. The collision mesh is indexed locally into `point_set_map`, and `point_set_map[i]` maps collision vertex `i` to a vertex in the full point set.

```cpp
stark::PointSetHandler point_set = /* full deformable point set */;

std::vector<int> surface_to_point_set = /* collision vertex -> point-set vertex */;
std::vector<std::array<int, 3>> surface_triangles = /* indexed into surface_to_point_set */;

stark::ContactHandler contact =
    simulation.interactions->contact->add_triangles(
        point_set,
        surface_triangles,
        surface_to_point_set,
        stark::ContactParams().set_contact_thickness(0.001)
    );
```

This is the manual equivalent of what presets do for volumetric objects: the simulation uses tetrahedra, but contact is registered on the boundary surface.
