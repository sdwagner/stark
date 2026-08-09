#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <stark>

using namespace Catch::Matchers;

#ifdef STARK_USE_IPC_TOOLKIT
namespace {
double run_ogc_sliding_scene(double mu, bool friction_enabled = true)
{
    stark::Settings settings;
    settings.output.simulation_name = "ogc_friction_slide";
    settings.output.output_directory = std::string(STARK_TESTS_OUTPUT_DIR) + "/test_output";
    settings.output.enable_output = false;
    settings.output.enable_frame_writes = false;
    settings.output.console_verbosity = symx::Verbosity::Minimal;
    settings.simulation.gravity.setZero();
    settings.simulation.initial_time_step_size = 0.02;
    settings.simulation.max_time_step_size = 0.02;
    settings.simulation.use_adaptive_time_step = false;
    settings.newton.linear_solver = symx::LinearSolver::DirectLLT;
    settings.newton.max_iterations = 80;
    settings.newton.residual_tolerance_abs = 1e-9;

    stark::Simulation simulation(settings);
    const std::vector<std::array<int, 3>> triangle{{0, 1, 2}};
    auto floor = simulation.deformables->point_sets->add({
        {-1.0, -1.0, 0.0}, {1.0, -1.0, 0.0}, {0.0, 1.0, 0.0}});
    auto slider = simulation.deformables->point_sets->add({
        {-0.25, -0.25, 0.007}, {0.25, -0.25, 0.007}, {0.0, 0.25, 0.007}});

    simulation.deformables->lumped_inertia->add(
        floor, std::vector<double>(3, 1.0),
        stark::EnergyLumpedInertia::Params().set_enable_gravity(false));
    simulation.deformables->lumped_inertia->add(
        slider, std::vector<double>(3, 1.0),
        stark::EnergyLumpedInertia::Params().set_enable_gravity(false));
    simulation.deformables->prescribed_positions->add(
        floor, floor.all(),
        stark::EnergyPrescribedPositions::Params().set_stiffness(1e12));
    for (int i = 0; i < slider.size(); ++i)
        slider.set_acceleration(i, {20.0, 0.0, -20.0});

    auto floor_contact = simulation.interactions->contact->add_triangles(
        floor, triangle, stark::ContactParams().set_contact_thickness(0.005));
    auto slider_contact = simulation.interactions->contact->add_triangles(
        slider, triangle, stark::ContactParams().set_contact_thickness(0.005));
    floor_contact.disable_collision(floor_contact);
    slider_contact.disable_collision(slider_contact);
    floor_contact.set_friction(slider_contact, mu);
    simulation.interactions->contact->set_global_params(
        stark::ContactGlobalParams()
            .set_contact_method(stark::ContactMethod::OGC)
            .set_friction_enabled(friction_enabled)
            .set_min_contact_stiffness(1e6));

    REQUIRE(simulation.stark.run_one_step());
    double mean_x = 0.0;
    for (int i = 0; i < slider.size(); ++i)
        mean_x += slider.get_position(i).x();
    return mean_x / slider.size();
}

struct PairSlidingResult
{
    double frictionless_x;
    double friction_x;
};

PairSlidingResult run_ogc_pair_sliding_scene()
{
    stark::Settings settings;
    settings.output.simulation_name = "ogc_pair_friction_slide";
    settings.output.output_directory = std::string(STARK_TESTS_OUTPUT_DIR) + "/test_output";
    settings.output.enable_output = false;
    settings.output.enable_frame_writes = false;
    settings.output.console_verbosity = symx::Verbosity::Minimal;
    settings.simulation.gravity.setZero();
    settings.simulation.initial_time_step_size = 0.02;
    settings.simulation.max_time_step_size = 0.02;
    settings.simulation.use_adaptive_time_step = false;
    settings.newton.linear_solver = symx::LinearSolver::DirectLLT;
    settings.newton.max_iterations = 80;
    settings.newton.residual_tolerance_abs = 1e-9;

    stark::Simulation simulation(settings);
    auto floor = simulation.deformables->point_sets->add({
        {-2.0, -2.0, 0.0}, {2.0, -2.0, 0.0}, {2.0, 2.0, 0.0}, {-2.0, 2.0, 0.0}});
    auto frictionless_slider = simulation.deformables->point_sets->add({
        {-0.25, -0.85, 0.007}, {0.25, -0.85, 0.007}, {0.0, -0.35, 0.007}});
    auto friction_slider = simulation.deformables->point_sets->add({
        {-0.25, 0.35, 0.007}, {0.25, 0.35, 0.007}, {0.0, 0.85, 0.007}});
    const std::vector<std::array<int, 3>> floor_triangles{{0, 1, 2}, {0, 2, 3}};
    const std::vector<std::array<int, 3>> slider_triangle{{0, 1, 2}};

    simulation.deformables->lumped_inertia->add(
        floor, std::vector<double>(4, 1.0),
        stark::EnergyLumpedInertia::Params().set_enable_gravity(false));
    simulation.deformables->prescribed_positions->add(
        floor, floor.all(), stark::EnergyPrescribedPositions::Params().set_stiffness(1e12));
    for (auto slider : {frictionless_slider, friction_slider}) {
        simulation.deformables->lumped_inertia->add(
            slider, std::vector<double>(3, 1.0),
            stark::EnergyLumpedInertia::Params().set_enable_gravity(false));
        for (int i = 0; i < slider.size(); ++i)
            slider.set_acceleration(i, {20.0, 0.0, -20.0});
    }

    auto floor_contact = simulation.interactions->contact->add_triangles(
        floor, floor_triangles, stark::ContactParams().set_contact_thickness(0.005));
    auto frictionless_contact = simulation.interactions->contact->add_triangles(
        frictionless_slider, slider_triangle, stark::ContactParams().set_contact_thickness(0.005));
    auto friction_contact = simulation.interactions->contact->add_triangles(
        friction_slider, slider_triangle, stark::ContactParams().set_contact_thickness(0.005));
    floor_contact.disable_collision(floor_contact);
    frictionless_contact.disable_collision(frictionless_contact);
    friction_contact.disable_collision(friction_contact);
    frictionless_contact.disable_collision(friction_contact);
    floor_contact.set_friction(frictionless_contact, 0.0);
    floor_contact.set_friction(friction_contact, 0.8);
    simulation.interactions->contact->set_global_params(
        stark::ContactGlobalParams()
            .set_contact_method(stark::ContactMethod::OGC)
            .set_min_contact_stiffness(1e6));

    REQUIRE(simulation.stark.run_one_step());
    PairSlidingResult result{0.0, 0.0};
    for (int i = 0; i < frictionless_slider.size(); ++i) {
        result.frictionless_x += frictionless_slider.get_position(i).x();
        result.friction_x += friction_slider.get_position(i).x();
    }
    result.frictionless_x /= frictionless_slider.size();
    result.friction_x /= friction_slider.size();
    return result;
}
} // namespace

TEST_CASE("OGC lagged friction resists tangential sliding", "[contact][ogc][friction]")
{
    const double frictionless_x = run_ogc_sliding_scene(0.0);
    const double friction_x = run_ogc_sliding_scene(0.8);
    REQUIRE(friction_x < frictionless_x - 1e-7);
}

TEST_CASE("OGC honors the global friction switch", "[contact][ogc][friction]")
{
    const double frictionless_x = run_ogc_sliding_scene(0.0);
    const double disabled_x = run_ogc_sliding_scene(0.8, false);
    REQUIRE_THAT(disabled_x, WithinAbs(frictionless_x, 1e-8));
}

TEST_CASE("OGC uses per-handler friction coefficients", "[contact][ogc][friction]")
{
    const PairSlidingResult result = run_ogc_pair_sliding_scene();
    REQUIRE(result.friction_x < result.frictionless_x - 1e-7);
}

TEST_CASE("OGC contact parameters are opt-in", "[contact][ogc]")
{
    stark::ContactGlobalParams params;
    REQUIRE(params.get_contact_method() == stark::ContactMethod::IPC);
    REQUIRE_THAT(params.get_ogc_relaxed_radius_scaling(), WithinAbs(0.9, 1e-12));
    REQUIRE_THAT(params.get_ogc_update_threshold(), WithinAbs(0.01, 1e-12));
    REQUIRE_THAT(params.get_ogc_max_query_radius(), WithinAbs(0.01, 1e-12));

    params = stark::ContactGlobalParams().set_contact_method(stark::ContactMethod::OGC)
        .set_ogc_relaxed_radius_scaling(0.8)
        .set_ogc_update_threshold(0.05)
        .set_ogc_max_query_radius(0.05);

    REQUIRE(params.get_contact_method() == stark::ContactMethod::OGC);
    REQUIRE_THAT(params.get_ogc_relaxed_radius_scaling(), WithinAbs(0.8, 1e-12));
    REQUIRE_THAT(params.get_ogc_update_threshold(), WithinAbs(0.05, 1e-12));
    REQUIRE_THAT(params.get_ogc_max_query_radius(), WithinAbs(0.05, 1e-12));
}

TEST_CASE("OGC filters a large deformable contact step", "[contact][ogc][integration]")
{
    stark::Settings settings;
    settings.output.simulation_name = "ogc_deformable_step_filter";
    settings.output.output_directory = std::string(STARK_TESTS_OUTPUT_DIR) + "/test_output";
    settings.output.enable_output = false;
    settings.output.enable_frame_writes = false;
    settings.output.console_verbosity = symx::Verbosity::Minimal;
    settings.simulation.gravity.setZero();
    settings.simulation.initial_time_step_size = 0.1;
    settings.simulation.max_time_step_size = 0.1;
    settings.simulation.use_adaptive_time_step = false;
    settings.newton.linear_solver = symx::LinearSolver::DirectLLT;
    settings.newton.max_iterations = 50;
    settings.newton.residual_tolerance_abs = 1e-8;

    stark::Simulation simulation(settings);
    auto floor = simulation.deformables->point_sets->add({
        {-1.0, -1.0, 0.0}, {1.0, -1.0, 0.0}, {0.0, 1.0, 0.0}});
    auto falling = simulation.deformables->point_sets->add({
        {-0.5, -0.5, 0.03}, {0.5, -0.5, 0.03}, {0.0, 0.5, 0.03}});
    const std::vector<std::array<int, 3>> triangle{{0, 1, 2}};

    simulation.deformables->lumped_inertia->add(
        floor, std::vector<double>(3, 1.0),
        stark::EnergyLumpedInertia::Params().set_enable_gravity(false));
    simulation.deformables->lumped_inertia->add(
        falling, std::vector<double>(3, 1.0),
        stark::EnergyLumpedInertia::Params().set_enable_gravity(false));
    simulation.deformables->prescribed_positions->add(
        floor, floor.all(),
        stark::EnergyPrescribedPositions::Params().set_stiffness(1e12));
    for (int i = 0; i < falling.size(); ++i) {
        falling.set_acceleration(i, {0.0, 0.0, -100.0});
    }

    auto floor_contact = simulation.interactions->contact->add_triangles(
        floor, triangle, stark::ContactParams().set_contact_thickness(0.005));
    auto falling_contact = simulation.interactions->contact->add_triangles(
        falling, triangle, stark::ContactParams().set_contact_thickness(0.005));
    floor_contact.disable_collision(floor_contact);
    falling_contact.disable_collision(falling_contact);
    REQUIRE_NOTHROW(floor_contact.set_friction(falling_contact, 0.1));
    simulation.interactions->contact->set_global_params(
        stark::ContactGlobalParams()
            .set_contact_method(stark::ContactMethod::OGC)
            .set_min_contact_stiffness(1e6));

    REQUIRE(simulation.stark.run_one_step());
    REQUIRE(simulation.stark.current_time_step == 1);
    REQUIRE(simulation.get_callbacks()->run_is_current_collision_state_valid());
    for (int i = 0; i < falling.size(); ++i) {
        REQUIRE(falling.get_position(i).z() > 0.0);
    }
    REQUIRE_THROWS_AS(
        simulation.interactions->contact->set_global_params(
            stark::ContactGlobalParams().set_contact_method(stark::ContactMethod::IPC)),
        std::logic_error);
}

TEST_CASE("OGC uses CCD instead of expanding its query for a large Newton step", "[contact][ogc][integration]")
{
    stark::Settings settings;
    settings.output.simulation_name = "ogc_adaptive_query_radius";
    settings.output.output_directory = std::string(STARK_TESTS_OUTPUT_DIR) + "/test_output";
    settings.output.enable_output = false;
    settings.output.enable_frame_writes = false;
    settings.output.console_verbosity = symx::Verbosity::Minimal;
    settings.simulation.gravity.setZero();
    settings.simulation.initial_time_step_size = 0.1;
    settings.simulation.max_time_step_size = 0.1;
    settings.simulation.use_adaptive_time_step = false;
    settings.newton.linear_solver = symx::LinearSolver::DirectLLT;
    settings.newton.max_iterations = 20;
    settings.newton.residual_tolerance_abs = 1e-8;

    stark::Simulation simulation(settings);
    const std::vector<Eigen::Vector3d> moving_vertices{
        {-0.5, -0.5, 0.0}, {0.5, -0.5, 0.0}, {0.0, 0.5, 0.0}};
    const std::vector<Eigen::Vector3d> distant_vertices{
        {-0.5, -0.5, 10.0}, {0.5, -0.5, 10.0}, {0.0, 0.5, 10.0}};
    const std::vector<std::array<int, 3>> triangle{{0, 1, 2}};

    auto moving = simulation.deformables->point_sets->add(moving_vertices);
    auto distant = simulation.deformables->point_sets->add(distant_vertices);
    auto moving_prescribed = simulation.deformables->prescribed_positions->add(
        moving, moving.all(),
        stark::EnergyPrescribedPositions::Params().set_stiffness(1e8));
    simulation.deformables->prescribed_positions->add(
        distant, distant.all(),
        stark::EnergyPrescribedPositions::Params().set_stiffness(1e8));
    for (int i = 0; i < moving.size(); ++i) {
        moving_prescribed.set_target_position(
            i, moving_vertices[i] + Eigen::Vector3d(0.3, 0.0, 0.0));
    }

    auto moving_contact = simulation.interactions->contact->add_triangles(
        moving, triangle, stark::ContactParams().set_contact_thickness(0.005));
    auto distant_contact = simulation.interactions->contact->add_triangles(
        distant, triangle, stark::ContactParams().set_contact_thickness(0.005));
    moving_contact.disable_collision(moving_contact);
    distant_contact.disable_collision(distant_contact);
    simulation.interactions->contact->set_global_params(
        stark::ContactGlobalParams()
            .set_contact_method(stark::ContactMethod::OGC)
            .set_ogc_max_query_radius(0.05)
            .set_min_contact_stiffness(1e6));

    REQUIRE(simulation.stark.run_one_step());

    const auto& query_radii = simulation.stark.context->logger->get_double_series(
        "ogc_query_radius");
    REQUIRE_FALSE(query_radii.empty());
    REQUIRE(*std::max_element(query_radii.begin(), query_radii.end()) <= 0.02 + 1e-12);

    const auto& ccd_fallbacks = simulation.stark.context->logger->get_int_series(
        "ogc_ccd_fallback");
    REQUIRE_FALSE(ccd_fallbacks.empty());
    REQUIRE(std::find(ccd_fallbacks.begin(), ccd_fallbacks.end(), 1)
        != ccd_fallbacks.end());

    for (int i = 0; i < moving.size(); ++i) {
        REQUIRE_THAT(
            moving.get_position(i).x(),
            WithinAbs(moving_vertices[i].x() + 0.3, 1e-6));
    }
}

TEST_CASE("OGC filters rigid translation and rotation", "[contact][ogc][integration]")
{
    stark::Settings settings;
    settings.output.simulation_name = "ogc_rigid_step_filter";
    settings.output.output_directory = std::string(STARK_TESTS_OUTPUT_DIR) + "/test_output";
    settings.output.enable_output = false;
    settings.output.enable_frame_writes = false;
    settings.output.console_verbosity = symx::Verbosity::Minimal;
    settings.simulation.gravity.setZero();
    settings.simulation.initial_time_step_size = 0.1;
    settings.simulation.max_time_step_size = 0.1;
    settings.simulation.use_adaptive_time_step = false;
    settings.newton.linear_solver = symx::LinearSolver::DirectLLT;
    settings.newton.max_iterations = 50;
    settings.newton.residual_tolerance_abs = 1e-8;

    stark::Simulation simulation(settings);
    auto floor = simulation.deformables->point_sets->add({
        {-1.0, -1.0, 0.0}, {1.0, -1.0, 0.0}, {0.0, 1.0, 0.0}});
    const std::vector<std::array<int, 3>> triangle{{0, 1, 2}};
    simulation.deformables->lumped_inertia->add(
        floor, std::vector<double>(3, 1.0),
        stark::EnergyLumpedInertia::Params().set_enable_gravity(false));
    simulation.deformables->prescribed_positions->add(
        floor, floor.all(),
        stark::EnergyPrescribedPositions::Params().set_stiffness(1e12));

    const std::vector<Eigen::Vector3d> local_triangle{
        {-0.5, -0.5, 0.0}, {0.5, -0.5, 0.0}, {0.0, 0.5, 0.0}};
    auto body = simulation.rigidbodies->add(
        1.0, stark::inertia_tensor_box(1.0, Eigen::Vector3d::Ones()));
    body.set_translation({0.0, 0.0, 0.03});
    body.set_acceleration({0.0, 0.0, -100.0});
    body.set_angular_acceleration({100.0, 0.0, 0.0});

    auto floor_contact = simulation.interactions->contact->add_triangles(
        floor, triangle, stark::ContactParams().set_contact_thickness(0.005));
    simulation.interactions->contact->add_triangles(
        body, local_triangle, triangle,
        stark::ContactParams().set_contact_thickness(0.005));
    floor_contact.disable_collision(floor_contact);
    simulation.interactions->contact->set_global_params(
        stark::ContactGlobalParams()
            .set_contact_method(stark::ContactMethod::OGC)
            .set_min_contact_stiffness(1e6));

    REQUIRE(simulation.stark.run_one_step());
    REQUIRE(simulation.stark.current_time_step == 1);
    REQUIRE(simulation.get_callbacks()->run_is_current_collision_state_valid());
}
#endif
