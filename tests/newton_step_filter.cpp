#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <symx>
#include <memory>

using namespace Catch::Matchers;

namespace {
std::pair<symx::spNewtonsMethod, std::shared_ptr<std::vector<Eigen::Vector3d>>> make_quadratic_solver(
    symx::spSolverCallbacks callbacks)
{
    auto potential = symx::GlobalPotential::create();
    auto dofs = std::make_shared<std::vector<Eigen::Vector3d>>(
        1, Eigen::Vector3d::Zero());
    potential->add_dof(*dofs, "u");

    static const std::vector<std::array<int32_t, 1>> connectivity{{0}};
    static const Eigen::Vector3d target(2.0, 0.0, 0.0);
    potential->add_potential(
        "quadratic", connectivity,
        [dofs](symx::MappedWorkspace<double>& mws, symx::Element& element) {
            symx::Vector u = mws.make_vector(*dofs, element[0]);
            const symx::Vector target_symbol = mws.make_vector(target);
            return 0.5 * (u - target_symbol).squared_norm();
        });

    auto solver = symx::NewtonsMethod::create(potential, symx::Context::create(), callbacks);
    solver->settings.linear_solver = symx::LinearSolver::DirectLLT;
    solver->settings.projection_mode = symx::ProjectionToPD::Newton;
    solver->settings.max_iterations = 1;
    solver->settings.max_iterations_as_success = true;
    solver->settings.enable_armijo_backtracking = true;
    return {solver, dofs};
}
}

TEST_CASE("step filters mutate the direction and return a conservative fallback", "[newton][step_filter]")
{
    auto callbacks = symx::SolverCallbacks::create(symx::Context::create());
    callbacks->add_step_filter(
        [](const Eigen::VectorXd& dofs, Eigen::VectorXd& step) {
            REQUIRE(dofs.isApprox(Eigen::Vector3d(1.0, 2.0, 3.0)));
            step[0] *= 0.5;
            step[1] *= 0.25;
            return 0.25;
        });
    callbacks->add_step_filter(
        [](const Eigen::VectorXd&, Eigen::VectorXd& step) {
            step[2] *= 0.75;
            return 0.75;
        });

    const Eigen::Vector3d dofs(1.0, 2.0, 3.0);
    Eigen::VectorXd step = Eigen::Vector3d(4.0, 8.0, 12.0);
    const double safe_scale = callbacks->run_step_filters(dofs, step);

    REQUIRE(step.isApprox(Eigen::Vector3d(2.0, 2.0, 9.0)));
    REQUIRE_THAT(safe_scale, WithinAbs(0.25, 1e-12));
}

TEST_CASE("Newton applies a filtered direction before Armijo", "[newton][step_filter]")
{
    auto context = symx::Context::create();
    auto callbacks = symx::SolverCallbacks::create(context);
    callbacks->add_step_filter(
        [](const Eigen::VectorXd&, Eigen::VectorXd& step) {
            step *= 0.5;
            return 0.5;
        });
    auto [solver, dofs] = make_quadratic_solver(callbacks);

    REQUIRE(solver->solve() == symx::SolverReturn::Successful);
    REQUIRE_THAT((*dofs)[0].x(), WithinAbs(1.0, 1e-9));
}

TEST_CASE("Newton preserves raw step convergence when no filters are registered", "[newton][step_filter]")
{
    auto context = symx::Context::create();
    auto callbacks = symx::SolverCallbacks::create(context);
    auto [solver, dofs] = make_quadratic_solver(callbacks);
    solver->settings.step_cap = 0.25;
    solver->settings.step_tolerance = 0.5;

    REQUIRE(solver->solve() == symx::SolverReturn::Successful);
    REQUIRE_THAT((*dofs)[0].x(), WithinAbs(0.25, 1e-9));
}

TEST_CASE("Newton caps the direction before step filtering", "[newton][step_filter]")
{
    auto context = symx::Context::create();
    auto callbacks = symx::SolverCallbacks::create(context);
    callbacks->add_step_filter(
        [](const Eigen::VectorXd&, Eigen::VectorXd& step) {
            REQUIRE_THAT(step[0], WithinAbs(0.25, 1e-9));
            return 1.0;
        });
    auto [solver, dofs] = make_quadratic_solver(callbacks);
    solver->settings.step_cap = 0.25;

    REQUIRE(solver->solve() == symx::SolverReturn::Successful);
    REQUIRE_THAT((*dofs)[0].x(), WithinAbs(0.25, 1e-9));
}

TEST_CASE("Newton falls back to a safe scalar when filtering loses descent", "[newton][step_filter]")
{
    auto context = symx::Context::create();
    auto callbacks = symx::SolverCallbacks::create(context);
    callbacks->add_step_filter(
        [](const Eigen::VectorXd&, Eigen::VectorXd& step) {
            step *= -1.0;
            return 0.25;
        });
    auto [solver, dofs] = make_quadratic_solver(callbacks);

    REQUIRE(solver->solve() == symx::SolverReturn::Successful);
    REQUIRE_THAT((*dofs)[0].x(), WithinAbs(0.5, 1e-9));
}

TEST_CASE("Newton accepts a filtered step below tolerance before descent fallback", "[newton][step_filter]")
{
    auto context = symx::Context::create();
    auto callbacks = symx::SolverCallbacks::create(context);
    callbacks->add_step_filter(
        [](const Eigen::VectorXd&, Eigen::VectorXd& step) {
            step.setZero();
            return 0.25;
        });
    auto [solver, dofs] = make_quadratic_solver(callbacks);
    solver->settings.step_tolerance = 1e-3;

    REQUIRE(solver->solve() == symx::SolverReturn::Successful);
    REQUIRE_THAT((*dofs)[0].x(), WithinAbs(0.0, 1e-12));
}

TEST_CASE("Newton accepts a safe fallback below step tolerance", "[newton][step_filter]")
{
    auto context = symx::Context::create();
    auto callbacks = symx::SolverCallbacks::create(context);
    callbacks->add_step_filter(
        [](const Eigen::VectorXd&, Eigen::VectorXd& step) {
            step *= -1.0;
            return 1e-6;
        });
    auto [solver, dofs] = make_quadratic_solver(callbacks);
    solver->settings.step_tolerance = 1e-3;

    REQUIRE(solver->solve() == symx::SolverReturn::Successful);
    REQUIRE_THAT((*dofs)[0].x(), WithinAbs(0.0, 1e-12));
}
