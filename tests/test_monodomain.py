import dolfinx
import numpy as np
import pytest
import ufl
from cbcbeatx import MonodomainSolver
from mpi4py import MPI
from petsc4py import PETSc


class TestMonodomainSolver:
    def setUp(self):
        N = 5
        self.mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, N, N, N)

        # Create stimulus
        c = dolfinx.fem.Constant(self.mesh, dolfinx.default_scalar_type(2.0))
        self.stimulus = c

        # Create conductivity "tensors"
        self.M_i = 1.0

        self.t0 = 0.0
        self.dt = 0.01

    @pytest.mark.fast
    def test_solve(self):
        "Test that solver runs."
        self.setUp()

        # Create solver and solve
        solver = MonodomainSolver(self.mesh, self.M_i, I_s=self.stimulus)
        solutions = solver.solve((self.t0, self.t0 + 2 * self.dt), self.dt)
        for (interval, fields) in solutions:
            (v_, vur) = fields

    @pytest.mark.fast
    def test_compare_with_basic_solve(self):
        """Test that solver with direct linear algebra gives same
        results as a straightforward implementation solver."""
        self.setUp()

        # Create solver and solve
        params_direct = {
            "petsc_options": {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        }
        solver = MonodomainSolver(
            self.mesh,
            self.M_i,
            I_s=self.stimulus,
            params=params_direct,
        )
        solutions = solver.solve((self.t0, self.t0 + 2 * self.dt), self.dt)

        # Create basic heat eq solver
        V = solver._V
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        u0 = dolfinx.fem.Function(V)
        dt = dolfinx.fem.Constant(self.mesh, dolfinx.default_scalar_type(self.dt))
        F = (
            ufl.inner(u - u0, v) * ufl.dx
            - dt * ufl.inner(self.M_i * ufl.grad(u), ufl.grad(v)) * ufl.dx
        )
        F -= dt * ufl.inner(self.stimulus, v) * ufl.dx
        a, L = ufl.system(F)
        solver = dolfinx.fem.petsc.LinearProblem(
            a,
            L,
            petsc_options=params_direct["petsc_options"],
        )

        for (interval, fields) in solutions:
            (_, vur) = fields
            uh = solver.solve()
            assert np.allclose(vur.x.array, uh.x.array)
            u0.x.array[:] = uh.x.array[:]

    @pytest.mark.skipif(
        MPI.COMM_WORLD.size > 1,
        reason="This test should only be run in serial.",
    )
    @pytest.mark.fast
    def test_compare_direct_iterative(self):
        "Test that direct and iterative solution give comparable results."
        self.setUp()

        # Create solver and solve
        params_direct = {"petsc_options": {"ksp_type": "preonly", "pc_type": "lu"}}
        solver = MonodomainSolver(
            self.mesh,
            self.M_i,
            I_s=self.stimulus,
            params=params_direct,
        )
        solutions = solver.solve((self.t0, self.t0 + 3 * self.dt), self.dt)
        for (interval, fields) in solutions:
            (v_, v) = fields
            v.vector.normBegin(PETSc.NormType.NORM_2)
            l2_norm = v.vector.normEnd(PETSc.NormType.NORM_2)

        # Create solver and solve using iterative means
        params_iter = {
            "petsc_options": {"ksp_type": "gmres", "pc_type": "ilu", "ksp_view": None},
        }
        solver = MonodomainSolver(
            self.mesh,
            self.M_i,
            I_s=self.stimulus,
            params=params_iter,
        )
        solutions = solver.solve((self.t0, self.t0 + 3 * self.dt), self.dt)
        for (interval, fields) in solutions:
            (v_, v) = fields
            v.vector.normBegin(PETSc.NormType.NORM_2)
            krylov_norm = v.vector.normEnd(PETSc.NormType.NORM_2)

        np.isclose(l2_norm, krylov_norm, atol=1e-4)


@pytest.mark.parametrize("theta", [0.5, 1.0])
@pytest.mark.parametrize("degree", [1, 2])
def test_manufactured_solution(theta: float, degree: int):
    """
    Test Monodomain solver against a manufactured solution

    Args:
        theta (float): Time discretization parameter
        degree (int): Degree of membrane potential function space
    """
    num_refs = 3
    dt = 1e-3
    N0 = 4
    t0 = 0.5
    t1 = 0.8

    eoxt = np.zeros(num_refs + 1, dtype=np.float64)
    hs = np.zeros(num_refs + 1, dtype=np.float64)
    metadata = {"quadrature_degree": 8}
    options = {
        "petsc_options": {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        "theta": theta,
        "polynomial_degree": degree,
    }
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N0, N0)
    for i in range(num_refs + 1):
        error_time = []
        if i > 0:
            mesh.topology.create_entities(1)
            mesh = dolfinx.mesh.refine(mesh)

        cmap = mesh.topology.index_map(mesh.topology.dim)
        cells_local = np.arange(cmap.size_local + cmap.num_ghosts, dtype=np.int32)
        hs[i] = np.max(mesh.h(mesh.topology.dim, cells_local))

        x = ufl.SpatialCoordinate(mesh)
        t = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(t0))
        t_var = ufl.variable(t)
        u = ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1]) * ufl.cos(t_var)
        du_dt = ufl.diff(u, t_var)
        M_i = 0.3 * ufl.as_tensor(((1, 0), (0, 1)))
        ict = du_dt - ufl.div(M_i * ufl.grad(u))
        solver = MonodomainSolver(mesh, M_i, v0=u, time=t, I_s=ict, params=options)

        # Create new expression to use constant that is not updated internally in solver
        t_eval = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
        u_exact = ufl.replace(u, {t_var: t_eval})
        diff = solver._vh - u_exact
        error = dolfinx.fem.form(
            ufl.inner(diff, diff) * ufl.dx(domain=mesh, metadata=metadata),
        )

        solutions = solver.solve((t0, t1), dt)
        for (interval, _) in solutions:
            _, ti = interval

            t_eval.value = ti
            error_time.append(
                mesh.comm.allreduce(dolfinx.fem.assemble_scalar(error), op=MPI.SUM),
            )
        eoxt[i] = np.sqrt(np.sum(error_time) * dt)

    rates = np.log(eoxt[1:] / eoxt[:-1]) / np.log(hs[1:] / hs[:-1])
    assert np.isclose(rates[-1], degree + 1, atol=0.05)
    print(f"Convergence rates {rates}")
