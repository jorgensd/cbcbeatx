import dolfinx
from mpi4py import MPI
from petsc4py import PETSc
import pytest
from cbcbeatx import MonodomainSolver, BasicMonodomainSolver
import numpy as np


class TestBasicMonodomainSolver():
    "Test functionality for the basic monodomain solver."

    def setUp(self):
        self.mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 5, 5, 5)
        self.time = 0

        # Create stimulus
        c = dolfinx.fem.Constant(self.mesh, 2.0)
        self.stimulus = c

        # Create conductivity "tensors"
        self.M_i = 1.0

        self.t0 = 0.0
        self.dt = 0.1

    @pytest.mark.fast
    def test_basic_solve(self):
        "Test that solver runs."
        self.setUp()

        # Create solver
        solver = BasicMonodomainSolver(self.mesh, self.time,
                                       self.M_i, I_s=self.stimulus)

        # Solve
        solutions = solver.solve((self.t0, self.t0 + 2*self.dt), self.dt)
        for (interval, fields) in solutions:
            (v_, vs) = fields

    @pytest.mark.fast
    def test_compare_solve_step(self):
        "Test that solve gives same results as single step"
        self.setUp()

        solver = BasicMonodomainSolver(self.mesh, self.time,
                                       self.M_i, I_s=self.stimulus)

        (v_, vs) = solver.solution_fields()

        # Solve
        interval = (self.t0, self.t0 + self.dt)
        solutions = solver.solve(interval, self.dt)
        for (interval, fields) in solutions:
            (v_, vur) = fields
            vur.vector.normBegin(PETSc.NormType.NORM_2)
            a = vur.vector.normEnd(PETSc.NormType.NORM_2)
        # Reset v_
        v_.x.set(0.0)

        # Step
        solver.step(interval)
        vs.vector.normBegin(PETSc.NormType.NORM_2)
        b = vs.vector.normEnd(PETSc.NormType.NORM_2)
        # Check that result from solve and step match.
        assert np.isclose(a, b)


class TestMonodomainSolver():
    def setUp(self):
        N = 5
        self.mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, N, N, N)
        self.time = 0.0

        # Create stimulus
        c = dolfinx.fem.Constant(self.mesh, 2.0)
        self.stimulus = c

        # Create conductivity "tensors"
        self.M_i = 1.0

        self.t0 = 0.0
        self.dt = 0.1

    @pytest.mark.fast
    def test_solve(self):
        "Test that solver runs."
        self.setUp()

        # Create solver and solve
        solver = MonodomainSolver(self.mesh,
                                  self.M_i, time=self.time, I_s=self.stimulus)
        solutions = solver.solve((self.t0, self.t0 + 2*self.dt), self.dt)
        for (interval, fields) in solutions:
            (v_, vur) = fields

    @pytest.mark.fast
    def test_compare_with_basic_solve(self):
        """Test that solver with direct linear algebra gives same
        results as basic monodomain solver."""
        self.setUp()

        # Create solver and solve
        params_direct = {"petsc_options": {"ksp_type": "preonly", "pc_type": "lu"}}
        solver = MonodomainSolver(self.mesh,
                                  self.M_i, time=self.time, I_s=self.stimulus,
                                  params=params_direct)
        solutions = solver.solve((self.t0, self.t0 + 2*self.dt), self.dt)
        results_mono = []
        for (interval, fields) in solutions:
            (v_, vur) = fields
            vur.vector.normBegin(PETSc.NormType.NORM_2)
            monodomain_result = vur.vector.normEnd(PETSc.NormType.NORM_2)
            results_mono.append(monodomain_result)
        # Create other solver and solve
        solver = BasicMonodomainSolver(self.mesh, self.time,
                                       self.M_i, I_s=self.stimulus)
        solutions = solver.solve((self.t0, self.t0 + 2*self.dt), self.dt)
        results_basic = []
        for (interval, fields) in solutions:
            (v_, vur) = fields
            vur.vector.normBegin(PETSc.NormType.NORM_2)
            basic_monodomain_result = vur.vector.normEnd(PETSc.NormType.NORM_2)
            results_basic.append(basic_monodomain_result)
        assert np.allclose(np.array(results_basic), np.array(results_mono),
                           atol=1e-13)

    @pytest.mark.fast
    def test_compare_direct_iterative(self):
        "Test that direct and iterative solution give comparable results."
        self.setUp()

        # Create solver and solve
        params_direct = {"petsc_options": {"ksp_type": "preonly", "pc_type": "lu"}}
        solver = MonodomainSolver(self.mesh, self.M_i, time=self.time, I_s=self.stimulus,
                                  params=params_direct)
        solutions = solver.solve((self.t0, self.t0 + 3*self.dt), self.dt)
        for (interval, fields) in solutions:
            (v_, v) = fields
            v.vector.normBegin(PETSc.NormType.NORM_2)
            l2_norm = v.vector.normEnd(PETSc.NormType.NORM_2)

        # Create solver and solve using iterative means
        params_iter = {"petsc_options": {"ksp_type": "gmres", "pc_type": "ilu", "ksp_view": None}}
        solver = MonodomainSolver(self.mesh, self.M_i,
                                  time=self.time, I_s=self.stimulus,
                                  params=params_iter)
        solutions = solver.solve((self.t0, self.t0 + 3*self.dt), self.dt)
        for (interval, fields) in solutions:
            (v_, v) = fields
            v.vector.normBegin(PETSc.NormType.NORM_2)
            krylov_norm = v.vector.normEnd(PETSc.NormType.NORM_2)
            print(v_.x.array, id(v_))

        np.isclose(l2_norm, krylov_norm, atol=1e-4)
