# Copyright (C) 2013 - 2022 Johan Hake (hake@simula.no), Jørgen S. Dokken
#
# SPDX-License-Identifier:    MIT
#
# Last changed: 2022-12-12

import dolfinx
import ufl
import typing
from .markerwisefield import rhs_with_markerwise_field, Markerwise
from petsc4py import PETSc
import numpy as np

__all__ = ["MonodomainSolver"]


class MonodomainSolver():
    """
    Solve the (pure) monodomain equations on the form:
    Find the transmembrane potential :math:`v = v(x, t)` such that

    .. math::

        \\frac{\\partial v}{\\partial t} - \\nabla \\cdot ( M_i \\nabla v) = I_s,

    where :math:`M_i` is the intracellular cardiac conductivity tensor;
    :math:`I_s` is prescribed input. In addition, initial conditions are
    given for :math:`v`:

    .. math::

        v(x, 0) = v_0

    This solver assumes pure homogeneous Neumann boundary conditions for :math:`v`.

    This solver is based on a :math:`\\theta`-scheme discretization in time
    and N-th order Lagrange space elements in space.

    Args:
        mesh: The spatial domain (mesh).
        M_i: The intracellular conductivity tensor (as an UFL expression).
        time: A constant holding the current time. If None is given, time is
            created for you, initialized to zero. This can be used as an internal
            variable in the RHS term :math:`I_s` to get a time-dependent input.
        I_s: A typically time-dependent external stimulus given as a dict,
            with domain markers as the key and a
            :py:class:`dolfinx.fem.Expression` as values. NB: it is assumed
            that the time dependence of I_s is encoded via the 'time'
            Constant.
        v0: Initial condition for v.
        params: Parameter dictionary specifying problem specific parameters.

            :code:`"polynomial_degree"`: The degree of the transmembral potential.

            :code:`"theta"`: Degree used in theta scheme.

            :code:`"jit_options"`: Dictionary with options for Just in time compilation.
            See `dolfinx/jit.py
            <https://github.com/FEniCS/dolfinx/blob/main/python/dolfinx/jit.py>`_
            for more information.

            :code:`"form_compiler_options"`: Dictionary with options for the FFCx form compiler.
            Call :code:`python3 -m ffcx --help` for all available options.

            :code:`"petsc_options"`: Dictionary containing options for the PETSc KSP solver.
            See the `PETSc documentation
            <https://petsc4py.readthedocs.io/en/stable/manual/ksp/>`_
            for more information.

            :code:`"use_custom_preconditioner"`: :code:`True`/:code:`False` Use
            :math:`\\int_\\Omega v\\cdot w + \\frac{\\Delta t}{2}
            (M_i \\nabla v)\\cdot \\nabla w~\\mathrm{d}x` as preconditioner

            :code:`dt`: Initial time step

    Examples:

        .. highlight:: python
        .. code-block:: python

            import dolfinx
            import ufl
            from mpi4py import MPI
            from petsc4py import PETSc
            from cbcbeatx import MonodomainSolver
            N = 10
            mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
            M_i = ufl.as_tensor(((0.2, 0), (0., 0.3)))
            time = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0.))
            x = ufl.SpatialCoordinate(mesh)
            I_s = x[0] * ufl.cos(x[1]) * time
            v_0 = ufl.sin(2*ufl.pi*x[0])
            petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
            params = {"polynomial_degree": 2, "theta": 0.5, "petsc_options": petsc_options,
            "use_custom_preconditioner": True}
            solver = MonodomainSolver(mesh, M_i, time, I_s, v_0, params)

    """
    _theta: dolfinx.fem.Constant  # Temporal discretization variable
    _V: dolfinx.fem.FunctionSpace  # Function space of solution
    _v: dolfinx.fem.Function  # Solution at previous time step
    _vh: dolfinx.fem.Function  # Solution at current time step

    _k_n: dolfinx.fem.Constant  # Delta t
    _t: dolfinx.fem.Constant  # Current time

    _solver: dolfinx.fem.petsc.LinearProblem  # Wrapper around PETSc KSP object

    _prec: dolfinx.fem.FormMetaClass
    _prec_matrix: PETSc.Mat
    _custom_prec: bool

    __slots__ = tuple(__annotations__)

    def __init__(self, mesh: dolfinx.mesh.Mesh, M_i: ufl.core.expr.Expr,
                 I_s: typing.Optional[tuple[ufl.core.expr.Expr, Markerwise]] = None,
                 v0: typing.Optional[ufl.core.expr.Expr] = None,
                 time: typing.Optional[dolfinx.fem.Constant] = None,
                 params: typing.Optional[dict] = None):

        # Get default parameters and overload with input parameters
        _params = self.default_parameters()
        if params is not None:
            _params.update(params)

        # Initialize class variables
        self._custom_prec = _params["use_custom_preconditioner"]
        self._theta = dolfinx.fem.Constant(mesh,  _params["theta"])
        self._V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange",  _params["polynomial_degree"]))

        # Set initial condition
        self._v = dolfinx.fem.Function(self._V)
        self._v.name = "v_prev"
        if v0 is not None:
            init_v = dolfinx.fem.Expression(v0, self._V.element.interpolation_points())
            self._v.interpolate(init_v)

        # Set initial simulation time
        self._t = time if time is not None else dolfinx.fem.Constant(mesh, PETSc.ScalarType(0.))

        # Extract integration measure and RHS of problem
        (dz, rhs) = rhs_with_markerwise_field(self._V, I_s)

        # Define variational form and KSP solver and pre-assemble
        self._init_linearproblem(
            _params["dt"], dz, M_i, rhs,  _params["form_compiler_options"], _params["jit_options"], _params["petsc_options"])

        # Initialize the preconditioner
        self._init_preconditioner(dz, M_i, _params["form_compiler_options"], _params["jit_options"])

    def _init_linearproblem(self, dt: float, dz: ufl.Measure, M_i: ufl.core.expr.Expr, rhs: ufl.form.Form,
                            form_compiler_options: dict, jit_options: dict, petsc_options: dict):
        """Initialize variational forms (`dolfinx.fem.Form`) for LHS and RHS.
        Preassemble the LHS.

        Args:
            dt (float): Initial time step
            dz (ufl.Measure): Mesh volume integration measure
            M_i (ufl.core.expr.Expr): The intracellular conductivity tensor (as an UFL expression).
            rhs (ufl.form.Form): The rhs forcing term (not multiplied by `dt`)
            form_compiler_options (dict): Options for form-compiler
            jit_options (dict): Options for JIT-compilation
            petsc_options (dict): Options for KSP-solver
        """
        mesh = self._V.mesh
        self._vh = dolfinx.fem.Function(self._V)
        self._vh.name = "vh"
        self._k_n = dolfinx.fem.Constant(mesh, dt)

        v = ufl.TrialFunction(self._V)
        w = ufl.TestFunction(self._V)
        Dt_v_k_n = (v - self._v)
        v_mid = self._theta * v + (1.-self._theta)*self._v
        theta_parabolic = ufl.inner(M_i * ufl.grad(v_mid), ufl.grad(w))*dz(domain=mesh)
        G = Dt_v_k_n*w*dz + self._k_n * theta_parabolic - self._k_n * rhs
        a, L = ufl.system(G)
        self._solver = dolfinx.fem.petsc.LinearProblem(
            a, L, u=self._vh,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            petsc_options=petsc_options)

        dolfinx.fem.petsc.assemble_matrix(self._solver.A, self._solver.a)  # type: ignore
        self._solver.A.assemble()

    def _init_preconditioner(self, M_i: ufl.core.expr.Expr, dz: ufl.Measure,
                             form_compiler_options: dict, jit_options: dict,):
        """
        Initialize custom preconditioner
        :math:`\\int_\\Omega v\\cdot w + \\frac{\\Delta t}{2}
        (M_i \\nabla v)\\cdot \\nabla w~\\mathrm{d}x`
        """
        if self._custom_prec:
            v = ufl.TrialFunction(self._V)
            w = ufl.TestFunction(self._V)
            self._prec = dolfinx.fem.form(
                (v*w + self._k_n/2.0*ufl.inner(M_i*ufl.grad(v), ufl.grad(w)))*dz,
                form_compiler_options=form_compiler_options, jit_options=jit_options)
            self._prec_matrix = dolfinx.fem.petsc.assemble_matrix(self._prec)
            self._solver.solver.setOperators(self._solver.A, self._prec_matrix)

    @staticmethod
    def default_parameters() -> dict:
        """Get the default parameters for the class

        Returns:
           dict: The default parameters
        """
        return {"polynomial_degree": 1,
                "dt": 0.1,
                "theta": 0.5,
                "use_custom_preconditioner": False,
                "jit_options": {},
                "form_compiler_options": {},
                "petsc_options": {"ksp_type": "preonly",
                                  "pc_type": "lu",
                                  "pc_factor_mat_solver_type": "mumps"}}

    def _update_matrices(self):
        """
        Re-assemble matrix.
        """
        self._solver.A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(self._solver.A, self._solver.a)  # type: ignore
        self._solver.A.assemble()

        if self._custom_prec:
            self._prec_matrix.zeroEntries()
            dolfinx.fem.petsc.assemble_matrix(self._prec_matrix, self._prec)
            self._prec_matrix.assemble()

    def _update_rhs(self):
        """
        Re-assemble RHS vector
        """
        with self._solver.b.localForm() as b_loc:
            b_loc.set(0)
        dolfinx.fem.petsc.assemble_vector(self._solver.b, self._solver.L)
        self._solver.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    def step(self, interval: tuple[float, float]):
        """
        Solve the problem on a time given interval :math:`[T_0, T_1]`

        Args:
            interval(tuple[float, float]): The time interval :math:`[T_0, T_1]`
            dt(float, optional): The time step :math:`\\Delta t`.
                Defaults to :code:`None`, which corresponds to :math:`\\Delta t = T_1-T_0`.

        """
        (t0, t1) = interval
        dt = t1 - t0
        if not np.isclose(self._k_n.value, dt):
            self._k_n.value = dt
            self._update_matrices()
        # Assemble RHS vector
        # Update t before updating RHS to capture possible time dependency
        self._t.value = t0 + self._theta.value * (t1 - t0)
        self._update_rhs()
        # Solve linear system and update ghost values in the solution
        self._solver.solver.solve(self._solver.b, self._vh.vector)
        self._vh.x.scatter_forward()

    def solve(self, interval: tuple[float, float],
              dt: typing.Optional[float] = None) -> typing.Generator[
        typing.Tuple[typing.Tuple[float, float],
                     typing.Tuple[dolfinx.fem.Function, dolfinx.fem.Function]], None, None]:
        """
        Solve the problem on a time given interval :math:`[T_0, T_1]`

        Args:
            interval(tuple[float, float]): The time interval :math:`[T_0, T_1]`
            dt(float, optional): The time step :math:`\\Delta t`.
                Defaults to :code:`None`, which corresponds to :math:`\\Delta t = T_1-T_0`.
        Yields:
            step_output: An iterator solving for each time step. Each element of the iterator is
                a tuple describing the time step, and a tuple of the previous and current solution.
        """
        (T0, T) = interval
        if dt is None:
            num_steps = int(1)
            dt = T-T0
        else:
            num_steps = int((T - T0) // dt)
        t0 = T0
        t1 = T0 + dt
        # Step through time steps
        for i in range(num_steps):
            self.step((t0, t1))
            yield (t0, t1), (self._v, self._vh)
            self._v.x.array[:] = self._vh.x.array[:]
            t0 = t1
            t1 = t0 + dt

    def solution_fields(self) -> tuple[dolfinx.fem.Function, dolfinx.fem.Function]:
        """
        Return a tuple of Functions :math:`(u_{n-1}, u_n)` where :math:`u_{n-1}` is the solution
        from the previous time step. :math:`u_n` the solution at the current time step

        Returns:
            tuple[dolfinx.fem.Function, dolfinx.fem.Function]: The functions
        """
        return self._v, self._vh

    @property
    def time(self) -> np.float64:
        """Return current time used in solver"""
        return self._t.value
