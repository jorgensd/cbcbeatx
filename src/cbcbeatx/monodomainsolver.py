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

__all__ = ["BasicMonodomainSolver", "MonodomainSolver"]


class BasicMonodomainSolver(object):
    """
    Solve the (pure) monodomain equations on the form: find
    the transmembrane potential :math:`v = v(x, t)` such that

    .. math::

        v_t - \\mathrm{div} ( G_i v) = I_s

    where the subscript :math:`t` denotes the time derivative; :math:`G_i`
    denotes a weighted gradient: :math:`G_i = M_i \\mathrm{grad}(v)` for,
    where :math:`M_i` is the intracellular cardiac conductivity tensor;
    :math:`I_s` ise prescribed input. In addition, initial conditions are
    given for :math:`v`:

    .. math::

        v(x, 0) = v_0

    Finally, boundary conditions must be prescribed. For now, this solver
    assumes pure homogeneous Neumann boundary conditions for :math:`v`.

    This solver is based on a theta-scheme discretization in time
    and CG_1 elements in space.

    .. note::

       For the sake of simplicity and consistency with other solver
       objects, this solver operates on its solution fields (as state
       variables) directly internally. More precisely, solve (and
       step) calls will act by updating the internal solution
       fields. It implies that initial conditions can be set (and are
       intended to be set) by modifying the solution fields prior to
       simulation.

    *Arguments*
      mesh
        The spatial domain (mesh)

      M_i
        The intracellular conductivity tensor (as an UFL expression)

      time
        A constant holding the current time. If None is given, time is
        created for you, initialized to zero.

      I_s
        A typically time-dependent external stimulus given as a dict,
        with domain markers as the key and a
        :py:class:`dolfinx.fem.Expression` as values. NB: it is assumed
        that the time dependence of I_s is encoded via the 'time'
        Constant.

      v0
        Initial condition for v.

      params
        Parameters for problem.f"polynomial_degree": The degree of the transmembral potential
        "theta": Degree used in theta scheme

      """

    _theta: dolfinx.fem.Constant  # Temporal discretization variable
    _V: dolfinx.fem.FunctionSpace  # Function space of solution
    _v: dolfinx.fem.Function  # Solution at previous time step
    _vh: dolfinx.fem.Function  # Solution at current time step

    _k_n: dolfinx.fem.Constant  # Delta t
    _t: float  # Current time

    _solver: dolfinx.fem.petsc.LinearProblem

    # Annotate all functions
    __slots__ = tuple(__annotations__)

    def __init__(self,
                 mesh: dolfinx.mesh.Mesh,
                 M_i: ufl.core.expr.Expr,
                 time: typing.Optional[float] = None,
                 I_s: typing.Optional[tuple[ufl.core.expr.Expr, Markerwise]] = None,
                 v0: typing.Optional[dolfinx.fem.Function] = None,
                 params: typing.Optional[dict] = None):

        # Get default parameters or input parameters
        params = {} if params is None else params
        degree = params.pop("polynomial_degree", 1)
        theta = params.pop("theta", 0.5)
        jit_options = params.pop("jit_options", {})
        form_compiler_options = params.pop("form_compiler_options", {})

        self._theta = dolfinx.fem.Constant(mesh, theta)

        self._V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", degree))

        # Get initial condition
        self._v = dolfinx.fem.Function(self._V)
        self._v.name = "v_"
        # Interpolate init condition
        if v0 is not None:
            init_v = dolfinx.fem.Expression(v0, self._V.element.interpolation_points())
            self._v.interpolate(init_v)

        # Set initial simulation time
        self._t = time if time is not None else 0

        self._vh = dolfinx.fem.Function(self._V)

        # Define variational form
        self._k_n = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0))
        v = ufl.TrialFunction(self._V)
        w = ufl.TestFunction(self._V)
        Dt_v_k_n = (v - self._vh)
        v_mid = self._theta * v + (1.-self._theta)*self._vh
        (dz, rhs) = rhs_with_markerwise_field(w, I_s)
        theta_parabolic = ufl.inner(M_i * ufl.grad(v_mid), ufl.grad(w))*dz(domain=mesh)
        G = Dt_v_k_n*w*dz + self._k_n * theta_parabolic - self._k_n * rhs
        a, L = ufl.system(G)
        self._solver = dolfinx.fem.petsc.LinearProblem(
            a, L, u=self._vh,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            petsc_options=params.pop("petsc_options", {}))

    def step(self, interval: tuple[float, float]):
        """
        Solve on the given time interval (t0, t1).
        It is assumed that `v_` is in the correct for `t0`. This function updates
        `v` to the correct state at `t1`.

        Args:
            interval (tuple[float, float]): The time interval
        """
        (t0, t1) = interval
        self._k_n.value = t1 - t0
        self._t = t0 + self._theta.value * (t1 - t0)
        self._solver.solve()

    def solution_fields(self) -> tuple[dolfinx.fem.Function, dolfinx.fem.Function]:
        return self._v, self._vh

    def solve(self, interval: tuple[float, float],
              dt: typing.Optional[float] = None) -> typing.Generator[
        typing.Tuple[typing.Tuple[float, float],
                     typing.Tuple[dolfinx.fem.Function, dolfinx.fem.Function]], None, None]:
        """
        Solve the discretization on a time given interval

        Args:
            interval (tuple[float, float]): _description_
            dt (float, optional): _description_. Defaults to None.
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
            # NOTE: This should probably be a named tuple
            yield (t0, t1), (self._v, self._vh)
            self._v.x.array[:] = self._vh.x.array[:]
            t0 = t1
            t1 = t0 + dt

    @property
    def time(self):
        return self._t.value


class MonodomainSolver(BasicMonodomainSolver):
    __doc__ = BasicMonodomainSolver.__doc__

    _prec: dolfinx.fem.FormMetaClass
    _prec_matrix: PETSc.Mat
    _custom_prec: bool

    __slots__ = tuple(__annotations__) + BasicMonodomainSolver.__slots__

    def __init__(self, mesh: dolfinx.mesh.Mesh, M_i: ufl.core.expr.Expr,
                 I_s: typing.Optional[tuple[ufl.core.expr.Expr, Markerwise]] = None,
                 v_: typing.Optional[ufl.core.expr.Expr] = None,
                 time: typing.Optional[float] = 0, params: typing.Optional[dict] = None):
        if params is None:
            params = {}
        self._custom_prec = params.pop("use_custom_preconditioner", False)

        super().__init__(mesh, M_i, time, I_s, v_, params)

        # Get default parameters or input parameters
        params = {} if params is None else params

        # Preassemble LHS
        dolfinx.fem.petsc.assemble_matrix(self._solver.A, self._solver.a)  # type: ignore
        if self._custom_prec:
            v = ufl.TrialFunction(self._V)
            w = ufl.TestFunction(self._V)
            (dz, _) = rhs_with_markerwise_field(I_s, w)
            self._prec = dolfinx.fem.form(
                (v*w + self._k_n/2.0*ufl.inner(M_i*ufl.grad(v), ufl.grad(w)))*dz)

            self._prec_matrix = dolfinx.fem.petsc.assemble_matrix(self._prec)
            self._solver.solver.setOperators(self._solver.A, self._prec_matrix)

    def _update_matrices(self):
        self._solver.A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(self._solver.A, self._solver.a)  # type: ignore
        self._solver.A.assemble()

        if self._custom_prec:
            self._prec_matrix.zeroEntries()
            dolfinx.fem.petsc.assemble_matrix(self._prec_matrix, self._prec)
            self._prec_matrix.assemble()

    def _update_rhs(self):
        # Assemble rhs
        with self._solver.b.localForm() as b_loc:
            b_loc.set(0)
        dolfinx.fem.petsc.assemble_vector(self._solver.b, self._solver.L)
        self._solver.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    def step(self, interval: tuple[float, float]):
        """
        Solve on the given time interval (t0, t1).
        It is assumed that `v_` is in the correct for `t0`. This function updates
        `v` to the correct state at `t1`.

        Args:
            interval (tuple[float, float]): The time interval
        """
        (t0, t1) = interval
        dt = t1 - t0
        if not np.isclose(self._k_n.value, dt):
            self._k_n.value = dt
            self._update_matrices()
        # Assemble RHS vector
        self._update_rhs()
        # Solve linear system and update ghost values in the solution
        self._solver.solver.solve(self._solver.b, self._vh.vector)
        self._vh.x.scatter_forward()
        self._t += dt

    def solve(self, interval: tuple[float, float],
              dt: typing.Optional[float] = None) -> typing.Generator[
        typing.Tuple[typing.Tuple[float, float],
                     typing.Tuple[dolfinx.fem.Function, dolfinx.fem.Function]], None, None]:
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
