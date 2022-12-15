# Copyright (C) 2016 - 2022 Marie E. Rognes (meg@simula.no), Jørgen S. Dokken
#
# SPDX-License-Identifier:    MIT
#
# Last changed: 2022-12-12
import typing

import dolfinx
import ufl

__all__ = ["rhs_with_markerwise_field", "Markerwise"]


class Markerwise:
    """
    A container class representing an object defined by a number of
    objects combined with a mesh tag defining mesh domains and
    a map from each object to the subdomain.

    Args:
        objects (list[ufl.core.expr.E[xr]): The forcing terms as ufl expressions
        keys (list[int]): The cell-tag integer associated with each forcing term
        marker (dolfinx.mesh.MeshTagsMetaClass): The cell tags

    Examples:
        Given `(g0, g1)`, `(2, 5)` and `cell_markers`, let

        .. math::

            g =\\begin{cases}
              g0 \\text{ on domains marked by 2 in } cell\\_markers \\\\
              g1 \\text{ on domains marked by 5 in } cell\\_markers
            \\end{cases}


        .. highlight:: python
        .. code-block:: python

            g = Markerwise((g0, g1), (2, 5), markers)
    """

    def __init__(
        self,
        objects: list[ufl.core.expr.Expr],
        keys: list[int],
        marker: dolfinx.mesh.MeshTagsMetaClass,
    ):
        assert len(objects) == len(keys)
        assert marker.dim == marker.mesh.topology.dim
        self._marker = marker
        self._objects = dict(zip(keys, objects))

    def items(self):
        return self._objects.items()

    @property
    def marker(self):
        "The cell marker"
        return self._marker

    def __getitem__(self, key: int) -> ufl.core.expr.Expr:
        """Get the ufl form for a given subdomain

        Args:
            key (int): The tag of the subdomain

        Returns:
            typing.Any: The corresponding ufl expression
        """
        return self._objects[key]


def rhs_with_markerwise_field(
    V: dolfinx.fem.FunctionSpace,
    g: typing.Optional[typing.Union[ufl.core.expr.Expr, Markerwise]],
) -> tuple[ufl.Measure, ufl.Form]:
    """
    Create the ufl-form :math:`G=\\int_\\Omega g v~\\mathrm{d}x` where `g` can be:

    1. A set of ufl expressions over subdomains (`Markerwise`)
    2. A single ufl expression over the whole domain (`ufl.core.expr.Expr`)
    3. If no expression is supplied, return a zero integral (will be optimized away later)

    Args:
        V: The function space to extract the test function `v` from
        g: The forcing term

    Returns:
       A tuple of the integration measure `dx` over the domain (without subdomain id set) and
       the source form `G`
    """
    v = ufl.TestFunction(V)
    if g is None:
        dz = ufl.dx
        rhs = 0.0
    try:
        marker = g.marker  # type: ignore
        dz = ufl.Measure("dx", domain=marker.mesh, subdomain_data=marker)
        rhs = sum([gi * v * dz(i) for (i, gi) in g.items()])  # type: ignore
    except AttributeError:
        dz = ufl.dx
        rhs = g * v * dz
    return (dz, rhs)
