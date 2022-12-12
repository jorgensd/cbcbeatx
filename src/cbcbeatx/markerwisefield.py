

# Copyright (C) 2016 - 2022 Marie E. Rognes (meg@simula.no), Jørgen S. Dokken
#
# SPDX-License-Identifier:    MIT
#
# Last changed: 2022-12-12


import dolfinx
import typing
import ufl
__all__ = ["rhs_with_markerwise_field", "Markerwise"]


class Markerwise():
    """
    A container class representing an object defined by a number of objects combined with a mesh tag defining mesh domains and 
    a map from each object to the subdomain.

    Args:
        objects (list[any]): _description_
        keys (list[int]): _description_
        marker (dolfinx.mesh.MeshTags): _description_

    Examples:
        Given (g0, g1), (2, 5) and `cell_markers`, let

        .. math::

            g = g0 on domains marked by 2 in markers
            g = g1 on domains marked by 5 in markers

        .. highlight:: python
        .. code-block:: python

            g = Markerwise((g0, g1), (2, 5), markers)
    """

    def __init__(self, objects: list[any], keys: list[int], marker: dolfinx.mesh.MeshTags):
        assert (len(object) == len(keys))
        self._marker = marker
        self._objects = dict(zip(keys, objects))

    def items(self):
        return self._objects.items()

    @property
    def marker(self):
        "The marker"
        return self._marker

    def __getitem__(self, key: int) -> any:
        return self._objects[key]


def rhs_with_markerwise_field(v: ufl.core.expr.Expr,
                              g: typing.Optional[typing.Union[ufl.core.expr.Expr, Markerwise]]) -> tuple[ufl.Measure, ufl.Form]:
    """
    Create a cell integral for either:
    1. A set of ufl expressions over subdomains 
    2. A single ufl expression over the whole domain
    3. If no expression is supplied, return a zero integral (will be optimized away later)
    """
    if g is None:
        dz = ufl.dx
        rhs = 0.0*dz
    try:
        marker = g.marker
        dz = ufl.Measure("dx", domain=marker.mesh, subdomain_data=marker)
        rhs = sum([gi*v*dz(i) for (i, gi) in g.items()])
    except AttributeError:
        dz = ufl.dx
        rhs = g*v*dz
    return (dz, rhs)
