"""Extract the key outline coordinates of a shapely shape."""

import numpy as np
import shapely.geometry
import pygeos


def extract_polygon_outline(shapely_geometry: shapely.geometry) -> np.ndarray:
    """
    Extract the key outline coordinates of a shapely shape.

    Extract the key outline coordinates of a shapely shape
    (including multi-shapes like MultiPolygon). The following
    types ares supported:
        * Polygon:              The outline of the polygon is returned
        * MultiPolygon:         The outline of the largest polygon is returned
        * GeometryCollection:   The largest polygon in the set is returned,
                                if no polygon is present 'None' is returned
        * LineString:           'None' is returned, since the shape is
                                a line and does not host volume information

    For any other type, an error is raised.

    :param shapely_geometry:    shapely-geometry of interest
    :returns:
        * **polygon_outline** - outline coordinates in form of
                                a numpy array with columns x, y

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        09.10.2020

    """
    polygon_outline = None

    if shapely_geometry.geom_type == 'Polygon':
        if not shapely_geometry.is_empty:
            polygon_outline = shapely_geometry.exterior.coords.xy

    elif shapely_geometry.geom_type == 'MultiPolygon':
        # extract largest polygon
        polygon_outline = max(shapely_geometry, key=lambda a: a.area).\
            exterior.coords.xy

    elif shapely_geometry.geom_type == 'GeometryCollection':
        # extract polygons
        polygons = []
        for geometry in shapely_geometry.geoms:
            if (
                geometry.geom_type == 'Polygon' and
                not shapely_geometry.is_empty
            ):
                polygons.append(geometry)

        # extract largest polygon
        if polygons:
            polygon_outline = max(polygons, key=lambda a: a.area).\
                exterior.coords.xy

    elif shapely_geometry.geom_type == 'LineString':
        # if just line left, skip
        pass

    else:
        raise ValueError("Faced unsupported shape '" +
                         str(shapely_geometry.geom_type) + "'!")

    # convert to numpy array
    if polygon_outline is not None:
        polygon_outline = np.column_stack((
            polygon_outline[0],
            polygon_outline[1]))

    return polygon_outline


def extract_polygon_outline_pygeos(pygeos_geometry, type_id) -> np.ndarray:
    """
    Extract the key outline coordinates of a shapely shape.

    Extract the key outline coordinates of a shapely shape
    (including multi-shapes like MultiPolygon). The following
    types ares supported:
        * Polygon:              The outline of the polygon is returned
        * MultiPolygon:         The outline of the largest polygon is returned
        * GeometryCollection:   The largest polygon in the set is returned,
                                if no polygon is present 'None' is returned
        * LineString:           'None' is returned, since the shape is
                                a line and does not host volume information

    For any other type, an error is raised.

    :param shapely_geometry:    shapely-geometry of interest
    :returns:
        * **polygon_outline** - outline coordinates in form of
                                a numpy array with columns x, y

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        09.10.2020

    """
    polygon_outline = None
    # type_id_test = pygeos.get_type_id(pygeos_geometry)
    # if not (type_id == type_id_test):
    #     print('false')

    # if shapely_geometry.geom_type == 'Polygon':
    if type_id == 3:
        if not pygeos.is_empty(pygeos_geometry):
            # polygon_outline = pygeos_geometry.exterior.coords.xy
            polygon_outline = pygeos.get_coordinates(pygeos.get_exterior_ring(pygeos_geometry))

    # elif shapely_geometry.geom_type == 'MultiPolygon':
    elif type_id == 6:
        # extract largest polygon

        # polygon_outline = max(pygeos_geometry, key=lambda a: pygeos.area(a)).exterior.coords.xy
        polygon_outline = pygeos.get_coordinates(pygeos.get_exterior_ring(max(pygeos.get_geometry(pygeos_geometry, range(pygeos.get_num_geometries(pygeos_geometry))),
                                                 key=lambda a: pygeos.area(a))))

    # elif shapely_geometry.geom_type == 'GeometryCollection':
    elif type_id == 7:
        # extract polygons
        polygons = []
        for geometry in pygeos_geometry:
            if (
                # geometry.geom_type == 'Polygon' and
                # not pygeos_geometry.is_empty
                pygeos.get_type_id(geometry) == 3 and not pygeos.is_empty(geometry)
            ):
                polygons.append(geometry)

        # extract largest polygon
        if polygons:
            # polygon_outline = max(polygons, key=lambda a: pygeos.area(a)).exterior.coords.xy
            polygon_outline = pygeos.get_coordinates(pygeos.get_exterior_ring(max(pygeos.get_geometry(pygeos_geometry, range(pygeos.get_num_geometries(pygeos_geometry))),
                                                     key=lambda a: pygeos.area(a))))

    # elif shapely_geometry.geom_type == 'LineString':
    elif type_id == 1:
        # if just line left, skip
        pass

    # else:
    #     raise ValueError("Faced unsupported shape '" +
    #                      str(shapely_geometry.geom_type) + "'!")
    else:
        raise ValueError("Faced unsupported shape '" +
                         str(type_id) + "'!")

    # convert to numpy array
    # if polygon_outline is not None:
    #     polygon_outline = np.column_stack((
    #         polygon_outline[0],
    #         polygon_outline[1]))

    return polygon_outline
