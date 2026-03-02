#!/usr/bin/env python3

import sys
import time
import tyro
import xacro

import matplotlib.pyplot as plt
import pinocchio as pin

from common import MODELS
from roboplan.core import Scene
from roboplan.example_models import get_package_share_dir
from roboplan.viser_visualizer import ViserVisualizer

import hppfcl
import numpy as np
from roboplan.core import Box, Scene, Sphere, OcTree


def create_table():
    x = np.linspace(0, 1, 10, endpoint=False)
    y = np.linspace(0, 1, 10, endpoint=False)
    z = np.linspace(0.4, 0.7, 2, endpoint=False)
    p3d1 = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

    x = np.linspace(0, 0.3, 2, endpoint=False)
    y = np.linspace(0, 0.3, 2, endpoint=False)
    z = np.linspace(0.0, 0.4, 4, endpoint=False)
    leg1 = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

    x = np.linspace(0.8, 1.0, 2, endpoint=False)
    y = np.linspace(0, 0.3, 2, endpoint=False)
    z = np.linspace(0.0, 0.4, 4, endpoint=False)
    leg2 = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

    x = np.linspace(0.8, 1.0, 2, endpoint=False)
    y = np.linspace(0.8, 1.0, 2, endpoint=False)
    z = np.linspace(0.0, 0.4, 4, endpoint=False)
    leg3 = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

    x = np.linspace(0, 0.3, 2, endpoint=False)
    y = np.linspace(0.8, 1.0, 2, endpoint=False)
    z = np.linspace(0.0, 0.4, 4, endpoint=False)
    leg4 = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

    p3d = np.concatenate([p3d1, leg1, leg2, leg3, leg4])

    return p3d


def setupScene(model: str = "ur5"):
    if model not in MODELS:
        print(f"Invalid model requested: {model}")
        sys.exit(1)

    model_data = MODELS[model]
    package_paths = [get_package_share_dir()]

    # Pre-process with xacro. This is not necessary for raw URDFs.
    urdf_xml = xacro.process_file(model_data.urdf_path).toxml()
    srdf_xml = xacro.process_file(model_data.srdf_path).toxml()

    # Specify argument names to distinguish overloaded Scene constructors from python.
    scene = Scene(
        "test_scene",
        urdf=urdf_xml,
        srdf=srdf_xml,
        package_paths=package_paths,
        yaml_config_path=model_data.yaml_config_path,
    )

    table3d = create_table()
    octree_geometry = hppfcl.makeOctree(table3d, 0.1)

    boxes = octree_geometry.toBoxes()
    resolution = octree_geometry.getResolution()
    name = "octree_cloud"
    parent_frame = "universe"
    tform = pin.SE3(np.eye(3), np.array([0.2, -0.5, 0.0])).homogeneous
    color = np.array([0.3, 1.0, 0.3, 1.0])
    scene.addOcTreeGeometry(
        name,  # name
        parent_frame,  # parent frame
        OcTree(boxes, resolution),
        tform,  # tform
        color,  # color
    )


def main(model: str = "ur5", host: str = "localhost", port: str = "8000"):

    if model not in MODELS:
        print(f"Invalid model requested: {model}")
        sys.exit(1)

    model_data = MODELS[model]
    package_paths = [get_package_share_dir()]

    # Pre-process with xacro. This is not necessary for raw URDFs.
    urdf_xml = xacro.process_file(model_data.urdf_path).toxml()
    srdf_xml = xacro.process_file(model_data.srdf_path).toxml()

    # Specify argument names to distinguish overloaded Scene constructors from python.
    scene = Scene(
        "test_scene",
        urdf=urdf_xml,
        srdf=srdf_xml,
        package_paths=package_paths,
        yaml_config_path=model_data.yaml_config_path,
    )

    # Create a redundant Pinocchio model just for visualization.
    # When Pinocchio 4.x releases nanobind bindings, we should be able to directly grab the model from the scene instead.
    model = pin.buildModelFromXML(urdf_xml)
    collision_model = pin.buildGeomFromUrdfString(
        model, urdf_xml, pin.GeometryType.COLLISION, package_dirs=package_paths
    )
    visual_model = pin.buildGeomFromUrdfString(
        model, urdf_xml, pin.GeometryType.VISUAL, package_dirs=package_paths
    )

    table3d = create_table()
    octree_geometry = hppfcl.makeOctree(table3d, 0.1)

    boxes = octree_geometry.toBoxes()
    resolution = octree_geometry.getResolution()
    name = "octree_cloud"
    parent_frame = "universe"
    tform = pin.SE3(np.eye(3), np.array([0.2, -0.5, 0.0])).homogeneous
    color = np.array([0.3, 1.0, 0.3, 1.0])
    scene.addOcTreeGeometry(
        name,  # name
        parent_frame,  # parent frame
        OcTree(boxes, resolution),
        tform,  # tform
        color,  # color
    )

    geom_obj = pin.GeometryObject(
        name,
        model.getFrameId(parent_frame),
        pin.SE3(tform),
        octree_geometry,
    )
    geom_obj.meshColor = color
    collision_model.addGeometryObject(geom_obj)
    visual_model.addGeometryObject(geom_obj)

    viz = ViserVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=True, loadModel=True, host=host, port=port)

    try:
        while True:
            time.sleep(10.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    tyro.cli(main)
