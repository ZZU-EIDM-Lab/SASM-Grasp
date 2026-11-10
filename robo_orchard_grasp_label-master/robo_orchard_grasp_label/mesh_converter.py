# Project RoboOrchard
#
# Copyright (c) 2024–2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Non-Commercial Open Source License Version 1.0.
# To view a copy of this license, visit:
#   https://github.com/HorizonRobotics/robo_orchard_grasp_label/blob/master/LICENSE
from __future__ import annotations
import asyncio
import os

import omni
import omni.kit.commands
from isaaclab.sim.converters.mesh_converter import (
    MeshConverter as MeshConverterBase,
)
from isaaclab.sim.converters.mesh_converter_cfg import MeshConverterCfg
from pxr import (
    Gf,
    PhysxSchema,
    Sdf,
    Tf,
    Usd,
    UsdGeom,
    UsdPhysics,
    UsdUtils,
)


class MeshConverter(MeshConverterBase):
    """Convert mesh files to USD format with additional configurations."""

    cfg: MeshConverterCfg
    """The configuration instance for mesh to USD conversion."""

    def _convert_asset(self, cfg: MeshConverterCfg):
        # resolve mesh name and format
        mesh_file_basename, mesh_file_format = os.path.basename(
            cfg.asset_path
        ).split(".")
        mesh_file_format = mesh_file_format.lower()

        # Check if mesh_file_basename is a valid USD identifier
        if not Tf.IsValidIdentifier(mesh_file_basename):
            # Correct the name to a valid identifier and update the basename
            mesh_file_basename_original = mesh_file_basename
            mesh_file_basename = Tf.MakeValidIdentifier(mesh_file_basename)
            omni.log.warn(
                f"Input file name '{mesh_file_basename_original}' \
                    is an invalid identifier for the mesh prim path."
                f" Renaming it to '{mesh_file_basename}' for the conversion."
            )

        # Convert USD
        asyncio.get_event_loop().run_until_complete(
            self._convert_mesh_to_usd(
                in_file=cfg.asset_path,
                out_file=self.usd_path,
            )
        )

        # Create a new stage, set Z up and meters per unit
        temp_stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(temp_stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(temp_stage, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(temp_stage, 1.0)
        # Add mesh to stage
        base_prim = temp_stage.DefinePrim(f"/{mesh_file_basename}", "Xform")
        prim = temp_stage.DefinePrim(
            f"/{mesh_file_basename}/geometry", "Xform"
        )
        prim.GetReferences().AddReference(self.usd_path)
        temp_stage.SetDefaultPrim(base_prim)
        # export the stage to a new USD file
        export_path = self.usd_path.replace(".usd", "_converted.usd")
        temp_stage.Export(export_path)

        # Open converted USD stage
        stage = Usd.Stage.Open(export_path)
        # Need to reload the stage to get the new prim structure,
        # otherwise it can be taken from the cache
        stage.Reload()
        # Add USD to stage cache
        stage_id = UsdUtils.StageCache.Get().Insert(stage)
        # Get the default prim (which is the root prim) "/{mesh_file_basename}"
        xform_prim = stage.GetDefaultPrim()
        geom_prim = stage.GetPrimAtPath(f"/{mesh_file_basename}/geometry")

        for child_mesh_prim in Usd.PrimRange(geom_prim):
            if child_mesh_prim.GetTypeName() == "Mesh":
                # Unified collision: convexDecomposition + shrinkWrap
                approx_attr = child_mesh_prim.GetAttribute(
                    "physics:approximation"
                )
                if not approx_attr:
                    approx_attr = child_mesh_prim.CreateAttribute(
                        "physics:approximation", Sdf.ValueTypeNames.Token
                    )
                approx_attr.Set("convexDecomposition")

                physx_conv_api = (
                    PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(
                        child_mesh_prim
                    )
                )
                physx_conv_api.GetShrinkWrapAttr().Set(True)

                bind_apis = [
                    "MaterialBindingAPI",
                    "PhysicsMeshCollisionAPI",
                    "PhysicsCollisionAPI",
                    "PhysxCollisionAPI",
                    "PhysicsMassAPI",
                    "PhysicsRigidBodyAPI",
                    "PhysxRigidBodyAPI",
                ]

                api_schemas = child_mesh_prim.GetMetadata("apiSchemas")
                if api_schemas is None:
                    api_schemas = Sdf.TokenListOp()

                api_list: list[str] = list(
                    api_schemas.GetAddedOrExplicitItems()
                )

                for api in bind_apis:
                    if api not in api_list:
                        api_list.append(api)

                api_schemas.appendedItems = api_list
                child_mesh_prim.SetMetadata("apiSchemas", api_schemas)

        # Delete the old Xform and make the new Xform the default prim
        stage.SetDefaultPrim(xform_prim)

        # Apply default Xform rotation to mesh,
        # enable to set rotation and scale
        omni.kit.commands.execute(
            "CreateDefaultXformOnPrimCommand",
            prim_path=xform_prim.GetPath(),
            **{"stage": stage},
        )

        # Apply translation, rotation, and scale to the Xform
        geom_xform = UsdGeom.Xform(geom_prim)
        geom_xform.ClearXformOpOrder()

        # Remove any existing rotation attributes
        rotate_attr = geom_prim.GetAttribute("xformOp:rotateXYZ")
        if rotate_attr:
            geom_prim.RemoveProperty(rotate_attr.GetName())

        # translation
        translate_op = geom_xform.AddTranslateOp(
            UsdGeom.XformOp.PrecisionDouble
        )
        translate_op.Set(Gf.Vec3d(*cfg.translation))
        # rotation
        orient_op = geom_xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        orient_op.Set(Gf.Quatd(*cfg.rotation))
        # scale
        scale_op = geom_xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
        scale_op.Set(Gf.Vec3d(*cfg.scale))

        # Save changes to USD stage
        stage.Save()
        if stage_id is not None:
            UsdUtils.StageCache.Get().Erase(stage_id)
