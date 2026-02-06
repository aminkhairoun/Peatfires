#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 19:16:13 2024

@author: Amin Khairoun
"""

import os, sys, glob
from osgeo import gdal, osr, ogr
from scipy.spatial import Voronoi


def points2wktpoly(points):
    polygon = 'POLYGON (('
    for point in points:
        polygon = '%s%f %f,' %(polygon, point[0], point[1])
    polygon = '%s%f %f))' %(polygon, points[0][0], points[0][1])
    return polygon

def create_shp_gdal(path, geom_type, proj, memory=False):
    driver = ogr.GetDriverByName('Esri Shapefile')
    if os.path.exists(path):
        driver.DeleteDataSource(path)
    if memory:
        shpObject = ogr.GetDriverByName("Memory").CreateDataSource('')
    else:
        shpObject = driver.CreateDataSource(path)
    try:
        spatialref = ogr.osr.SpatialReference(proj)
    except:
        spatialref = ogr.osr.SpatialReference(proj.ExportToWkt())
    layerName = os.path.splitext(os.path.basename(path))[0]
    layer = shpObject.CreateLayer(layerName, spatialref, geom_type=geom_type)
    return shpObject

def add_field_gdal(shpObject, fieldName, dataType=ogr.OFTString, values=None, parameters={}):
    layer = shpObject.GetLayer()
    field = ogr.FieldDefn(str(fieldName), dataType)
    if 'width' in parameters:
        field.SetWidth(parameters['width'])
    if 'precision' in parameters:
        field.SetPrecision(parameters['precision'])
    layer.CreateField(field)
    if values:
        for i in range(layer.GetFeatureCount()):
            feat = layer.GetFeature(i)
            feat.SetField(
                field.GetNameRef(),
                values[i])
            layer.SetFeature(feat)

def add_feature_gdal(shpObject, geometry, fieldsAndValues):
    layer = shpObject.GetLayer()
    layerDefinition = layer.GetLayerDefn()
    feature = ogr.Feature(layerDefinition)
    geometry = ogr.CreateGeometryFromWkt(str(geometry))
    feature.SetGeometry(geometry)
    for i in fieldsAndValues.keys():
        feature.SetField(str(i), fieldsAndValues[i])
    layer.CreateFeature(feature)
    feature = None
        
def remove_features_gdal(shpObject, field_name, list_remove=[], list_keep=[]):
    """ Define either list_remove or list_keep not both """
    source_layer = shpObject.GetLayer()
    for i in range(source_layer.GetLayerDefn().GetFieldCount()):
        field = source_layer.GetLayerDefn().GetFieldDefn(i)
        if field.GetName() == field_name:
            idx = i
            break
    if len(list_keep) > 0:
        for v in list_keep:
            ## SQL command
            clause = f"{field_name} <> '{v}'"
            ## Use the SQL WHERE clause to create a feature filter
            source_layer.SetAttributeFilter(clause)
            for feature in source_layer:
                source_layer.DeleteFeature(feature.GetFID())
            ## Reset the attribute filter
            source_layer.SetAttributeFilter(None)
    else:
        for v in list_remove:
            clause = f"{field_name} = '{v}'"
            source_layer.SetAttributeFilter(clause)
            for feature in source_layer:
                source_layer.DeleteFeature(feature.GetFID())
            source_layer.SetAttributeFilter(None)       

def clip_shp(inpt, extent, output, memory=False, buf=0):
    if isinstance(inpt, str):
        driver = ogr.GetDriverByName('ESRI Shapefile')
        datasource = driver.Open(inpt, 0)
        source_layer = datasource.GetLayer(0)
    elif isinstance(inpt, ogr.DataSource):
        source_layer = inpt.GetLayer(0)
    elif isinstance(inpt, ogr.Layer):
        source_layer = inpt

    if 'shp' in extent.keys():
        path = extent['shp']
        clip_source = driver.Open(path, 0)
        clip_layer = clip_source.GetLayer()
    else:
        ring = ogr.Geometry(ogr.wkbLinearRing)
        if 'bbox' in extent.keys():
            bbox = extent['bbox']
            ring.AddPoint(bbox[0], bbox[1])
            ring.AddPoint(bbox[2], bbox[1])
            ring.AddPoint(bbox[2], bbox[3])
            ring.AddPoint(bbox[0], bbox[3])
            ring.AddPoint(bbox[0], bbox[1])
        elif 'coords' in extent.keys():
            coords = extent['coords']
            ring.AddPoint(coords[0][0], coords[0][1])
            ring.AddPoint(coords[1][0], coords[1][1])
            ring.AddPoint(coords[3][0], coords[3][1])
            ring.AddPoint(coords[2][0], coords[2][1])
            ring.AddPoint(coords[0][0], coords[0][1])           
        polygon = ogr.Geometry(ogr.wkbPolygon)
        polygon.AddGeometry(ring)
        polygon = polygon.Buffer(buf)
        clip_source = ogr.GetDriverByName('Memory').CreateDataSource('')
        clip_layer = clip_source.CreateLayer("polygon_layer", source_layer.GetSpatialRef(), 
                              ogr.wkbPolygon)
        feature_defn = clip_layer.GetLayerDefn()
        feature = ogr.Feature(feature_defn)
        feature.SetGeometry(polygon)
        clip_layer.CreateFeature(feature)
        
    if memory:
        clipped_source = ogr.GetDriverByName("Memory").CreateDataSource('')
    else:
        clipped_source = ogr.GetDriverByName('ESRI Shapefile').CreateDataSource(output)
    clipped_layer = clipped_source.CreateLayer(source_layer.GetName(), source_layer.GetSpatialRef(), 
                              geom_type=source_layer.GetGeomType())
    source_layer.Clip(clip_layer, clipped_layer)
    
    clip_source = None
    datasource = None
    return clipped_source

def create_ring(extent, buf=0):
    ring = ogr.Geometry(ogr.wkbLinearRing)
    if 'bbox' in extent.keys():
        bbox = extent['bbox']
        ring.AddPoint(bbox[0], bbox[1])
        ring.AddPoint(bbox[2], bbox[1])
        ring.AddPoint(bbox[2], bbox[3])
        ring.AddPoint(bbox[0], bbox[3])
        ring.AddPoint(bbox[0], bbox[1])
    elif 'coords' in extent.keys():
        coords = extent['coords']
        ring.AddPoint(coords[0][0], coords[0][1])
        ring.AddPoint(coords[1][0], coords[1][1])
        ring.AddPoint(coords[3][0], coords[3][1])
        ring.AddPoint(coords[2][0], coords[2][1])
        ring.AddPoint(coords[0][0], coords[0][1])           
    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ring)
    polygon = polygon.Buffer(buf) 
    return polygon


def delete_shp(path):
    if os.path.isfile(path):
        driver = ogr.GetDriverByName("ESRI Shapefile")
        driver.DeleteDataSource(path)
    
def warp_shp(inpt, output, targetprj, ref=None, memory_input=False, memory_output=False):
    delete_shp(output)
    if ref:
        if 'tif' in ref.keys():
            tif = gdal.Open(ref['tif'])
            targetprj = osr.SpatialReference()
            targetprj.ImportFromWkt(tif.GetProjection())
        if 'shp' in ref.keys():
            ref_drive = ogr.GetDriverByName("ESRI Shapefile")
            ref_ds = ref_drive.Open(ref['shp'], 0)
            ref_layer = ref_ds.GetLayer()
            targetprj = ref_layer.GetSpatialRef() 
    if isinstance(inpt, str):
        driver = ogr.GetDriverByName('ESRI Shapefile')
        shpObject = driver.Open(inpt, 0)
        source_layer = shpObject.GetLayer(0)
    elif isinstance(inpt, ogr.DataSource):
        source_layer = inpt.GetLayer(0)
    elif isinstance(inpt, ogr.Layer):
        source_layer = inpt            
    # source_layer = check_shp_layer(inpt)
    sourceprj = source_layer.GetSpatialRef()
    
    if memory_input:
        transform = osr.CoordinateTransformation(sourceprj, targetprj)
        if memory_output:
            ## "MEM" for rasters & "Memory" for vectors
            outsource = ogr.GetDriverByName("Memory").CreateDataSource('')            
        else:
            os.makedirs(os.path.dirname(output), exist_ok=True)
            outsource = ogr.GetDriverByName("ESRI Shapefile").CreateDataSource(output)
        outlayer = outsource.CreateLayer(source_layer.GetName(), targetprj, source_layer.GetGeomType())
        for i in range(source_layer.GetLayerDefn().GetFieldCount()):
            field_def = source_layer.GetLayerDefn().GetFieldDefn(i)
            outlayer.CreateField(field_def)  
        for feature in source_layer:
            transformed = feature.GetGeometryRef()
            transformed.Transform(transform)
            geom = ogr.CreateGeometryFromWkb(transformed.ExportToWkb())
            # if "Sinusoidal" in sourceprj.ExportToWkt() and (int(targetprj.GetAttrValue("AUTHORITY", 1)) == 4326):
            if int(targetprj.GetAttrValue("AUTHORITY", 1)) == 4326:
                geom.SwapXY()
            defn = outlayer.GetLayerDefn()
            feat = ogr.Feature(defn)
            feat.SetGeometry(geom)
            for i in range(source_layer.GetLayerDefn().GetFieldCount()):
                feat.SetField(
                    source_layer.GetLayerDefn().GetFieldDefn(i).GetNameRef(),
                    feature.GetField(i))
            outlayer.CreateFeature(feat)
        output = outsource
        # datasource = None
        return output
    else: ## faster this way
        sourceprj = sourceprj.ExportToProj4()
        targetprj = targetprj.ExportToProj4() 
        source_layer = None
        cmd = 'ogr2ogr -f "ESRI Shapefile" -s_srs "%s" -t_srs "%s" %s %s' \
        %(sourceprj, targetprj, output, inpt) 
        os.system(cmd)
        return output

def check_shp_layer(inpt):
    if isinstance(inpt, str):
        driver = ogr.GetDriverByName('ESRI Shapefile')
        shpObject = driver.Open(inpt, 0)
        layer = shpObject.GetLayer(0)
    elif isinstance(inpt, ogr.DataSource) or isinstance(inpt, gdal.Dataset):
        layer = inpt.GetLayer(0)
    elif isinstance(inpt, ogr.Layer):
        layer = inpt
    return layer

def union_shp(inputPath, outputPath, layer=None):
    if not layer:
        layer = os.path.splitext(os.path.basename(inputPath))[0]
    layer = "'" + layer + "'"
    cmd = f'ogr2ogr -f "ESRI Shapefile" {outputPath} {inputPath} \
    -dialect SQLite -sql "SELECT ST_Union(geometry) AS geometry FROM {layer}"'
    os.system(cmd)
    
def dissolve_shp(inputPath, outputPath, attribute, layer=None):
    if not layer:
        layer = os.path.splitext(os.path.basename(inputPath))[0]
    layer = "'" + layer + "'"
    attribute = "'" + attribute + "'"
    cmd = f'ogr2ogr -f "ESRI Shapefile" {outputPath} {inputPath} \
    -dialect SQLite -sql "SELECT ST_Union(geometry), {attribute} FROM {layer} GROUP BY DOY" {attribute}"'
    os.system(cmd)    
    
def rasterize_gdal(inpt, output, gt, Xsize, Ysize, attribute, dtype, memory_input=False, bands=[1],
           memory_output=False, init_values=None, burn_values=None, noData=None, allTouched=True, options=[], kwargs={}):
    creation_options = ["COMPRESS=LZW", "TILED=YES"]
    if attribute:
        options.append(f"ATTRIBUTE={attribute}")
    if burn_values:
        options.append(f"BURN_VALUE={burn_values}")
    if allTouched:
        options.append(f"ALL_TOUCHED={allTouched}")
    n_bands = len(bands)
    if not (memory_input or memory_output):
        xmin = gt[0]
        ymax = gt[3]
        XRes = gt[1]
        YRes = gt[5]
        xmax = xmin + XRes * Xsize
        ymin = ymax + YRes * Ysize
        bounds = xmin, ymin, xmax, ymax
        shpObject = gdal.OpenEx(inpt)
        gdal.Rasterize(output, shpObject, outputType=dtype, xRes=XRes, yRes=YRes, outputBounds=bounds,
            burnValues=burn_values, initValues=init_values, noData=noData, attribute=attribute, format='GTiff', 
            allTouched=allTouched, creationOptions=['COMPRESS=LZW', 'TILED=YES'], **kwargs)  
        
    else:
        if isinstance(inpt, str):
            driver = ogr.GetDriverByName('ESRI Shapefile')
            datasource = driver.Open(inpt, 0)
            source_layer = datasource.GetLayer(0)
        elif isinstance(inpt, ogr.DataSource):
            source_layer = inpt.GetLayer(0)
        elif isinstance(inpt, ogr.Layer):
            source_layer = inpt
        # source_layer = check_shp_layer(inpt)
        if memory_output: 
            ## "MEM" for rasters & "Memory" for vectors
            target_ds = gdal.GetDriverByName('MEM').Create('', Xsize, Ysize, n_bands, dtype)
        else:
            target_ds = gdal.GetDriverByName('GTiff').Create(output, Xsize, Ysize, n_bands, dtype, options=creation_options)
        target_ds.SetGeoTransform(gt)
        target_ds.SetProjection(source_layer.GetSpatialRef().ExportToWkt())
        for i in range(n_bands):
            target_band = target_ds.GetRasterBand(i+1)
            target_band.Fill(init_values)
            target_band.SetNoDataValue(noData)  
        if not burn_values:
            burn_values = []
        gdal.RasterizeLayer(target_ds, bands, source_layer, burn_values=burn_values, options=options, **kwargs)
        if memory_output: 
            return target_ds
        else:
            target_ds = None
            
