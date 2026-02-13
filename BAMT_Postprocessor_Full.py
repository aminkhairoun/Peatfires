#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 18:50:54 2024

@author: amin
"""

from osgeo import gdal, ogr, osr
import os
import numpy as np
import geopandas as gpd
import pandas as pd
import shapely, warnings
from shapely.errors import ShapelyDeprecationWarning
from sklearn.metrics import confusion_matrix
from shapely.geometry import box
import itertools
import glob
import rasterio as rio
from datetime import datetime, timedelta, date, timezone
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import cv2
import json
import io
import operator
from skimage import measure
from scipy.spatial import Voronoi
import pyproj


import Global_Functions
import Gdal_Functions



def filter_BA(local_path, tile, zone, dataset, date_pre, date_post, 
              lower_threshold, min_seeds, logfile=None):
    """All thresholds are derived from RF Models except for Sahel"""
    if not logfile:
        logfile = io.StringIO()
    # Global_Functions.print_and_log(logfile, f"Tile: {tile}, {date_pre}-{date_post}")
    naming = f'BAMT_BA_{zone}_{dataset}_{date_pre}-{date_post}_{tile}'
    with rio.open(f'{local_path}/{zone}/ByTile/{tile}/GEE/PROB/{naming}_PROB.tif') as rs:
        prob = rs.read(1)
    with rio.open(f'{local_path}/{zone}/ByTile/{tile}/GEE/BA/{naming}_BA.tif') as rs:
        BA = rs.read(1)   
        profile = rs.profile
        
    connectivity = 4
    num_labels, labels = cv2.connectedComponents(BA, connectivity, ltype=cv2.CV_32S) ## 32bytes
    components = measure.regionprops(labels, intensity_image=prob)
    filtered_labels = labels.copy()
    masked_labels = []
    for i, prop in enumerate(components):
        if prop.area > min_seeds-1:
            nmax = np.sort(prop.intensity_image.flat)[::-1][min_seeds-1]
            if nmax < lower_threshold:
                masked_labels.append(i+1)
        else:
             masked_labels.append(i+1)
    np.place(filtered_labels, np.isin(filtered_labels, masked_labels), 0) 
    kept = np.count_nonzero(np.unique(filtered_labels))
    Global_Functions.print_and_log(logfile, 
               f"{len(masked_labels)} of {num_labels-1} patches are masked: {kept} are kept")
    np.place(BA, filtered_labels==0, 0)
    return BA


def mosaic_LC(anc_path, tile, year, zone, crs, lc='HRLC',
              res=0.00025, bounds=None, logfile=None):
    if not logfile:
        logfile = io.StringIO()
    if int(tile[5:7]) <= 50:
        gridX = 2
        gridY = 2
    elif int(tile[5:7]) <= 70:
        gridX = 3
        gridY = 2
    else:
        gridX = 6
        gridY = 2
    
    if not bounds:
        top = int(tile[5:7]) if tile[7] == 'N' else -int(tile[5:7])
        bottom = top - gridY
        left = int(tile[8:11]) if tile[11] == 'E' else -int(tile[8:11])
        right = left + gridX
        bounds =  [left, bottom, right, top]
    Global_Functions.print_and_log(logfile, f'\t\t *** {lc} is used for LC masking ***')        
    Global_Functions.print_and_log(logfile, f'\t\t *** Tile bounding box: {bounds} ***')    
    if lc == 'HRLC':
        zones = {'Sahel': 'A01', 
                 'Amazonia': 'A02',
                 'Siberia': 'A03'}
        y_lc = (year - 1) // 5 * 5
        y_lc = 2019 if year == 2020 else y_lc
        if y_lc < 1990:
            y_lc = 1990
        Global_Functions.print_and_log(logfile, f'\t\*** HRLC year used is {y_lc} ***')     
    
        nc_file = f'{anc_path}/Ancillary/ESACCI_HRLC/Raw/ESACCI-HRLC-L4-MAP-CL01-{zones[zone]}MOSAIC-30m-P5Y-{y_lc}-fv01.2.nc'
        output = f'/vsimem/LC_{tile}_{year}.tif'
        gdal.Warp(output, nc_file, resampleAlg='near', dstSRS=crs, 
                  outputBounds=bounds, xRes=res, yRes=res, dstNodata=0,
                  creationOptions = ['COMPRESS=LZW', 'TILED=YES'])
        with rio.open(output) as tmp:
            LC = tmp.read(1)
            np.place(LC, LC == 0, 1)
            np.place(LC, LC > 110, 0)
            np.place(LC, LC > 0, 1)
        gdal.Unlink(output)
    
    elif lc == 'Land_Mask':
        shp_file = f'{anc_path}/Regions/{zone}_Land.shp'
        prj = osr.SpatialReference()
        prj.ImportFromWkt(crs)
        epsg = int(prj.GetAttrValue("AUTHORITY", 1))
        if epsg != 4326:
            warp_land = f'{anc_path}/Ancillary/Land_Masks/Land_Mask_{epsg}_{tile}.shp'
            cmd = "ogr2ogr -f 'ESRI Shapefile' -s_srs EPSG:4326 -t_srs '%s' -clipdst %s %s %s" \
            %(crs, ' '.join([str(i) for i in bounds]), warp_land, shp_file) 
            if not os.path.isfile(warp_land):
                os.system(cmd)
            if len(gpd.read_file(warp_land)) == 0:
                os.system(cmd)
            if len(gpd.read_file(warp_land)) == 0:
                Global_Functions.print_and_log(logfile, f'\t--- Error: Land Mask is failing ---')   
            shp_file = warp_land
        
        rasterized = f'/vsimem/Land_{tile}_{year}.tif'
        rasterized = f'{anc_path}/{zone}/Trials/test_{tile}_{year}.tif'

        gt = [bounds[0], res, 0, bounds[3], 0, -res]
        height = round((bounds[3] - bounds[1]) / res)
        width = round((bounds[2] - bounds[0]) / res)
        rasterized = Gdal_Functions.rasterize_gdal(shp_file, rasterized, gt, width, height, attribute=None, 
                                    dtype=gdal.GDT_Byte, memory_input=False, bands=[1],
                   memory_output=True, init_values=0, burn_values=[1], noData=0, allTouched=True)

        LC = rasterized.GetRasterBand(1).ReadAsArray()
        rasterized = None
    
    return LC

def getTemporalWindow(date_pre, date_post):
    dates = pd.date_range(start=date_pre, end=date_post, freq='MS') ## M start in the end of month
    JDs = [int(datetime.strftime(i, '%Y%j')) for i in dates]
    return JDs

def getMonthlyJD(tile, inputPath, outputPath, anc_path, zone, dataset, date_pre, date_post, logfile=None, 
         patches='SHP', kernel=(3, 3), res=0.00025, process_prob=False, lc='HRLC', overwrite=False, kwargs=None):
    ## Better not to use it, GDAL staurates and the processing freezes
    # gdal.SetCacheMax(64 * 1024 * 1024)
    if not logfile:
        logfile = io.StringIO()
    Global_Functions.print_and_log(logfile, f'\n\t\t{15 * "-"} {tile} {15 * "-"}')
    naming = f'BAMT_BA_{zone}_{dataset}_{date_pre}-{date_post}_{tile}'
    if not overwrite:
        if os.path.isfile(f'{inputPath}/{zone}/ByTile/{tile}/JD/{naming}_JD.tif'):
            return print('already exists')    
        
    if not os.path.isfile(f'{inputPath}/{zone}/ByTile/{tile}/GEE/DATES/{naming}_DATES.tif'):
        Global_Functions.print_and_log(logfile, f'\t\t ** DATES raster of {tile} is missing **')
    if not os.path.isfile(f'{inputPath}/{zone}/ByTile/{tile}/GEE/PROB/{naming}_PROB.tif'):
        Global_Functions.print_and_log(logfile, f'\t\t ** PROB raster of {tile} is missing **')
    if patches == 'SHP':
        if not os.path.isfile(f'{inputPath}/{zone}/ByTile/{tile}/GEE/SHP/{naming}_SHP.shp'):
            Global_Functions.print_and_log(logfile, f'\t\t ** BA shapefile of {tile} is missing **')
    elif not os.path.isfile(f'{inputPath}/{zone}/ByTile/{tile}/GEE/BA/{naming}_BA.tif'):
                Global_Functions.print_and_log(logfile, f'\t\t ** BA binary raster of {tile} is missing **')        
    else:
        Global_Functions.print_and_log(logfile, '\t\t\t+++ All files are available +++')
            
    with rio.open(f'{inputPath}/{zone}/ByTile/{tile}/GEE/DATES/{naming}_DATES.tif') as rs:
        dates = rs.read(1)
        profile = rs.profile
        bounds = rs.bounds
        xRes, yRes = profile['transform'][0], -profile['transform'][4]
        crs = profile['crs'].to_wkt()
    with rio.open(f'{inputPath}/{zone}/ByTile/{tile}/GEE/PROB/{naming}_PROB.tif') as rs:
        prob = rs.read(1)

    if patches == 'SHP':
        shp_file = f'{inputPath}/{zone}/ByTile/{tile}/GEE/SHP/{naming}_SHP.shp'
        missing_dict = json.load(open(f'{inputPath}/{zone}/Logs/Empty_Files_{zone}_{dataset}.json', 'r'))
        if not os.path.isfile(shp_file):
            if tile[5:] in missing_dict[f'{date_pre}-{date_post}']['SHP']:
                Global_Functions.print_and_log(logfile, 
                   f'\t\t ** SHP is missing becasuse no fire, creating a dummy point SHP **')            
                xmin = int(tile[8:11]) if tile[11] == 'E' else -int(tile[8:11])
                ymax = int(tile[5:7]) if tile[7] == 'N' else -int(tile[5:7])
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)    
                    geo = gpd.points_from_xy([xmin+0.5], [ymax-0.5], crs=crs)
                    shp = gpd.GeoDataFrame({'label':[1]}, geometry=geo, crs=crs)
                    shp.to_file(shp_file)
                    Global_Functions.print_and_log(logfile, f'\t\t+++ SHP shapefile of {tile} is saved +++')
            else:
                Global_Functions.print_and_log(logfile, 
                   f'\t\t--- Error: SHP is missing >> Reprocess GEE ---')    

        driver = ogr.GetDriverByName('ESRI Shapefile')
        shpObject = driver.Open(shp_file, 0)
        a = shpObject.GetLayer(0)
        corrupted = []
        for i in range(a.GetFeatureCount()):
            f = a.GetFeature(i)
            if not f.GetGeometryRef():
                corrupted.append(i)
        shpObject = None

        if len(corrupted) > 0:
            Global_Functions.print_and_log(logfile, 
                   f'\t\t--- Error: Tile {tile} is not processed. Shapefile is corrupted ---')
            return {'tile': tile, 'text': logfile.getvalue()}

        else:
            rasterized = f'/vsimem/BA_{tile}_{date_pre}.tif'
            gdal.Rasterize(rasterized, shp_file, outputType=gdal.GDT_Byte, burnValues=1, noData=0,
                       outputSRS=crs, xRes=xRes, yRes=yRes, outputBounds=bounds, format='GTiff',
                           creationOptions = ['COMPRESS=LZW', 'TILED=YES'])
            with rio.open(rasterized) as rs:
                BA = rs.read(1)
    else:
        BA = filter_BA(inputPath, tile, zone, dataset, date_pre, date_post, logfile=logfile, **kwargs)    
    lat = int(tile[5:7]) - 1
    lon = int(tile[8:11])
    prj = osr.SpatialReference()
    prj.ImportFromWkt(crs)
    if int(prj.GetAttrValue("AUTHORITY", 1)) == 4326:
        crs_name = 'WGS 84'
        window = 30 * kernel[0]    ## 3 pixels 
        distance = Global_Functions.create_buffer(lon, lat, window/1000)[0] - lon
        k_lon = round(distance / res)
        kernel = (kernel[0], k_lon)
        bounds = [round(i) for i in bounds]
        ## below 50N we have 2x2 deg, below 70N 3x2 and 6x2 beyond 
        if int(tile[5:7]) <= 50:
            gridX = 2
            gridY = 2
        elif int(tile[5:7]) <= 70:
            gridX = 3
            gridY = 2
        else:
            gridX = 6
            gridY = 2
        ## check whether is is only 1 deg
        if profile['height'] < 4500:
            gridY = 1
        elif profile['width'] < 4500:
            gridX = 1
        width, height = int(gridX / res), int(gridY / res)
        
        Global_Functions.print_and_log(logfile, 
           f'\t\t*** {tile}: {gridX}x{gridY}° and {width}x{height} pixels, crs: {crs_name}, kernel: {kernel} ***')
    else:
        crs_name = prj.GetAttrValue("PROJCS", 0)
        height, width = BA.shape
        res = xRes
        Global_Functions.print_and_log(logfile, 
           f'\t\t*** {tile}: {width}x{height} pixels, crs: {crs_name}, kernel: {kernel} ***')
        
    k = np.ones(kernel, dtype=np.uint8)
    opened = cv2.morphologyEx(BA, cv2.MORPH_OPEN, k, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k, iterations=1)  
    np.place(dates, closed != 1, 0)
    np.place(dates, prob == 200, -1)
    del BA, opened, closed

    if process_prob:
        np.place(prob, prob == 0, 1)
        np.place(prob, dates == -1, 0)
        masked_path_PROB = f'/vsimem/PROB_{tile}_{date_pre}.tif'
        profile2 = profile.copy()
        profile2['dtype'] = 'uint8'
        Global_Functions.write_rasterio(prob, masked_path_PROB, profile=profile2)
    del prob

    masked_path_JD = f'/vsimem/JD_{tile}_{date_pre}.tif'
    Global_Functions.write_rasterio(dates, masked_path_JD, profile=profile)

    os.makedirs(f'{outputPath}/{zone}/ByTile/{tile}/Seasonal/JD', exist_ok=True)
    tmpJD = f'{outputPath}/{zone}/ByTile/{tile}/Seasonal/JD/{naming}_SeasonalJD.tif'
    gdal.Warp(tmpJD, masked_path_JD, resampleAlg='nearest', width=width, height=height,
            srcSRS=crs, dstSRS=crs, outputBounds=bounds, creationOptions = ['COMPRESS=LZW'])
    gdal.Unlink(masked_path_JD)
    if process_prob:
        os.makedirs(f'{outputPath}/{zone}/ByTile/{tile}/Seasonal/PROB', exist_ok=True)
        tmpPROB = f'{outputPath}/{zone}/ByTile/{tile}/Seasonal/PROB/{naming}_SeasonalPROB.tif'
        gdal.Warp(tmpPROB, masked_path_PROB, resampleAlg='nearest', width=width, height=height,
                srcSRS=crs, dstSRS=crs, outputBounds=bounds, creationOptions = ['COMPRESS=LZW'])
        gdal.Unlink(masked_path_PROB)

    dates = rio.open(tmpJD).read(1)
    LC = mosaic_LC(anc_path, tile, int(date_pre[:4]), zone, crs, lc, res, bounds, logfile)
    breaks = getTemporalWindow(date_pre, date_post)
    index = np.digitize(dates, breaks, right=False).astype(np.int8)
    os.makedirs(f'{outputPath}/{zone}/ByTile/{tile}/JD', exist_ok=True)
    profile['width'] = width
    profile['height'] = height
    gt = {'a': res, 'b': profile['transform'][1], 'c': round(profile['transform'][2]), 
          'd': profile['transform'][3], 'e': -res, 'f': round(profile['transform'][5])}
    profile['transform'] = rio.Affine(**gt)
    np.place(dates, LC == 0, -2)
    Global_Functions.write_rasterio(dates, tmpJD, profile=profile)
    Global_Functions.print_and_log(logfile, f'\t\t+++ Seasonal JD is saved +++++')
    profile['dtype'] = 'int16'
    if process_prob:
        probability = rio.open(tmpPROB).read(1)
        profile2 = profile.copy()
        profile2['dtype'] = 'uint8'
        ## np.place(probability, dates == 0, 1) WHY WHY WHY !!!
        np.place(probability, dates < 0, 0)
        Global_Functions.write_rasterio(probability, tmpPROB, profile=profile2)
        Global_Functions.print_and_log(logfile, f'\t\t+++ Seasonal PROB is saved +++++')
    
    Global_Functions.print_and_log(logfile, f'\t\t*** Filtering {len(breaks[:-1])} monthly outputs ***\n')
    for i, b in enumerate(breaks[:-1]):
        month = datetime.strftime(datetime.strptime(str(b), '%Y%j'), '%Y%m01')   
        JD = np.where(index==i+1, dates % 1000, 0).astype(np.int16)
        np.place(JD, dates == -1, -1)
        np.place(JD, dates == -2, -2)
        basename = f'BAMT_BA_{zone}_{dataset}_{month}_{tile}'
        Global_Functions.write_rasterio(JD, 
            f'{outputPath}/{zone}/ByTile/{tile}/JD/{basename}_JD.tif', profile=profile)

        Global_Functions.print_and_log(logfile, 
           f'\t\t+++ Pixels of month {month} of {tile} are: {np.count_nonzero(JD > 0)} +++')
        if process_prob:
            profile2 = profile.copy()
            profile2['dtype'] = 'uint8'
            os.makedirs(f'{outputPath}/{zone}/ByTile/{tile}/PROB', exist_ok=True)
            Global_Functions.write_rasterio(probability, 
                f'{outputPath}/{zone}/ByTile/{tile}/PROB/{basename}_PROB.tif', profile=profile2)                    
        
    return {'tile': tile, 'text': logfile.getvalue()}

def aggregate_tile(tile, year, inputPath, zone, dataset, inputs='seasonal', 
                   save=False, logfile=None):
    if not logfile:
        logfile = io.StringIO()
    # if isinstance(year, int):
    naming = f'{inputPath}/{zone}/ByTile/{tile}/JD/BAMT_BA_{zone}_{dataset}_{year}'
    files = glob.glob(f'{naming.replace("/JD", "/Seasonal/JD")}*_{tile}_SeasonalJD.tif')
    with rio.open(files[0]) as rst:
        arr = rst.read(1)
        profile = rst.profile
        profile['dtype'] = 'int16'

    np.place(arr, arr > 0, arr[arr > 0] % 1000)
    arr = arr.astype(np.int16)
    if len(files) > 1:
        for f in files:
            new = rio.open(f).read(1)
            np.place(new, new > 0, new[new > 0] % 1000)
            new = new.astype(np.int16)
            np.place(arr, (new > 0) & (arr < 0), 0)
            np.place(new, arr != 0, 0)
            arr += new

    if save:
        os.makedirs(f'{inputPath}/{zone}/ByTile/{tile}/Yearly', exist_ok=True)
        file = f'{naming.replace("/JD", "/Yearly")}_{tile}_JD.tif'
        Global_Functions.write_rasterio(arr, file, profile=profile)
    else:
        file = f'/vsimem/BAMT_BA_{zone}_{dataset}_{year}_{tile}_JD.tif'
        Global_Functions.write_rasterio(arr, file, profile=profile) 

    return {'tile': tile, 'text': logfile.getvalue(), 'file': file}
    
def tile_prj_param(tile, anc_path, zone):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    grid_shp = driver.Open(f'{anc_path}/../Regions/Tiles_{zone}.shp', 0)
    layer = grid_shp.GetLayer()
    for i in range(layer.GetLayerDefn().GetFieldCount()):
        name = layer.GetLayerDefn().GetFieldDefn(i).GetName()
        if name == 'TILE':
            t_idx = i
        elif name == 'PROJ':
            p_idx = i
    for i in range(layer.GetFeatureCount()):
        t = layer.GetFeature(i).GetField(t_idx)
        if t == tile[5:]:
            ft = layer.GetFeature(i)
            epsg = int(ft.GetField(p_idx)[5:])
            minlon, maxlon, minlat, maxlat = ft.GetGeometryRef().GetEnvelope()
            break
    grid_shp = None
    layer = None
    wgs = osr.SpatialReference()
    wgs.ImportFromEPSG(4326)
    dst_prj = osr.SpatialReference()
    dst_prj.ImportFromEPSG(epsg)
    transform = osr.CoordinateTransformation(wgs, dst_prj)
    xmin, ymin = np.inf, np.inf
    xmax, ymax = -np.inf, -np.inf
    for i, j in list(itertools.product([minlon, maxlon], [minlat, maxlat])):
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(j, i)  
        point.AssignSpatialReference(dst_prj)
        point.Transform(transform)
        xmin = min(point.GetX(), xmin)
        xmax = max(point.GetX(), xmax)
        ymin = min(point.GetY(), ymin)
        ymax = max(point.GetY(), ymax)   
    bounds = dict(UTM = (xmin, ymin, xmax, ymax),
                  WGS = (minlon, minlat, maxlon, maxlat))
    return dst_prj, bounds
    
def create_voronoi(inpt, output, field, bounds, memory=False):
    layer = Gdal_Functions.check_shp_layer(inpt)       
    crs = layer.GetSpatialRef().ExportToWkt()
    points, values = [], []
    for i in range(layer.GetFeatureCount()):
        ft = layer.GetFeature(i)
        points.append((ft.GetGeometryRef().GetX(), ft.GetGeometryRef().GetY()))
        values.append(ft.GetField(field))
    
    ## we add this ring to have a complete voronoi regions
    xmin, ymin, xmax, ymax = bounds
    maximumDistance = 4 * np.linalg.norm(np.array([xmin, ymax]) - np.array([xmax, ymin]))
    points.append((xmin, ymax+maximumDistance))
    points.append((xmax, ymax+maximumDistance))
    points.append((xmax+maximumDistance, ymax))
    points.append((xmax+maximumDistance, ymin))
    points.append((xmax, ymin-maximumDistance))
    points.append((xmin, ymin-maximumDistance))
    points.append((xmin-maximumDistance, ymin))
    points.append((xmin-maximumDistance, ymax))
    
    vor = Voronoi(points)
    voronoi_polygons = []
    for i, regionID in enumerate(vor.point_region):
        vertices = vor.regions[regionID]
        if -1 not in vertices:
            points = vor.vertices[vertices]
            polygon = Gdal_Functions.points2wktpoly(points)
            voronoi_polygons.append(polygon)

    shpObject = Gdal_Functions.create_shp_gdal('', geom_type=ogr.wkbPolygon, proj=crs, memory=True)
    layer_defn = layer.GetLayerDefn()
    for i in range(layer_defn.GetFieldCount()):
        fdefn = layer_defn.GetFieldDefn(i)  
        if fdefn.GetName() == field:
            datadtype = fdefn.GetType()
            width = fdefn.GetWidth()
            break
    Gdal_Functions.add_field_gdal(shpObject, field, datadtype, {'width': width})
    for i, polygon in enumerate(voronoi_polygons):
        Gdal_Functions.add_feature_gdal(shpObject, polygon, {field: values[i]})
    
    extent = {'coords': list(itertools.product([xmin, xmax], [ymin, ymax]))}
    shpObject = Gdal_Functions.clip_shp(shpObject, extent, output, memory, buf=1000)
    polygon = None
    layer = None
    return shpObject

def get_HS_voronoi(local_path, anc_path, year, tile, zone, dataset, date_field, 
                   buffer=5000, save_wgs=True, save_utm=True, logfile=False):
    if not logfile:
        logfile = io.StringIO()
    ras_file = f'{local_path}/ByTile/{tile}/Yearly/BAMT_BA_{zone}_{dataset}_{year}_{tile}_JD.tif'
    rs = gdal.Open(ras_file)
    xmin = rs.GetGeoTransform()[0]
    ymax = rs.GetGeoTransform()[3]
    XRes = rs.GetGeoTransform()[1]
    YRes = rs.GetGeoTransform()[5]
    xmax = xmin + XRes * rs.RasterXSize
    ymin = ymax + YRes * rs.RasterYSize
    bounds = xmin, ymin, xmax, ymax
    Global_Functions.print_and_log(logfile, f'\n\t\t{15 * "-"} {tile}: {bounds} {15 * "-"}')
    sinus = osr.SpatialReference()
    sinus.ImportFromWkt('PROJCS["unnamed",\
GEOGCS["Unknown datum based upon the custom spheroid", \
DATUM["Not specified (based on custom spheroid)", \
SPHEROID["Custom spheroid",6371007.181,0]], \
PRIMEM["Greenwich",0],\
UNIT["degree",0.0174532925199433]],\
PROJECTION["Sinusoidal"], \
PARAMETER["longitude_of_center",0], \
PARAMETER["false_easting",0], \
PARAMETER["false_northing",0], \
UNIT["Meter",1]]')
    dst_prj = osr.SpatialReference()
    dst_prj.ImportFromWkt(rs.GetProjection())
    transform = osr.CoordinateTransformation(dst_prj, sinus)
    coords = list(itertools.product([xmin, xmax], [ymin, ymax]))
    coords_sinus = []
    for i, j in coords:
        ## for some reason it wasn't working with SIN <--> UTM: USE PYPROJ 
        ## check also if the swap work
        # print(i, j)
        # point = ogr.Geometry(ogr.wkbPoint)
        # point.AddPoint(j, j)  
        # point.AssignSpatialReference(dst_prj)
        # point.Transform(transform)
        # coords.append([point.GetX(), point.GetY()])
        transformer = pyproj.Transformer.from_crs(dst_prj.ExportToWkt(), sinus.ExportToWkt(), always_xy=True)
        coords_sinus.append(transformer.transform(i, j))
        
    toClip = glob.glob(f'{anc_path}/Active_Fires/SIN-proj/*_SIN-proj_{zone}_{year}.shp')[0]
    clipped = Gdal_Functions.clip_shp(toClip, {'coords': coords_sinus}, '', memory=True, buf=buffer)
    clipped_rep = Gdal_Functions.warp_shp(clipped, f'{anc_path}/Trials/VSNPP_clipped.shp', dst_prj, memory_input=True, memory_output=True)
    layer = clipped_rep.GetLayer()
    layer_defn = layer.GetLayerDefn()
    for i in range(layer_defn.GetFieldCount()):
        fdefn = layer_defn.GetFieldDefn(i) 
        if fdefn.GetName() == date_field:
            field_idx = i
            break
    doys = []
    for i in range(layer.GetFeatureCount())[:]:
        v = layer.GetFeature(i).GetField(field_idx)
        try:
            doys.append(int(datetime.strptime(v, '%Y/%m/%d').strftime('%j')))
        except:
            doys.append(int(datetime.strptime(v, '%Y-%m-%d').strftime('%j')))
    Gdal_Functions.add_field_gdal(clipped_rep, 'DOY', ogr.OFTInteger, doys, {'width': 3})
    os.makedirs(f'{local_path}/ByTile/{tile}/Voronoi/TIF', exist_ok=True)
    if len(doys) == 0:
        Global_Functions.print_and_log(logfile, f'\t\t*** 0 DOYs ***') 
        arr = np.zeros((rs.RasterYSize, rs.RasterXSize))
        Global_Functions.write_rasterio(arr, 
            f'{local_path}/ByTile/{tile}/Voronoi/TIF/VSNPP_Hotspots_{year}_{tile}.tif', refPath=ras_file)
        Global_Functions.print_and_log(logfile, f'\t\t+++ Empty raster saved correctly +++')
    else:
        Global_Functions.print_and_log(logfile, 
           f'\t\t*** {len(doys)} DOYs between: [{np.min(doys)}, {np.max(doys)}] ***')       
        os.makedirs(f'{local_path}/ByTile/{tile}/Voronoi/SHP', exist_ok=True)
        ## creating the voronois in UTM
        voronoi_shp = create_voronoi(clipped_rep, 
                         f'{local_path}/ByTile/{tile}/Voronoi/SHP/VSNPP_Hotspots_{year}_{tile}.shp',
                         'DOY', bounds, memory=True)
        clipped = None
        n = voronoi_shp.GetLayer().GetFeatureCount()
        if save_utm:
            gt = [xmin, XRes, 0, ymax, 0, YRes]
            Gdal_Functions.rasterize_gdal(voronoi_shp, f'{local_path}/ByTile/{tile}/Voronoi/TIF/VSNPP_Hotspots_{year}_{tile}.tif', 
                           gt, rs.RasterXSize, rs.RasterYSize,  attribute='DOY', dtype=gdal.GDT_Int16, memory_input=True, 
                           bands=[1], memory_output=False, init_values=0, burn_values=[], noData=0, allTouched=True)
        
        if save_wgs:
            wgs = osr.SpatialReference()
            wgs.ImportFromEPSG(4326)
            _, bounds = tile_prj_param(tile, anc_path, zone)
            bounds = bounds['WGS']
            XRes, YRes = 0.00025, -0.00025
            gt = (bounds[0], XRes, 0, bounds[3], 0, YRes)
            height = int((bounds[1] - bounds[3]) / YRes)
            width = int((bounds[2] - bounds[0]) / XRes)
            os.makedirs(f'{anc_path}/ByTile/{tile}/Voronoi/TIF', exist_ok=True)
            voronoi_rep = Gdal_Functions.warp_shp(voronoi_shp, '', 
                                   wgs, memory_input=True, memory_output=True)
            n = voronoi_rep.GetLayer().GetFeatureCount()
            voronoi_shp = None
            Gdal_Functions.rasterize_gdal(voronoi_rep, f'{anc_path}/ByTile/{tile}/Voronoi/TIF/VSNPP_Hotspots_{year}_{tile}.tif', 
                           gt, width, height,  attribute='DOY', dtype=gdal.GDT_Int16, memory_input=True, 
                           bands=[1], memory_output=False, init_values=0, burn_values=[], noData=0, allTouched=True)
        Global_Functions.print_and_log(logfile, f'\t\t+++ {n} voronois saved correctly +++') 
        
    return {'tile': tile, 'text': logfile.getvalue()}

def get_HS_boxes(local_path, anc_path, year, tile, zone, dataset, date_field, 
                 buffer=3000, edge=0, save_output=True, target_crs='WGS', logfile=False):
    """ Maximum dimensions:
            MODIS-TRACK: 2 KM
            MODIS-SCAN: 4.8 KM
            VIIRS: 0.8 KM """ 
    if not logfile:
        logfile = io.StringIO()

    sinus = osr.SpatialReference()
    sinus.ImportFromWkt('PROJCS["unnamed",\
GEOGCS["Unknown datum based upon the custom spheroid", \
DATUM["Not specified (based on custom spheroid)", \
SPHEROID["Custom spheroid",6371007.181,0]], \
PRIMEM["Greenwich",0],\
UNIT["degree",0.0174532925199433]],\
PROJECTION["Sinusoidal"], \
PARAMETER["longitude_of_center",0], \
PARAMETER["false_easting",0], \
PARAMETER["false_northing",0], \
UNIT["Meter",1]]')
    os.makedirs(f'{local_path}/ByTile/{tile}/HS_Boxes', exist_ok=True)
    ras_file = f'{local_path}/ByTile/{tile}/Yearly/BAMT_BA_{zone}_{dataset}_{year}_{tile}_JD.tif'
    yfield = 0
    xfield = 1
    ## scan is Xdim, track is Ydim
    scan = 3
    track = 4
    rs = gdal.Open(ras_file)
    xmin = rs.GetGeoTransform()[0]
    ymax = rs.GetGeoTransform()[3]
    XRes = rs.GetGeoTransform()[1]
    YRes = rs.GetGeoTransform()[5]
    width = rs.RasterXSize
    height = rs.RasterYSize
    xmax = xmin + XRes * width
    ymin = ymax + YRes * height
    dst_prj = osr.SpatialReference()
    
    if (target_crs == 'UTM') and (XRes == 0.00025):
        wgs = osr.SpatialReference()
        wgs.ImportFromEPSG(4326)
        grid = gpd.read_file(f'{anc_path}/../Regions/BAMT_GEE_downloadableTiles_2d.shp')
        if tile[5:] in grid.TILE.values:
            epsg = grid.loc[grid.TILE == tile[5:], 'PROJ'].values[0][5:]
        else:
            epsg = grid.loc[grid.TILE == f'{tile[5:8]}{int(tile[8:11])-3:03}{tile[11:]}', 'PROJ'].values[0][5:]
        dst_prj.ImportFromEPSG(int(epsg))
        transformer = pyproj.Transformer.from_crs(wgs.ExportToWkt(), dst_prj.ExportToWkt(), always_xy=True)
        ## grids are larger in the bottom in UTM proj
        bbox = [transformer.transform(i, j) for i, j in itertools.product([xmin, xmax], [ymin, ymax])]
        xmin, ymin = np.array(bbox).min(axis=0)
        xmax, ymax = np.array(bbox).max(axis=0)
        XRes, YRes = 30, -30
        width = round((xmax - xmin) / XRes) 
        height = round((ymin - ymax) / YRes)   
    else:
        dst_prj.ImportFromWkt(rs.GetProjection())
           
    bounds = xmin, ymin, xmax, ymax
    Global_Functions.print_and_log(logfile, f'\n\t\t{15 * "-"} {tile}: {bounds} {15 * "-"}')

    transform = osr.CoordinateTransformation(dst_prj, sinus)
    transformer = pyproj.Transformer.from_crs(dst_prj.ExportToWkt(), sinus.ExportToWkt(), always_xy=True)
    coords = list(itertools.product([xmin, xmax], [ymin, ymax]))
    coords_sinus = []
    for i, j in coords:
        coords_sinus.append(transformer.transform(i, j))
        
    toClip = glob.glob(f'{anc_path}/Active_Fires/SIN-proj/*_SIN-proj_{zone}_{year}.shp')[0]
    clipped = Gdal_Functions.clip_shp(toClip, {'coords': coords_sinus}, '', memory=True, buf=buffer)
    layer = clipped.GetLayer()
    layer_defn = layer.GetLayerDefn()
    for i in range(layer_defn.GetFieldCount()):
        fdefn = layer_defn.GetFieldDefn(i) 
        if fdefn.GetName() == date_field:
            field_idx = i
            break
    doys = []
    for i in range(layer.GetFeatureCount())[:]:
        v = layer.GetFeature(i).GetField(field_idx)
        try:
            doys.append(int(datetime.strptime(v, '%Y/%m/%d').strftime('%j')))
        except:
            doys.append(int(datetime.strptime(v, '%Y-%m-%d').strftime('%j')))
    if len(doys) == 0:
        Global_Functions.print_and_log(logfile, f'\t\t*** 0 DOYs ***') 
        arr = np.zeros((height, width))
        if save_output:
            Global_Functions.write_rasterio(arr, 
                f'{local_path}/ByTile/{tile}/HS_Boxes/Hotspot_Boxes_{year}_{tile}.tif', refPath=ras_file)
            Global_Functions.print_and_log(logfile, f'\t\t+++ Empty raster saved correctly +++')
    else:    
        Global_Functions.print_and_log(logfile, 
           f'\t\t*** {len(doys)} DOYs between: [{np.min(doys)}, {np.max(doys)}] ***')   
        layer = clipped.GetLayer()
        hotspot_boxes = Gdal_Functions.create_shp_gdal('', ogr.wkbPolygon, sinus.ExportToWkt(), memory=True)
        Gdal_Functions.add_field_gdal(hotspot_boxes, 'FID', ogr.OFTInteger, {'width': 10})
        Gdal_Functions.add_field_gdal(hotspot_boxes, 'DOY', ogr.OFTInteger, {'width': 3})
        for i in range(layer.GetFeatureCount()):
            feat = layer.GetFeature(i)
            centroid = feat.geometry().GetX(), feat.geometry().GetY()
            x_dim, y_dim = feat.GetField(scan) * 1000 / 2, feat.GetField(track) * 1000 / 2
            bbox = [centroid[0] - x_dim, centroid[1] - y_dim, 
                     centroid[0] + x_dim, centroid[1] + y_dim]
            poly = Gdal_Functions.create_ring({'bbox': bbox})   
            Gdal_Functions.add_feature_gdal(hotspot_boxes, poly, {'DOY': doys[i], 'FID': i}) 
        n = hotspot_boxes.GetLayer().GetFeatureCount()
        xmin = xmin - edge
        ymax = ymax + edge
        gt = [xmin, XRes, 0, ymax, 0, YRes]
        height = height + (2 * int(edge / XRes)) 
        if (xmax == 180) and (int(dst_prj.GetAttrValue("AUTHORITY", 1)) == 4326):
            width = width + int(edge / XRes)
        else:
            width = width + (2 * int(edge / XRes)) 
        
        if target_crs == "SINUS":
            if (xmax == 180) and (int(dst_prj.GetAttrValue("AUTHORITY", 1)) == 4326):
                pass
            else:
                xmax = xmax + edge
            ymin = ymin - edge
            topleft = ogr.Geometry(ogr.wkbPoint)
            topleft.AddPoint(ymax, xmin)  
            topleft.AssignSpatialReference(dst_prj)
            topleft.Transform(transform)
            botright = ogr.Geometry(ogr.wkbPoint)
            botright.AddPoint(ymin, xmax)  
            botright.AssignSpatialReference(dst_prj)
            botright.Transform(transform)
            YRes = (botright.GetY() - topleft.GetY()) / height
            XRes = -YRes
            width = round((botright.GetX() - topleft.GetX()) / XRes)
            gt = [topleft.GetX(), XRes, 0, topleft.GetY(), 0, YRes]       
        else:
            hotspot_boxes = Gdal_Functions.warp_shp(hotspot_boxes, f'{anc_path}/Trials/Hotspot_Boxes.shp', 
                                     dst_prj, memory_input=True, memory_output=True)
        ds = Gdal_Functions.rasterize_gdal(hotspot_boxes, f'{local_path}/ByTile/{tile}/HS_Boxes/Hotspot_Boxes_{year}_{tile}.tif', 
                       gt, width, height, attribute='DOY', dtype=gdal.GDT_Int16, memory_input=True, 
                       bands=[1], memory_output= not save_output, init_values=0, burn_values=None, noData=0, allTouched=True) 

        hotspot_boxes = None
        rep_boxes = None
        Global_Functions.print_and_log(logfile, f'\t\t+++ {n} hotspot pixels saved correctly +++')  
    if save_output == False:
        if not 'ds' in locals().keys():
            ds, n, geo_param = None, 0, None
        else:
            geo_param = {'gt': gt, 'height': height, 'width': width, 'crs': ds.GetProjection()}
        return ds, n, geo_param
    else:
        return {'tile': tile, 'text': logfile.getvalue()}

def get_HS_distance(local_path, anc_path, year, tile, maximum=5000, res=30): 
    ## to consider max_distance:5000 + and pixel_dim_max:4800/2 ~ 8000
    buf = maximum + 3000
    if res == 0.00025:
        lon, lat = int(tile[8:11]), int(tile[5:7])
        max_distance = Global_Functions.create_buffer(lon, lat, maximum/1000)[0] - lon
    else:
        max_distance = maximum
    max_pix = round(max_distance / res)
    boxes_ds, n, geo_param = get_HS_boxes(local_path, anc_path, year, tile, zone, dataset,'ACQ_DATE', 
                          buffer=buf, edge=max_distance, target_crs='UTM', save_output=False)
    if n > 0: 
        src_band = boxes_ds.GetRasterBand(1)
        dst_ds = gdal.GetDriverByName('MEM').Create('', boxes_ds.RasterXSize, boxes_ds.RasterYSize, 1, 
                                        gdal.GDT_Byte)
        dst_ds.SetProjection(boxes_ds.GetProjection())
        dst_ds.SetGeoTransform(boxes_ds.GetGeoTransform())
        dst_band = dst_ds.GetRasterBand(1)
        gdal.ComputeProximity(
            src_band,
            dst_band,
            options=[F"MAXDIST={max_pix}", "NODATA=255"],
        )
        array = dst_band.ReadAsArray() 
        boxes_ds = None
        dst_ds = None
        return array, n, geo_param
    else:
        return n, n, n

def get_water(water_path, local_path, year, tile, profile, zone, dataset, logfile=False):
    if not logfile:
        logfile = io.StringIO()  

    height = profile['height']
    width = profile['width']
    res = profile['transform'][0]
    if res == 30:
        ## expand edges
        edge = 0.1
    else:
        edge = 0
    lon, lat = int(tile[8:11]) - edge, int(tile[5:7]) + edge
    if tile[7] == 'S':
        lat = -lat + 2*edge
    if tile[11] == 'W':
        lon = -lon - 2*edge
    if lat <= 50:
        gridX = 2
        gridY = 2
    elif lat <= 70:
        gridX = 3
        gridY = 2
    else:
        gridX = 6
        gridY = 2
    top = int(np.ceil(lat / 10) * 10)
    left = int(np.floor(lon / 10) * 10)
    if res == 30:
        ## capture the right edge
        right = int(np.floor((lon + gridX + 2*edge) / 10) * 10)
        if right == 180:
            right = left
        botton = int(np.ceil((lat - gridY - 2*edge) / 10) * 10)
        coord_list = list(itertools.product(np.unique([left, right]), [top, botton]))
    else:
        ## - 0.1 to avoid 180E in 66N177E for instance 
        right = int(np.floor((lon + 2 * (width / height) - 0.1) / 10)) * 10
        coord_list = list(itertools.product(np.unique([left, right]), [top]))
    listFiles = []
    Global_Functions.print_and_log(logfile, f'\t\t*** left-top of water tile window: {coord_list} ***')
    for coords in coord_list:
        if coords[1] >= 0:
            lat_name = f'{coords[1]:02}N'
        else:
            lat_name = f'{abs(coords[1]):02}S'
        if coords[0] >= 0:
            lon_name = f'{coords[0]:03}E'
        else:
            lon_name = f'{abs(coords[0]):03}W'
        file = f"{water_path}/{year}/Water_{lat_name}_{lon_name}_{year}_percent.tif"
        if not os.path.isfile(file):
            file = f"{water_path}/{year-1}/Water_{lat_name}_{lon_name}_{year-1}_percent.tif"
            if not os.path.isfile(file):
                file = f"{water_path}/{year-2}/Water_{lat_name}_{lon_name}_{year-2}_percent.tif"
        listFiles.append(file)
    vrt = gdal.BuildVRT(f'/vsimem/Water_{year}_{tile}.vrt', listFiles)
    if res == 30:
        tmp = f'/vsimem/Water_resampled_{year}_{tile}.tif'
        Global_Functions.resample_by_ref(f'/vsimem/Water_{year}_{tile}.vrt', tmp,
                    f'{local_path}/ByTile/{tile}/Yearly/BAMT_BA_{zone}_{dataset}_{year}_{tile}_JD.tif')
        with rio.open(tmp) as rs:
            water = rs.read(1)
    else:
        counterX = int((lon - left) / res)
        counterY = int((top - lat) / res)
        water = vrt.ReadAsArray()[counterY:counterY+height, counterX:counterX+width]
    vrt.FlushCache()
    mask = (water > 80) & (water < 101)
    return mask

def enhance_BA(local_path, anc_path, water_path, year, tile, zone, dataset,
               add_boxes=False, res=0.00025, start_year=2001, end_year=2023, logfile=False):
    if not logfile:
        logfile = io.StringIO() 

    BA_path = f'{local_path}/ByTile/{tile}/Yearly/BAMT_BA_{zone}_{dataset}_{year}_{tile}_JD.tif'
    tmp_BA_NoBoxes = f'{local_path}/ByTile/{tile}/Enhanced/Temporal/BAMT_BA_{zone}_{dataset}_{year}_{tile}_prevJD_NoBoxes.tif'
    tmp_BA = f'{local_path}/ByTile/{tile}/Enhanced/Temporal/BAMT_BA_{zone}_{dataset}_{year}_{tile}_prevJD.tif'
    out_smoothed = f'{local_path}/ByTile/{tile}/Enhanced/Yearly/BAMT_BA_{zone}_{dataset}_{year}_{tile}_JD_Smooth_Dates.tif'
    out_patches = f'{local_path}/ByTile/{tile}/Enhanced/Yearly/BAMT_BA_{zone}_{dataset}_{year}_{tile}_JD_Correct_Patches.tif'
    
    with rio.open(BA_path) as rs:
        BA = rs.read(1)
        profile = rs.profile
    proximity, n, geo_param = get_HS_distance(local_path, anc_path, year, tile, maximum=5000, res=30)
    Global_Functions.print_and_log(logfile, f'\n\t\t{15 * "-"} {tile}: {n} hotspots {15 * "-"}')  
    if year > start_year:
        with rio.open(tmp_BA_NoBoxes) as rs:
            prevMask = rs.read(1)
        np.place(BA, prevMask == 1, 0)
        del prevMask
        
    if n == 0:
        os.makedirs(os.path.dirname(out_smoothed), exist_ok=True)
        # shutil.copy(f'{local_path}/ByTile/{tile}/Yearly/BAMT_BA_{zone}_{dataset}_{year}_{tile}_JD.tif', outfile) 
        if int(tile[5:7]) > 70:     ## eliminate commissions in 72N and 74N
            np.place(BA, BA > 0, 0)
        Global_Functions.write_rasterio(BA, out_smoothed, profile=profile) 
        Global_Functions.print_and_log(logfile, f'\t\t+++ Date enhancement saved correctly: 0 Hotspots +++')
        if add_boxes:
            # water = get_water(water_path, local_path, year, tile, profile, zone, dataset, logfile)
            # np.place(BA, water, -2)
            if year > start_year:
                with rio.open(out_patches.replace(str(year), str(year-1))) as rs:
                    prevBA = rs.read(1)
                np.place(BA, prevBA > 0, 0)
                del prevBA
            os.makedirs(os.path.dirname(out_patches), exist_ok=True)
            Global_Functions.write_rasterio(BA, out_patches, profile=profile) 
            Global_Functions.print_and_log(logfile, f'\t\t+++ Patch enhancement saved correctly: 0 Hotspots +++')
        
        if year < end_year:
            next_condition = np.full(BA.shape, False)
            profile_bin = profile.copy()
            profile_bin['dtype'] = 'uint8'
            os.makedirs(f'{local_path}/ByTile/{tile}/Enhanced/Temporal', exist_ok=True)
            Global_Functions.write_rasterio(next_condition, tmp_BA_NoBoxes.replace(str(year), str(year+1)), 
                                            profile=profile_bin)
            if add_boxes:
                Global_Functions.write_rasterio(next_condition, tmp_BA.replace(str(year), str(year+1)), 
                                            profile=profile_bin) 
    else:  
        max_next_dist = 2000
        profile_sns = profile.copy()
        profile_sns.update({'width': geo_param['width'], 'height': geo_param['height'], 
                  'crs': rio.crs.CRS.from_string(geo_param['crs']), 'dtype': 'uint8', 
                   'transform': rio.Affine(*[geo_param['gt'][i] for i in [1, 2, 0, 4, 5, 3]])})
        tmp = f'/vsimem/prox_{year}_{tile}.tif'
        Global_Functions.write_rasterio(proximity, tmp, profile=profile_sns)
        Global_Functions.resample_by_ref(tmp, tmp.replace('.tif', '_rep.tif'), refPath=BA_path)
        with rio.open(tmp.replace('.tif', '_rep.tif')) as rs:
            proximity = rs.read(1)
        # if int(tile[5:7]) > 70:    ## eliminate commissions in 72N and 74N
        #     np.place(BA, (BA > 0) & (proximity == 255), 0)
        if year > start_year:
            with rio.open(tmp_BA_NoBoxes) as rs:
                prevMask = rs.read(1)
            with rio.open(out_smoothed.replace(str(year), str(year-1))) as rs:
                prevBA = rs.read(1)
            np.place(BA, (prevMask == 1) & (proximity > int(max_next_dist/30)), 0)
            np.place(BA, (prevBA > 0) & (proximity == 255), 0)
            del prevBA, prevMask
        if year < end_year:
            with rio.open(BA_path.replace(str(year), str(year+1))) as rs:
                nextBA = rs.read(1)
            ## if a next year BA pixel is not very near to a hotspot and near to the current year hotspots then it is relocated
            next_proximity, n2, geo_param2 = get_HS_distance(local_path, anc_path, year+1, tile, maximum=5000, res=30)
            if n2 > 0:  
                profile_next = profile.copy()
                profile_next.update({'width': geo_param2['width'], 'height': geo_param2['height'], 
                          'crs': rio.crs.CRS.from_string(geo_param2['crs']), 'dtype': 'uint8', 
                           'transform': rio.Affine(*[geo_param2['gt'][i] for i in [1, 2, 0, 4, 5, 3]])})
                tmp = f'/vsimem/prox_next_{year}_{tile}.tif'
                Global_Functions.write_rasterio(next_proximity, tmp, profile=profile_next)
                Global_Functions.resample_by_ref(tmp, tmp.replace('.tif', '_rep.tif'), refPath=BA_path)
                with rio.open(tmp.replace('.tif', '_rep.tif')) as rs:
                    next_proximity = rs.read(1)
            else:
                next_proximity = np.full(BA.shape, 255, dtype=np.uint8)
    
        with rio.open(f'{local_path}/ByTile/{tile}/Voronoi/TIF/VSNPP_Hotspots_{year}_{tile}.tif') as rs:
            HS_voronoi = rs.read(1)
        HS_voronoi_full = HS_voronoi.copy()
        np.place(HS_voronoi, proximity == 255, 0)

        if year < end_year:
            ## late BA (later than the highest signal detected) detected in the next year (most are early before June) but not near to any hotspot
            next_condition = (nextBA > 0) & (HS_voronoi >= BA) & (next_proximity > proximity)
            os.makedirs(f'{local_path}/ByTile/{tile}/Enhanced/Temporal', exist_ok=True)
            profile_bin = profile.copy()
            profile_bin['dtype'] = 'uint8'
            Global_Functions.write_rasterio(next_condition, tmp_BA_NoBoxes.replace(str(year), str(year+1)), 
                                            profile=profile_bin)
            del next_proximity, nextBA
        else:
            next_condition = np.full(BA.shape, False)

        # condition = (BA > HS_voronoi) & (HS_voronoi > 0) 
        np.place(BA, next_condition, HS_voronoi[next_condition])  

        BA_binary = (BA > 0) * np.ones_like(BA, dtype=np.uint8)
        connectivity = 8
        num_labels, labels = cv2.connectedComponents(BA_binary, connectivity, ltype=cv2.CV_32S) 
        components = measure.regionprops(labels, intensity_image=HS_voronoi)
        for i, prop in enumerate(components[:]):
            bbox = prop.bbox
            try:
                intensity_max = prop.max_intensity
            except:
                intensity_max = prop.intensity_max
            if intensity_max > 0:
                subset = BA[bbox[0]:bbox[2], bbox[1]:bbox[3]] 
                lab_sub = labels[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                HS_sub = HS_voronoi_full[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                np.place(subset, lab_sub == i+1, HS_sub[lab_sub == i+1])

        os.makedirs(os.path.dirname(out_smoothed), exist_ok=True)
        Global_Functions.write_rasterio(BA, out_smoothed, profile=profile)
        Global_Functions.print_and_log(logfile, f'\t\t+++ Date enhancement saved correctly +++')
        # del condition, proximity

        if add_boxes:
            water = get_water(water_path, local_path, year, tile, profile, zone, dataset, logfile)
            with rio.open(f'{local_path}/ByTile/{tile}/HS_Boxes/Hotspot_Boxes_{year}_{tile}.tif') as rs:
                HS_boxes = rs.read(1)
            k = np.ones((3, 3), dtype=np.uint8)
            np.place(HS_boxes, HS_boxes > 0, 1)
            HS_boxes = HS_boxes.astype(np.uint8)
            np.place(HS_boxes, water > 0, 0)
            HS_boxes = cv2.morphologyEx(HS_boxes, cv2.MORPH_CLOSE, k, iterations=1)  
            HS_boxes = cv2.morphologyEx(HS_boxes, cv2.MORPH_OPEN, k, iterations=1)  
            connectivity = 8
            num_labels, labels = cv2.connectedComponents(HS_boxes, connectivity, ltype=cv2.CV_32S) 
            components = measure.regionprops(labels, intensity_image=BA)
            filtered_labels = labels.copy()
            masked_labels = []
            for i, prop in enumerate(components):
                try:
                    intensity_max = prop.max_intensity
                except:
                    intensity_max = prop.intensity_max
                if intensity_max == 0:
                    masked_labels.append(i+1)
            np.place(filtered_labels, np.isin(filtered_labels, masked_labels), 0) 
            kept = np.count_nonzero(np.unique(filtered_labels))
            Global_Functions.print_and_log(logfile, 
               f"\t\t*** {len(masked_labels)} of {num_labels-1} patches are masked: {kept} are kept ***")
            condition = (filtered_labels != 0) & (HS_boxes > 0)
            np.place(BA, condition, HS_voronoi[condition]) 
            del labels, num_labels
            
            # np.place(BA, water, -2)
            # BA = cv2.morphologyEx(BA, cv2.MORPH_CLOSE, k, iterations=1)  
            # BA = cv2.morphologyEx(BA, cv2.MORPH_OPEN, k, iterations=1)   
            if year > start_year:
                with rio.open(tmp_BA) as rs:
                    prevMask = rs.read(1)
                with rio.open(out_patches.replace(str(year), str(year-1))) as rs:
                    prevBA = rs.read(1)
                np.place(BA, (prevMask == 1) & (proximity > int(max_next_dist/30)), 0)
                np.place(BA, (prevBA > 0) & (proximity == 255), 0)
                del prevBA, prevMask
            if year < end_year:
                np.place(next_condition, condition, True)
                Global_Functions.write_rasterio(next_condition, tmp_BA.replace(str(year), str(year+1)), profile=profile_bin)
                del condition, next_condition, HS_boxes

            os.makedirs(os.path.dirname(out_patches), exist_ok=True)
            Global_Functions.write_rasterio(BA, out_patches, profile=profile)   
            Global_Functions.print_and_log(logfile, f'\t\t+++ Patch enhancement saved correctly +++')
    
    return {'tile': tile, 'text': logfile.getvalue()}

def get_grids(lat):
    if lat <= 50:
        gridX = 2
        gridY = 2
    elif lat <= 70:
        gridX = 3
        gridY = 2
    else:
        # gridX = 6
        gridX = 3
        gridY = 2
    return gridX, gridY

def coords2tile(lon, lat):
    if lon >= 0:
        lon = f'{lon:03}E'
    else:
        lon = f'{-lon:03}W'
    if lat >= 0:
        lat = f'{lat:02}N'
    else:
        lat = f'{-lat:02}S'
    return f'{lat}{lon}'
        
def retrieve_patches(local_path, year, tile, zone, dataset, kernel=(11, 11), 
             cutoff=16, connectivity=4, res=0.00025, process_peat=True, logfile=False):
    
    if not logfile:
        logfile = io.StringIO() 

    with rio.open(f'{local_path}/ByTile/{tile}/Enhanced/Yearly/BAMT_BA_{zone}_{dataset}_{year}_{tile}_JD_Correct_Patches.tif') as rs:
        BA0 = rs.read(1)
        profile = rs.profile
    if process_peat:
        with rio.open(f'{local_path}/PeatFire/ByTile/{tile}/Pixel/BA_Peatland_{tile}_{year}.tif') as rs:
            peat0 = rs.read(1)

    lat = int(tile[5:7])
    lon = int(tile[8:11])
    if tile[7] == 'S':
        lat = -lat
    if tile[11] == 'W':
        lon = -lon    
    gridX, gridY = get_grids(lat)

    list_tiles = [tile]
    list_tiles.append(f'TILE-{coords2tile(lon, lat+gridY)}')
    gridXtop = get_grids(lat+gridY)[0]
    list_tiles.append(f'TILE-{coords2tile(lon-gridXtop, lat+gridY)}')
    list_tiles.append(f'TILE-{coords2tile(lon+gridXtop, lat+gridY)}')
    list_tiles.append(f'TILE-{coords2tile(lon, lat-gridY)}')
    gridXbot = get_grids(lat-gridY)[0]
    list_tiles.append(f'TILE-{coords2tile(lon-gridXbot, lat-gridY)}')
    list_tiles.append(f'TILE-{coords2tile(lon+gridXbot, lat-gridY)}')
    list_tiles.append(f'TILE-{coords2tile(lon-gridX, lat)}')
    list_tiles.append(f'TILE-{coords2tile(lon+gridX, lat)}')
    list_tiles = sorted(list_tiles)
    files_BA = []
    files_peat = []
    left, right, botton, top = 0, 0, 0, 0
    for t in list_tiles:
        f = f'{local_path}/ByTile/{t}/Enhanced/Yearly/BAMT_BA_{zone}_{dataset}_{year}_{t}_JD_Correct_Patches.tif'
        if os.path.isfile(f):
            files_BA.append(f)
            files_peat.append(f'{local_path}/PeatFire/ByTile/{t}/Pixel/BA_Peatland_{t}_{year}.tif')
            ## add 3 deg in Lon edge and 1 deg in lat
            deg = int(1 / res)
            Xedge = 3 * deg
            Yedge = 2 * deg
            if str(lon-gridX) in f:
                left = Xedge
            if str(lon+gridX) in f:
                right = Xedge
            if str(lat+gridY) in f:
                top = Yedge
            if str(lat-gridY) in f:
                botton = Yedge

    list_tiles_avail = [f.split('_')[-4] for f in files_BA]
    Global_Functions.print_and_log(logfile, f'\n\t\t{15 * "-"} {tile}: Neighboors {list_tiles_avail} {15 * "-"}')              
    width = profile['width'] + left + right 
    height = profile['height'] + top + botton
    boundary = [top, left, top + profile['height'], profile['width'] + left]
    Global_Functions.print_and_log(logfile, f"\t\t*** {tile}: Boundary corners in 3x2 deg added edges: {boundary} ***")

    BA = np.zeros(shape=(int(height), int(width)), dtype=np.int16)
    BA[boundary[0]:boundary[2], boundary[1]:boundary[3]] = BA0
    if process_peat:
        peatfire = np.zeros(shape=(int(height), int(width)), dtype=np.int16)
        peatfire[boundary[0]:boundary[2], boundary[1]:boundary[3]] = peat0
        del peat0
    del BA0

    Xgrids = [gridXtop, gridX, gridXbot]
    for i in range((top + botton) // Yedge + 1):
        uly = (lat + gridY) - (gridY * i)
        for j in range((left + right) // Xedge + 1):
            ulx = (lon - Xgrids[i]) + (Xgrids[i] * j)
            t = f'TILE-{uly:02}N{ulx:03}E'
            sliceY = [Yedge * i + (gridY * deg - Yedge) * (i == 2), Yedge * i + (Yedge) * (i == 0) + (gridY * deg) * (i > 0)]
            sliceX = [Xedge * j + (gridX * deg - Xedge) * (j == 2), Xedge * j + (Xedge) * (j == 0) + (gridX * deg) * (j > 0)]
            if t in list_tiles_avail:
                arr = rio.open(f'{local_path}/ByTile/{t}/Enhanced/Yearly/BAMT_BA_{zone}_{dataset}_{year}_{t}_JD_Correct_Patches.tif').read(1)
                arr = arr[-Yedge*(i==0):Yedge*(i==2)+arr.shape[0]*(i!=2), -Xedge*(j==0):Xedge*(j==2)+arr.shape[1]*(j!=2)]
                BA[sliceY[0]:sliceY[1], sliceX[0]:sliceX[1]] = arr
                if process_peat:
                    arr = rio.open(f'{local_path}/PeatFire/ByTile/{t}/Pixel/BA_Peatland_{t}_{year}.tif').read(1)
                    arr = arr[-Yedge*(i==0):Yedge*(i==2)+arr.shape[0]*(i!=2), -Xedge*(j==0):Xedge*(j==2)+arr.shape[1]*(j!=2)]
                    peatfire[sliceY[0]:sliceY[1], sliceX[0]:sliceX[1]] = arr      

    gt = rio.Affine(res, 0.0, lon - (left / deg),
                0.0, -res, lat + (top / deg))
    profile_exp = profile.copy()
    profile_exp.update({'width': width, 'height': height, 'transform': gt})    
    a = Global_Functions.get_pixel_area(profile_exp, path=False, profile=True, area_1d=True)
    np.place(BA, BA <= 0, 0)
    if process_peat:
        val_old = np.array([-3, -2, 0, 2, 3])
        val_new = np.array([0, 25, 0, 75, 100])
        peatfire = Global_Functions.replace_numpy_list(peatfire, val_old, val_new).astype(np.int8)
    window = 30 * kernel[0]  
    distance = Global_Functions.create_buffer(lon, lat-1, window/1000)[0] - lon
    k_lon = round(distance / res)
    kernel = (kernel[0], k_lon)
    k = np.ones(kernel, dtype=np.uint8)
    ## to smooth and catch splitted fires closer than 5*30*2 = 300 m
    dilated = cv2.dilate(BA, k, iterations=1)
    smoothed = BA.copy()
    smoothed[(dilated>0) & (smoothed==0)] = dilated[(dilated>0) & (smoothed==0)]
    del dilated  
    BA_binary = (smoothed > 0) * np.ones_like(smoothed, dtype=np.uint8)
    num_labels, labels = cv2.connectedComponents(BA_binary, connectivity, ltype=cv2.CV_32S) 
    Global_Functions.write_log(logfile, f"\t\t*** Number of connected componnents of {t}: {num_labels-1} ***")
    del BA_binary 
    components = measure.regionprops(labels, intensity_image=smoothed)

    i = 0
    j = 0
    added = 100000
    for prop in components[:]:
        bbox = prop.bbox
        centroid = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        if any([x > boundary[1] and x < boundary[3] for x in [bbox[1], bbox[3], centroid[1]]]):
            if any([y > boundary[0] and y < boundary[2] for y in [bbox[0], bbox[2], centroid[0]]]):
                # print(f'component: {i}: {bbox}, {centroid}' )
                intensity_image = prop.intensity_image.astype(np.float32)
                intensity_image[intensity_image == 0] = np.nan
                patches = intensity_image.copy()
                ## to avoid mixing with DOYs and labels that can exceed even 1000
                j = added + 1 + j % added    
                stop = False
                while not stop:
                    min_idx = np.argwhere(intensity_image == np.nanmin(intensity_image))
                    cv2.floodFill(patches, None, min_idx[0][::-1], j, upDiff=cutoff, loDiff=cutoff, flags=8)
                    intensity_image[patches == j] = np.nan
                    j += 1
                    if np.count_nonzero(np.isfinite(intensity_image)) == 0:
                        j -= 1
                        stop = True
                del intensity_image
                lab_subset = labels[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                lab_subset[patches >= added] = patches[patches >= added].astype(lab_subset.dtype)
                i += 1

    np.place(labels, labels < added, added)
    labels -= added  ## go back to start from 1
    np.place(labels, BA <= 0, 0)
    regions = measure.regionprops(labels)
    ds_patches = {}
    ds_patches['ID'] = []
    ds_patches['BOUNDS'] = []
    ds_patches['AREA'] = []
    ds_patches['DUPPLICATED'] = []
    ds_patches['MIN_JD'] = []
    ds_patches['MAX_JD'] = []
    ds_patches['DURATION'] = []
    ds_patches['MED_JD'] = []
    ds_patches['MEAN_JD'] = []
    ds_patches['ALL_JDs'] = []     
    if process_peat:
        ds_patches['PEAT_FRACTION'] = []
        ds_patches['PEAT_CONFIDENT'] = []
    l = 1
    i = 1
    for p in regions:
        centroid = (p.bbox[2] + p.bbox[0]) / 2, (p.bbox[3] + p.bbox[1]) / 2
        xmin = lon + (p.bbox[1] - left) * res
        xmax = lon + (p.bbox[3] - left) * res
        ymin = lat - (p.bbox[2] - top) * res
        ymax = lat - (p.bbox[0] - top) * res
        bounds = xmin, ymin, xmax, ymax
        inside = False
        if ((p.coords[:, 0] > boundary[0]) & (p.coords[:, 0] < boundary[2]) & \
                (p.coords[:, 1] > boundary[1]) & (p.coords[:, 1] < boundary[3])).any():
            inside = True

        if inside:
            sub = labels[max(p.bbox[0], boundary[0]):min(p.bbox[2], boundary[2]), 
                           max(p.bbox[1], boundary[1]):min(p.bbox[3], boundary[3])]
            while np.count_nonzero(sub == l) == 0:
                l += 1
            del sub
            # print(f'component: {i}: {p.bbox}, {centroid}, {l}')
            patch = labels[p.bbox[0]:p.bbox[2], p.bbox[1]:p.bbox[3]]
            np.place(patch, patch == l, i)
            ds_patches['ID'].append(f'{tile}_{i:05}')
            ds_patches['BOUNDS'].append([xmin, ymin, xmax, ymax])  
            lab_subset = labels[p.bbox[0]:p.bbox[2], p.bbox[1]:p.bbox[3]].copy()
            lab_subset[lab_subset != i] = 0
            lab_subset[lab_subset > 0] = 1
            area = (a[p.bbox[0]:p.bbox[2]] * lab_subset.T).T
            ds_patches['AREA'].append(area.sum())
            jd = BA[p.bbox[0]:p.bbox[2], p.bbox[1]:p.bbox[3]]
            jd = jd[lab_subset==1]
            del lab_subset
            ds_patches['ALL_JDs'].append(jd)
            ds_patches['MIN_JD'].append(jd.min())
            ds_patches['MAX_JD'].append(jd.max())
            ds_patches['DURATION'].append(jd.max() - jd.min() + 1)
            ds_patches['MEAN_JD'].append(jd.mean())
            ds_patches['MED_JD'].append(np.median(jd))
            if any([x < boundary[1] or x > boundary[3] for x in [p.bbox[1], p.bbox[3], centroid[1]]]):
                ds_patches['DUPPLICATED'].append(1)
            elif any([y < boundary[0] and y > boundary[2] for y in [p.bbox[0], p.bbox[2], centroid[0]]]):
                ds_patches['DUPPLICATED'].append(1)
            else:
                ds_patches['DUPPLICATED'].append(0)     
            if process_peat:
                peat = peatfire[p.bbox[0]:p.bbox[2], p.bbox[1]:p.bbox[3]]
                ds_patches['PEAT_FRACTION'].append(np.nansum(peat * area) / area.sum())
                ds_patches['PEAT_CONFIDENT'].append((area * (peat == 100)).sum())
            i += 1
            l += 1

    labels = labels[boundary[0]:boundary[2], boundary[1]:boundary[3]]
    os.makedirs(f"{local_path}//ByTile/{tile}/Patches/TIF", exist_ok=True)
    outfile = f"{local_path}/ByTile/{tile}/Patches/TIF/BAMT_BA_{zone}_{dataset}_Patches-{cutoff}_{year}_{tile}.tif"
    Global_Functions.write_rasterio(labels, outfile, profile=profile)    
    Global_Functions.print_and_log(logfile, f'\t\t+++ {tile}: {i} patches are saved correctly +++')
    logs = {'tile': tile, 'text': logfile.getvalue()}
    df = pd.DataFrame(ds_patches)
    os.makedirs(f"{local_path}/ByTile/{tile}/Patches/Tabular", exist_ok=True)
    df.drop(columns='ALL_JDs').to_csv(f"{local_path}/ByTile/{tile}/Patches/Tabular/BAMT_BA_{zone}_{dataset}_Patch-Stats-{cutoff}_{year}_{tile}.csv", index=False)
    df.to_json(f"{local_path}/ByTile/{tile}/Patches/Tabular/BAMT_BA_{zone}_{dataset}_Patch-Stats-{cutoff}_{year}_{tile}.json", orient="records")

    return dict(logs)


def aggregate_peatBA(tile, year, local_path, res=0.25, dataset='Lndst', 
                     zone='Siberia', process_pixel=True, logfile=None): 
    if not logfile:
        logfile = io.StringIO()   
    with rio.open(f'{local_path}/ByTile/{tile}/Enhanced/Yearly/BAMT_BA_{zone}_{dataset}_{year}_{tile}_JD_Correct_Patches.tif') as ras:
        BA = ras.read(1)
        profile = ras.profile
        bounds = ras.bounds
        height, width = ras.height, ras.width
        dstSRS = ras.crs
    np.place(BA, BA > 1, 1)
    np.place(BA, BA < 0, 0)
    count_BA = np.count_nonzero(BA > 0)
    with rio.open(f'{local_path}/PeatMap/Resampled/ByTile/PeatMap_{tile}.tif') as ras:
        peat = ras.read(1).astype(np.int8)
    if process_pixel:
        Global_Functions.print_and_log(logfile, f'\n\t\t #### {tile}: {width}x{height} pixels')
        np.place(peat, peat == 0, -3)
        np.place(peat, peat == 1, -2)
        np.place(peat, peat == 4, 0)
        peatfire_class = peat.copy()
        np.place(peat, peat > 0, 1) 
        np.place(peat, peat < 0, 0)
        peatfire = BA.copy().astype(np.int8)
        peatfire_class = peatfire_class.copy().astype(np.int8)
        np.place(peatfire, peat == 0, 0)
        np.place(peatfire_class, BA == 0, 0)
        count_peatfire = np.count_nonzero(peatfire > 0)
        if count_BA > 0:
            pc = count_peatfire / count_BA * 100
        else:
            pc = 0

        Global_Functions.print_and_log(logfile, 
                f'\t ** Of {count_BA} pixels burned, {count_peatfire} are inside peat ({pc:.2f}%)')
        os.makedirs(f'{local_path}/PeatFire/ByTile/{tile}/Pixel', exist_ok=True)
        Global_Functions.write_rasterio(peatfire_class, 
                f'{local_path}/PeatFire/ByTile/{tile}/Pixel/BA_Peatland_{tile}_{year}.tif', profile=profile)
        Global_Functions.print_and_log(logfile, f'\t +++ BA_Peatland_{tile}_{year}.tif has been created')
    else:
        with rio.open(f'{local_path}/PeatFire/ByTile/{tile}/Pixel/BA_Peatland_{tile}_{year}.tif') as ras:
            peatfire = ras.read(1)
        peatfire[peatfire < 0] = 0
        peatfire[peatfire > 0] = 1
        peat[(peat < 2) | (peat > 3)] = 0
        peat[peat > 0] = 1
        count_peatfire = np.count_nonzero(peatfire > 0)
        Global_Functions.print_and_log(logfile, 
                f'\t ** Of {count_BA} pixels burned, {count_peatfire} are inside peat ({pc:.2f}%)')
        
    area = Global_Functions.get_pixel_area(profile, path=False, profile=True, area_1d=True)
    scale = int(height / (2 / res))
    BA = (BA.transpose() * area).transpose()
    peat = (peat.transpose() * area).transpose()
    agg_BA = BA.reshape(BA.shape[0] // scale, scale, 
                        BA.shape[1] // scale, scale).sum(axis=(1, 3))
    del BA
    agg_peat = peat.reshape(peat.shape[0] // scale, scale, 
                            peat.shape[1] // scale, scale).sum(axis=(1, 3))
    del peat
    agg_peatfire = peatfire.reshape(peatfire.shape[0] // scale, scale, 
                                  peatfire.shape[1] // scale, scale).sum(axis=(1, 3))
    del peatfire
    
    gt = rio.Affine(res, 0, bounds[0], 0, -res, bounds[3])
    width = int(width / height * 2 / res)
    height = int(2 / res)
    dtype = 'float32'
    profile.update(dict(zip(['transform', 'height', 'width', 'dtype'], 
                            [gt, height, width, dtype])))
    
    frac_fire_in_peat = agg_peatfire / agg_peat * 100
    frac_peat_in_BA = agg_peatfire / agg_BA * 100
    
    os.makedirs(f'{local_path}/PeatFire/ByTile/{tile}/Grid', exist_ok=True)
    Global_Functions.write_rasterio(agg_BA, 
            f'{local_path}/PeatFire/ByTile/{tile}/Grid/BA_Area_{str(res).replace(".", "")}D_{tile}_{year}.tif', profile=profile)
    Global_Functions.write_rasterio(agg_peat, 
            f'{local_path}/PeatFire/ByTile/{tile}/Grid/PeatMap_Area_{str(res).replace(".", "")}D_{tile}_{year}.tif', profile=profile)
    Global_Functions.write_rasterio(agg_peatfire, 
            f'{local_path}/PeatFire/ByTile/{tile}/Grid/PeatFire_Area_{str(res).replace(".", "")}D_{tile}_{year}.tif', profile=profile)
    Global_Functions.write_rasterio(frac_fire_in_peat, 
            f'{local_path}/PeatFire/ByTile/{tile}/Grid/BA_Fraction_Peatland_{str(res).replace(".", "")}D_{tile}_{year}.tif', profile=profile)
    Global_Functions.write_rasterio(frac_peat_in_BA, 
            f'{local_path}/PeatFire/ByTile/{tile}/Grid/PeatFire_Fraction_{str(res).replace(".", "")}D_{tile}_{year}.tif', profile=profile)
    Global_Functions.print_and_log(logfile, f'\t +++ aggregataed files have been created')
    logs = {'tile': tile, 'text': logfile.getvalue()}
    stats = {'Area_BA': agg_BA.sum(), 
             'Area_PeatFire': agg_peatfire.sum(), 
             'Area_Peat': agg_peat.sum(), 
             'PeatFire_frac': agg_peatfire.sum() / agg_BA.sum() * 100,
             'BurnPeat_frac': agg_peatfire.sum() / agg_peat.sum() * 100,
            }
    return dict(logs, **stats)
    
    
def get_FRP(local_path, anc_path, year, tile, zone, dataset, fieldname, 
   buffer=5000, save_wgs=True, save_utm=False, layername='FRP', logfile=False):
    if not logfile:
        logfile = io.StringIO()
    ras_file = f'{local_path}/ByTile/{tile}/Yearly/BAMT_BA_{zone}_{dataset}_{year}_{tile}_JD.tif'
    rs = gdal.Open(ras_file)
    xmin = rs.GetGeoTransform()[0]
    ymax = rs.GetGeoTransform()[3]
    XRes = rs.GetGeoTransform()[1]
    YRes = rs.GetGeoTransform()[5]
    xmax = xmin + XRes * rs.RasterXSize
    ymin = ymax + YRes * rs.RasterYSize
    bounds = xmin, ymin, xmax, ymax
    Global_Functions.print_and_log(logfile, f'\n\t\t{15 * "-"} {tile}: {bounds} {15 * "-"}')
    sinus = osr.SpatialReference()
    sinus.ImportFromWkt('PROJCS["unnamed",\
GEOGCS["Unknown datum based upon the custom spheroid", \
DATUM["Not specified (based on custom spheroid)", \
SPHEROID["Custom spheroid",6371007.181,0]], \
PRIMEM["Greenwich",0],\
UNIT["degree",0.0174532925199433]],\
PROJECTION["Sinusoidal"], \
PARAMETER["longitude_of_center",0], \
PARAMETER["false_easting",0], \
PARAMETER["false_northing",0], \
UNIT["Meter",1]]')
    dst_prj = osr.SpatialReference()
    dst_prj.ImportFromWkt(rs.GetProjection())
    transform = osr.CoordinateTransformation(dst_prj, sinus)
    coords = list(itertools.product([xmin, xmax], [ymin, ymax]))
    coords_sinus = []
    for i, j in coords:
        transformer = pyproj.Transformer.from_crs(dst_prj.ExportToWkt(), sinus.ExportToWkt(), always_xy=True)
        coords_sinus.append(transformer.transform(i, j))
        
    toClip = glob.glob(f'{anc_path}/Active_Fires/SIN-proj/*_SIN-proj_{zone}_{year}.shp')[0]
    clipped = Gdal_Functions.clip_shp(toClip, {'coords': coords_sinus}, '', memory=True, buf=buffer)
    clipped_rep = Gdal_Functions.warp_shp(clipped, '', dst_prj, memory_input=True, memory_output=True)
    layer = clipped_rep.GetLayer()
    layer_defn = layer.GetLayerDefn()
    for i in range(layer_defn.GetFieldCount()):
        fdefn = layer_defn.GetFieldDefn(i) 
        if fdefn.GetName() == fieldname:
            field_idx = i
            break
    
    n = layer.GetFeatureCount()
    os.makedirs(f'{local_path}/ByTile/{tile}/Voronoi/{layername}', exist_ok=True)
    if n == 0:
        Global_Functions.print_and_log(logfile, f'\t\t*** 0 Hotspots ***') 
        arr = np.zeros((rs.RasterYSize, rs.RasterXSize))
        if save_utm:
            os.makedirs(f'{local_path}/ByTile/{tile}/Voronoi/{layername}', exist_ok=True)
            Global_Functions.write_rasterio(arr, 
                f'{local_path}/ByTile/{tile}/Voronoi/{layername}/Hotspots_{layername}_{year}_{tile}.tif', refPath=ras_file)
        if save_wgs:
            os.makedirs(f'{anc_path}/ByTile/{tile}/Voronoi/{layername}', exist_ok=True)

            _, bounds = tile_prj_param(tile, anc_path, zone)
            bounds = bounds['WGS']
            XRes, YRes = 0.00025, -0.00025
            height = int((bounds[1] - bounds[3]) / YRes)
            width = int((bounds[2] - bounds[0]) / XRes)  
            arr = np.zeros((height, width), dtype=np.int8) 
            Global_Functions.write_rasterio(arr, 
                f'{anc_path}/ByTile/{tile}/Voronoi/{layername}/Hotspots_{layername}_{year}_{tile}.tif', 
                refPath=ras_file.replace(local_path, anc_path))
            
        Global_Functions.print_and_log(logfile, f'\t\t+++ Empty raster saved correctly +++')
    else:
        Global_Functions.print_and_log(logfile,  f'\t\t*** {n} HS points ***')       
        ## creating the voronois in UTM
        voronoi_shp = create_voronoi(clipped_rep, '', fieldname, bounds, memory=True)
        shp = None 
        clipped = None
        clipped_shp = None
        n = voronoi_shp.GetLayer().GetFeatureCount()
        if save_utm:
            gt = [xmin, XRes, 0, ymax, 0, YRes]
            Gdal_Functions.rasterize_gdal(voronoi_shp, f'{local_path}/ByTile/{tile}/Voronoi/{layername}/Hotspots_{layername}_{year}_{tile}.tif', 
                           gt, rs.RasterXSize, rs.RasterYSize, attribute=fieldname, dtype=gdal.GDT_Float32, memory_input=True, 
                           bands=[1], memory_output=False, init_values=0, burn_values=[], noData=0, allTouched=True)
        
        if save_wgs:
            wgs = osr.SpatialReference()
            wgs.ImportFromEPSG(4326)
            _, bounds = tile_prj_param(tile, anc_path, zone)
            bounds = bounds['WGS']
            XRes, YRes = 0.00025, -0.00025
            gt = (bounds[0], XRes, 0, bounds[3], 0, YRes)
            height = int((bounds[1] - bounds[3]) / YRes)
            width = int((bounds[2] - bounds[0]) / XRes)
            os.makedirs(f'{anc_path}/ByTile/{tile}/Voronoi/{layername}', exist_ok=True)
            voronoi_rep = Gdal_Functions.warp_shp(voronoi_shp, '', 
                                   wgs, memory_input=True, memory_output=True)
            n = voronoi_rep.GetLayer().GetFeatureCount()
            voronoi_shp = None
            Gdal_Functions.rasterize_gdal(voronoi_rep, f'{anc_path}/ByTile/{tile}/Voronoi/{layername}/Hotspots_{layername}_{year}_{tile}.tif', 
                           gt, width, height, attribute=fieldname, dtype=gdal.GDT_Float32, memory_input=True, 
                           bands=[1], memory_output=False, init_values=0, burn_values=[], noData=0, allTouched=True)
            
        Global_Functions.print_and_log(logfile, f'\t\t+++ {n} voronois saved correctly +++') 
        
    return {'tile': tile, 'text': logfile.getvalue()} 


def get_zonal_param(zone, ESACCI=True, UTM=False):
    if zone == 'Amazonia':
        anc_path = '/home/amin/firecci/0_Inputs/Amin/BAMT_GEE'
        local_path = '/home/amin/firecci/0_Inputs/Amin/BAMT_GEE'   
        lc, threshold, min_seeds  = 'HRLC', 88, 1
        Years = range(2019, 1989, -1)
    elif (zone == 'Siberia') or (zone == 'ABoVE'):
        if ESACCI:
            anc_path = '/home/amin/firecci/0_Inputs/Amin/BAMT_GEE'
            local_path = '/home/amin/firecci/0_Inputs/Amin/BAMT_GEE'
            lc, threshold, min_seeds  = 'HRLC', 87, 4
            Years = range(2019, 1989, -1)
        elif UTM:
            anc_path = '/media/amin/STORAGE/STORAGE/OneDrive/PhD/Landsat_BA'
            local_path = '/home/amin/Desktop/Amin/PhD/Landsat_BA'
            lc, threshold, min_seeds  = 'Land_Mask', 87, 4   
            Years = range(2023, 2000, -1)
        else:
            anc_path = '/media/amin/STORAGE/STORAGE/OneDrive/PhD/Landsat_BA'
            local_path = '/media/amin/STORAGE/STORAGE/OneDrive/PhD/Landsat_BA'
            lc, threshold, min_seeds  = 'Land_Mask', 87, 4  
            Years = range(2023, 2000, -1)
    elif zone == 'Sahel':  
        anc_path = '/home/amin/firecci/0_Inputs/Amin/BAMT_GEE'
        local_path = '/home/amin/firecci/0_Inputs/Amin/BAMT_GEE'       
        lc, threshold, min_seeds  = 'HRLC', 86, 1
        Years = range(2019, 1989, -1)
        
    return dict(anc_path=anc_path, local_path=local_path, ESACCI=ESACCI,
               lc=lc, threshold=threshold, min_seeds=min_seeds, Years=Years)
    
    
#%%    
    

""" 1) Geometric operations """
anc_path = '/media/amin/STORAGE/STORAGE/OneDrive/PhD/Landsat_BA'

''' Siberia ESACCI'''
local_path = '/home/amin/firecci/0_Inputs/Amin/BAMT_GEE'
zone, lc, dataset, patches, threshold, min_seeds, ESACCI = 'Siberia', 'HRLC', 'Lndst', 'GeoTiff', 87, 4, True
# '''Siberia WGS'''
local_path = '/media/amin/STORAGE/STORAGE/OneDrive/PhD/Landsat_BA'
zone, lc, dataset, patches, threshold, min_seeds, ESACCI = 'Siberia', 'Land_Mask', 'Lndst', 'GeoTiff', 87, 4, False
zone, lc, dataset, patches, threshold, min_seeds, ESACCI = 'ABoVE', 'Land_Mask', 'Lndst', 'GeoTiff', 87, 4, False
'''Siberia UTM'''
local_path = '/home/amin/Desktop/Amin/PhD/Landsat_BA'
zone, lc, dataset, patches, threshold, min_seeds, ESACCI = 'Siberia', 'Land_Mask', 'Lndst', 'GeoTiff', 87, 4, False
zone, lc, dataset, patches, threshold, min_seeds, ESACCI = 'ABoVE', 'Land_Mask', 'Lndst', 'GeoTiff', 87, 4, False

process_prob = True
Years = np.arange(2023, 1988, -1)
if zone == 'Siberia' or zone == 'ABoVE':
    if ESACCI:
        Periods = [f'{year}0301-{year}1201' for year in Years[4:-1]]
    else:
        Periods = [f'{year}0301-{year}1201' for year in Years[:24]]


Tiles = os.listdir(f'{local_path}/{zone}/ByTile')
os.makedirs(f'{local_path}/{zone}/Logs', exist_ok=True)
LogFile = open(f'{local_path}/{zone}/Logs/Logfile_Geometric_Operations_\
{Periods[-1][:8]}-{Periods[0][9:]}_{zone}_{dataset}.txt', 'w')
# LogFile = io.StringIO()
Global_Functions.print_and_log(LogFile, 
            f'{30 * "-"} LogFile of Geometric operations of {zone}-{dataset}, \
processing probability = {process_prob}, LC = {lc} {30 * "-"}\n\n')

time0 = datetime.now()
for p in list(reversed(sorted(Periods)))[:]:
    inputs = glob.glob(f'{local_path}/{zone}/ByTile/**/GEE/PROB/*{p}*.tif', recursive=True)
    Tiles = [f.split('_')[-2] for f in inputs]
    if len(Tiles) > 0:
        Global_Functions.print_and_log(LogFile, 
                f'\n{10 * "#"} Processing of period {p}: {len(Tiles)} tiles {10 * "#"}\n\n')
    
        kwargs = dict(zone=zone, dataset=dataset, 
                      date_pre=p.split('-')[0], date_post=p.split('-')[1],
                      logfile=None, overwrite=True, kernel=(3, 3), res=0.00025, 
                      patches=patches, process_prob=process_prob, lc=lc)
        kwargs_FilterBA = dict(lower_threshold=threshold, min_seeds=min_seeds)
    
        time1 = datetime.now()
        listdict = Parallel(n_jobs=6, verbose=100, backend='threading') (delayed(getMonthlyJD) (tile, 
                        local_path, local_path, anc_path, kwargs=kwargs_FilterBA, **kwargs) for tile in Tiles[:]) 
        listdict.sort(key=operator.itemgetter('tile'))
        [LogFile.write(listdict[i]['text']) for i in range(len(listdict))]
        time2 = datetime.now()
        delta = time2 - time1
        hours, minutes, seconds = Global_Functions.convert_timedelta(delta)
        Global_Functions.print_and_log(LogFile, '\n%s Period %s is processed in %s hours, %s minutes, %s seconds %s '
                                  %(10 *'#', p, hours, minutes, seconds, 10 * '#'))    
timef = datetime.now()
delta = timef - time0    
hours, minutes, seconds = Global_Functions.convert_timedelta(delta)
Global_Functions.print_and_log(LogFile, '\n\n%s FINISHED in %s hours, %s minutes, %s seconds %s'
                          %(20 *'-', hours, minutes, seconds, 20 * '-'))    
LogFile.close()




#%%


""" 2) Yearly Aggregation """

# '''Siberia WGS'''
# local_path = '/media/amin/STORAGE/STORAGE/OneDrive/PhD/Landsat_BA'
'''Siberia UTM'''
local_path = '/home/amin/Desktop/Amin/PhD/Landsat_BA'
zone, dataset = 'ABoVE', 'Lndst'
kwargs = dict(zone=zone, dataset=dataset,
              inputs='seasonal', save=True)
Years = range(2023, 2000, -1)
Tiles = os.listdir(f'{local_path}/{zone}/ByTile')
# Years = [f'{year}0401-{year+1}0401' for year in Years]
LogFile = open(f'{local_path}/{zone}/Logs/Logfile_Yearly_Aggregation_{Years[-1]}-{Years[0]}_{zone}_{dataset}.txt', 'w')
LogFile.write(f'{30 * "-"} LogFile of Yearly Aggregation {Years[-1]}-{Years[0]} for {zone}-{dataset} {30 * "-"}\n\n')
time0 = datetime.now()
# Parallel(n_jobs=10, verbose=100) (delayed(aggreagate_year) (year, local_path, 
#                                    saveTiles=True, saveAgg=False, **kwargs) for year in Years)
for year in Years[:]:
    inputs = glob.glob(f'{local_path}/{zone}/ByTile/**/GEE/PROB/*{year}1201*.tif', recursive=True)
    Tiles = [f.split('_')[-2] for f in inputs]
    if len(Tiles) > 0:
        Global_Functions.print_and_log(LogFile, 
                f'{10 * "#"} Processing of Year: {year} {10 * "#"}\n\n')    
        time1 = datetime.now()
        listdict = []
        listdict = Parallel(n_jobs=10, verbose=300, backend='threading') (delayed(aggregate_tile) (tile, year, 
                                            local_path, **kwargs) for tile in Tiles)
        listdict.sort(key=operator.itemgetter('tile'))
        [LogFile.write(listdict[i]['text']) for i in range(len(listdict))]  
        time2 = datetime.now()
        delta = time2 - time1
        hours, minutes, seconds = Global_Functions.convert_timedelta(delta)
        Global_Functions.print_and_log(LogFile, '\n%s Period %s is processed in %s hours, %s minutes, %s seconds %s '
                                  %(10 *'#', year, hours, minutes, seconds, 10 * '#'))    
timef = datetime.now()
delta = timef - time0    
hours, minutes, seconds = Global_Functions.convert_timedelta(delta)
Global_Functions.print_and_log(LogFile, '\n\n%s FINISHED in %s hours, %s minutes, %s seconds %s'
                          %(20 *'-', hours, minutes, seconds, 20 * '-'))    
LogFile.close()


#%%

""" 3) Voronoi generation """

zone = 'ABoVE'; dataset='Lndst'
anc_path = f'/media/amin/STORAGE/STORAGE/OneDrive/PhD/Landsat_BA/{zone}'
# '''Siberia WGS'''
# local_path = f'/media/amin/STORAGE/STORAGE/OneDrive/PhD/Landsat_BA/{zone}'
## You should not use WGS
'''Siberia UTM'''
local_path = f'/home/amin/Desktop/Amin/PhD/Landsat_BA/{zone}'
kwargs = dict(zone=zone, dataset=dataset, date_field='ACQ_DATE', 
              buffer=5000, save_wgs=True, save_utm=False, logfile=False)
Years = range(2023, 2000, -1) ## in 2000 there no active fires
Tiles = next(os.walk(f'{local_path}/ByTile'))[1]
os.makedirs(f'{local_path}/Logs', exist_ok=True)
LogFile = open(f'{local_path}/Logs/Logfile_Hotspots_Voronoi_{zone}.txt', 'w')
# LogFile = io.StringIO()
Global_Functions.print_and_log(LogFile, 
            f'{30 * "-"} LogFile of hotspot Voronoi generation of {zone} {30 * "-"}\n\n')

time0 = datetime.now()
for year in list(Years)[:]:
    inputs = glob.glob(f'{local_path}/ByTile/**/GEE/PROB/*{year}1201*.tif', recursive=True)
    Tiles = [f.split('_')[-2] for f in inputs]
    if len(Tiles) > 0:
        Global_Functions.print_and_log(LogFile, 
                f'\n{10 * "#"} Processing of Year: {year} {10 * "#"}\n\n')
        time1 = datetime.now()
        listdict = []
        listdict = Parallel(n_jobs=16, verbose=100)(delayed(get_HS_voronoi)(local_path, anc_path, year, tile, **kwargs) 
                                    for tile in Tiles)
        listdict.sort(key=operator.itemgetter('tile'))
        [LogFile.write(listdict[i]['text']) for i in range(len(listdict))]
        time2 = datetime.now()
        delta = time2 - time1
        hours, minutes, seconds = Global_Functions.convert_timedelta(delta)
        Global_Functions.print_and_log(LogFile, '\n%s Year %s is processed in %s hours, %s minutes, %s seconds %s '
                                  %(10 *'#', year, hours, minutes, seconds, 10 * '#'))    
timef = datetime.now()
delta = timef - time0    
hours, minutes, seconds = Global_Functions.convert_timedelta(delta)
Global_Functions.print_and_log(LogFile, '\n\n%s FINISHED in %s hours, %s minutes, %s seconds %s'
                          %(20 *'-', hours, minutes, seconds, 20 * '-'))    
LogFile.close()

#%%


""" 4) Hotspot pixel generation """

zone = 'ABoVE'; dataset='Lndst'
anc_path = f'/media/amin/STORAGE/STORAGE/OneDrive/PhD/Landsat_BA/{zone}'
'''Siberia WGS'''
local_path = f'/media/amin/STORAGE/STORAGE/OneDrive/PhD/Landsat_BA/{zone}'
'''Siberia UTM'''
# local_path = f'/home/amin/Desktop/Amin/PhD/Landsat_BA/{zone}'

kwargs = dict(zone=zone, dataset=dataset, date_field='ACQ_DATE', 
              buffer=3000, edge=0, save_output=True, target_crs='WGS', logfile=False)
Years = range(2023, 2000, -1) ## in 2000 there no active fires
Tiles = next(os.walk(f'{local_path}/ByTile'))[1]
LogFile = open(f'{local_path}/Logs/Logfile_Hotspot_Boxes_{zone}.txt', 'w')
# LogFile = io.StringIO()
Global_Functions.print_and_log(LogFile, 
            f'{30 * "-"} LogFile of hotspot pixel generation of {zone} {30 * "-"}\n\n')

time0 = datetime.now()
for year in list(Years)[:]:
    inputs = glob.glob(f'{local_path}/ByTile/**/GEE/PROB/*{year}1201*.tif', recursive=True)
    Tiles = [f.split('_')[-2] for f in inputs]
    if len(Tiles) > 0:
        Global_Functions.print_and_log(LogFile, 
                f'\n{10 * "#"} Processing of Year: {year} {10 * "#"}\n\n')
        time1 = datetime.now()
        listdict = []
        listdict = Parallel(n_jobs=16, verbose=300, backend='threading')(delayed(get_HS_boxes)(local_path, anc_path, 
                                            year, tile, **kwargs) for tile in Tiles)
        listdict.sort(key=operator.itemgetter('tile'))
        [LogFile.write(listdict[i]['text']) for i in range(len(listdict))]
        time2 = datetime.now()
        delta = time2 - time1
        hours, minutes, seconds = Global_Functions.convert_timedelta(delta)
        Global_Functions.print_and_log(LogFile, '\n%s Year %s is processed in %s hours, %s minutes, %s seconds %s '
                                  %(10 *'#', year, hours, minutes, seconds, 10 * '#'))    
timef = datetime.now()
delta = timef - time0    
hours, minutes, seconds = Global_Functions.convert_timedelta(delta)
Global_Functions.print_and_log(LogFile, '\n\n%s FINISHED in %s hours, %s minutes, %s seconds %s'
                          %(20 *'-', hours, minutes, seconds, 20 * '-'))    
LogFile.close()


#%%

""" 5) BA enhancement """

zone, dataset = 'Siberia', 'Lndst'
ESACCI=False; UTM=False
local_path = f'/media/amin/STORAGE/STORAGE/OneDrive/PhD/Landsat_BA/{zone}'
add_boxes = True

local_path1 = f'/media/amin/STORAGE/STORAGE/OneDrive/PhD/Landsat_BA/{zone}'
water_path = '/media/amin/EXT2/Landsat_BA/Ancillary/Hansen_Water'
Years = range(2019, 2000, -1) ## in 2000 there no active fires
kwargs = dict(zone=zone, dataset=dataset, add_boxes=add_boxes, 
              res=0.00025, start_year=Years[-1], end_year=Years[0])
Tiles = next(os.walk(f'{local_path}/ByTile'))[1]
LogFile = open(f'{local_path}/Logs/Logfile_BA_Enhancement_{zone}.txt', 'w')
# LogFile = io.StringIO()
Global_Functions.print_and_log(LogFile, 
            f'{30 * "-"} LogFile of burned area enhancement of {zone} {30 * "-"}\n\n')

time0 = datetime.now()
for year in sorted(list(Years))[:]:
    if zone == 'ABoVE':
        inputs = glob.glob(f'{local_path}/ByTile/**/GEE/PROB/*{year}1201*.tif', recursive=True)
        Tiles = [f.split('_')[-2] for f in inputs]
        if len(Tiles) == 0:
            continue
        else:
            listdict = []
            for tile in Tiles[:]:
                if os.path.isfile(f'{local_path}/ByTile/{tile}/Yearly/BAMT_BA_{zone}_{dataset}_{year-1}_{tile}_JD.tif'):
                    kwargs['start_year'] = year-1
                else:
                    kwargs['start_year'] = year
                if os.path.isfile(f'{local_path}/ByTile/{tile}/Yearly/BAMT_BA_{zone}_{dataset}_{year+1}_{tile}_JD.tif'):
                    kwargs['end_year'] = year+1
                else:
                    kwargs['end_year'] = year   
                listdict.append(enhance_BA(local_path, local_path, water_path, year, tile, **kwargs))            
    else:
        Global_Functions.print_and_log(LogFile, 
                f'\n{10 * "#"} Processing of Year: {year} {10 * "#"}\n\n')
        time1 = datetime.now()
        listdict = []
        listdict = Parallel(n_jobs=10, verbose=300, backend='threading', max_nbytes=None)(delayed(
            enhance_BA)(local_path, local_path1, water_path, year, tile, **kwargs) for tile in Tiles)
        listdict.sort(key=operator.itemgetter('tile'))
        [LogFile.write(listdict[i]['text']) for i in range(len(listdict))]
        time2 = datetime.now()
        delta = time2 - time1
        hours, minutes, seconds = Global_Functions.convert_timedelta(delta)
        Global_Functions.print_and_log(LogFile, '\n%s Year %s is processed in %s hours, %s minutes, %s seconds %s '
                                  %(10 *'#', year, hours, minutes, seconds, 10 * '#'))    
timef = datetime.now()
delta = timef - time0    
hours, minutes, seconds = Global_Functions.convert_timedelta(delta)
Global_Functions.print_and_log(LogFile, '\n\n%s FINISHED in %s hours, %s minutes, %s seconds %s'
                          %(20 *'-', hours, minutes, seconds, 20 * '-'))    
LogFile.close()



#%%
""" 6) Patch generation """

zone, dataset = 'Siberia', 'Lndst'
local_path = f'/media/amin/STORAGE/STORAGE/OneDrive/PhD/Landsat_BA/{zone}'
LogFile = open(f'{local_path}/Logs/Logfile_Patch_Aggregation_{zone}_{dataset}.txt', 'w')
# LogFile = io.StringIO() 
Tiles = os.listdir(f'{local_path}/ByTile')
Years = range(2023, 2000, -1)
list_ds = []
# time.sleep(4*3600)
time0 = datetime.now()
kwargs = dict(zone=zone, dataset=dataset, kernel=(11, 11), cutoff=16, 
              process_peat=True, connectivity=8, res=0.00025)
LogFile.write(f'{30 * "-"} LogFile of PeatFire Patch aggregation of {zone} using {kwargs["cutoff"]} cutoff value {30 * "-"}\n\n')

for year in Years:
    if zone == 'ABoVE':
        inputs = glob.glob(f'{local_path}/ByTile/**/GEE/PROB/*{year}1201*.tif', recursive=True)
        Tiles = [f.split('_')[-2] for f in inputs]
        if len(Tiles) == 0:
            continue
    else:
        time1 = datetime.now()
        Global_Functions.print_and_log(LogFile, f'\n\n{10 * "#"} Processing of Year: {year} {10 * "#"}\n\n')
        listdict = Parallel(n_jobs=6, verbose=300, backend='threading') (delayed (retrieve_patches) (
                                        local_path, year, tile, **kwargs) for tile in Tiles[:])
        listdict.sort(key=operator.itemgetter('tile'))
        [LogFile.write(listdict[i]['text']) for i in range(len(listdict))]
        time2 = datetime.now()
        delta = time2 - time1
        hours, minutes, seconds = Global_Functions.convert_timedelta(delta)
        Global_Functions.print_and_log(LogFile, '\n\n%s Year %s is processed in %s hours, %s minutes, %s seconds %s '
                                  %(10 *'+', year, hours, minutes, seconds, 10 * '+'))    
timef = datetime.now()
delta = timef - time0    
hours, minutes, seconds = Global_Functions.convert_timedelta(delta)
Global_Functions.print_and_log(LogFile, '\n\n%s FINISHED in %s hours, %s minutes, %s seconds %s'
                          %(20 *'-', hours, minutes, seconds, 20 * '-'))    
LogFile.close()

#%%


""" 7) Grid generation """

zone, dataset = 'ABoVE', 'Lndst'
local_path = f'/media/amin/STORAGE/STORAGE/OneDrive/PhD/Landsat_BA/{zone}'
res = 0.25
kwargs = dict(zone=zone, 
              dataset=dataset, 
              res=res,
              process_pixel=False)
LogFile = open(f'{local_path}/Logs/Logfile_PeatFire_Aggregation_{zone}_{dataset}_{str(res).replace(".", "")}D.txt', 'w')
LogFile.write(f'{30 * "-"} LogFile of PeatFire aggregation from 30m to {res}° {30 * "-"}\n\n')
Tiles = os.listdir(f'{local_path}/ByTile')
Years = range(2023, 2000, -1)
list_ds = []
time0 = datetime.now()
for year in Years[:]:
    inputs = glob.glob(f'{local_path}/ByTile/**/GEE/PROB/*{year}1201*.tif', recursive=True)
    Tiles = [f.split('_')[-2] for f in inputs]
    if len(Tiles) > 0:
        Global_Functions.print_and_log(LogFile, 
                f'\n{10 * "#"} Processing of Year: {year} {10 * "#"}\n\n')
        time1 = datetime.now()
        Global_Functions.print_and_log(LogFile, f'\n\n{10 * "#"} Processing of Year: {year} {10 * "#"}\n\n')
        listdict = Parallel(n_jobs=18, verbose=100, backend='threading') (delayed (aggregate_peatBA) (tile, year, 
                                              local_path, **kwargs) for tile in Tiles[:])
        listdict.sort(key=operator.itemgetter('tile'))
        [LogFile.write(listdict[i]['text']) for i in range(len(listdict))]
        time2 = datetime.now()
        delta = time2 - time1
        hours, minutes, seconds = Global_Functions.convert_timedelta(delta)
        Global_Functions.print_and_log(LogFile, '\n\n%s Year %s is processed in %s hours, %s minutes, %s seconds %s '
                                  %(10 *'+', year, hours, minutes, seconds, 10 * '+'))    
timef = datetime.now()
delta = timef - time0    
hours, minutes, seconds = Global_Functions.convert_timedelta(delta)
Global_Functions.print_and_log(LogFile, '\n\n%s FINISHED in %s hours, %s minutes, %s seconds %s'
                          %(20 *'-', hours, minutes, seconds, 20 * '-'))    
LogFile.close()

#%%


""" 8) FRP Voronoi generation """

zone = 'ABoVE'; dataset='Lndst'; layername = 'FRP'
anc_path = f'/media/amin/STORAGE/STORAGE/OneDrive/PhD/Landsat_BA/{zone}'
# '''Siberia WGS'''
# local_path = f'/media/amin/STORAGE/STORAGE/OneDrive/PhD/Landsat_BA/{zone}'
## You should not use WGS
'''Siberia UTM'''
local_path = f'/home/amin/Desktop/Amin/PhD/Landsat_BA/{zone}'
kwargs = dict(zone=zone, dataset=dataset, fieldname='FRP_km2', layername=layername,
              buffer=5000, save_wgs=True, save_utm=False, logfile=False)
Years = range(2023, 2000, -1) ## in 2000 there no active fires
Tiles = next(os.walk(f'{local_path}/ByTile'))[1]
os.makedirs(f'{local_path}/Logs', exist_ok=True)
LogFile = open(f'{local_path}/Logs/Logfile_{layername}_Hotspots_Voronoi_{zone}.txt', 'w')
# LogFile = io.StringIO()
Global_Functions.print_and_log(LogFile, 
            f'{30 * "-"} LogFile of {layername} Voronoi generation of {zone} {30 * "-"}\n\n')

time0 = datetime.now()
for year in list(Years)[:]:
    inputs = glob.glob(f'{local_path}/ByTile/**/GEE/PROB/*{year}1201*.tif', recursive=True)
    Tiles = [f.split('_')[-2] for f in inputs]
    if len(Tiles) > 0:
        Global_Functions.print_and_log(LogFile, 
                f'\n{10 * "#"} Processing of Year: {year} {10 * "#"}\n\n')
        time1 = datetime.now()
        listdict = []
        listdict = Parallel(n_jobs=16, verbose=100)(delayed(get_FRP) 
                    (local_path, anc_path, year, tile, **kwargs) for tile in Tiles)
        listdict.sort(key=operator.itemgetter('tile'))
        [LogFile.write(listdict[i]['text']) for i in range(len(listdict))]
        time2 = datetime.now()
        delta = time2 - time1
        hours, minutes, seconds = Global_Functions.convert_timedelta(delta)
        Global_Functions.print_and_log(LogFile, '\n%s Year %s is processed in %s hours, %s minutes, %s seconds %s '
                                  %(10 *'#', year, hours, minutes, seconds, 10 * '#'))    
timef = datetime.now()
delta = timef - time0    
hours, minutes, seconds = Global_Functions.convert_timedelta(delta)
Global_Functions.print_and_log(LogFile, '\n\n%s FINISHED in %s hours, %s minutes, %s seconds %s'
                          %(20 *'-', hours, minutes, seconds, 20 * '-'))    
LogFile.close()
