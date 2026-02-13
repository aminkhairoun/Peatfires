#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 10:01:22 2023

@author: Amin Khairoun
"""

import os
import numpy as np
import glob
import pandas as pd
from osgeo import gdal, ogr, osr
from numpy import matlib
import geopandas as gpd
import math
import time
from datetime import datetime, time, timedelta
import netCDF4
import rasterio as rio
import xarray as xr
import rioxarray as rxr
import tarfile 
import gzip
import sys
import io


def write_rasterio(Array, OutPath, refPath=None, profile=None, custom_dtype=False, dtype='uint8', nodata=0):
    if refPath:
        with rio.open(refPath) as ref:
            profile = ref.profile
    # try:
    profile['compress'] = 'lzw'
    if custom_dtype:
        profile['dtype'] = dtype
        profile['nodata'] = nodata
    with rio.open(OutPath, 'w', **profile) as dst:
        dst.write(Array, 1)
    # except:
    #     if not (refPath or profile):
    #         print('Not saved: provide a profile or refPath')
    #     else:
    #         print('Not saved: check errors')

def get_pixel_area(ref, path=True, rio_array=False, ncdf=False, profile=False, area_1d=False):
    ## Path == True for geotiff file, rio_array for rasterio image, NetCDF file, otherwise xarray reader
    ## rio transform: [xres, xskew, xmin, yskew, yres, ymax]
    ## gdal transform: [xmin, xres, xskew, ymax, yskew, yres]
    ## netcdf i2m: [xres, xskew, yskew, yres, xmin, ymax]
    if path == True:
        with rio.open(ref) as dst:
            gt = dst.transform
            pix_width = gt[0]
            ulX = gt[2]
            ulY = gt[5]
            rows = dst.height
            cols = dst.width
            lrX = ulX + gt[0] * cols
            lrY = ulY + gt[4] * rows

    elif rio_array == True:
        gt = ref.transform
        pix_width = gt[0]
        ulX = gt[2]
        ulY = gt[5]
        rows = ref.height
        cols = ref.width
        lrX = ulX + gt[0] * cols
        lrY = ulY + gt[4] * rows

    elif ncdf == True:
        with netCDF4.Dataset(ref, 'r') as nc:
            geomatrixText = nc['crs'].i2m
            splittedText = geomatrixText.split(',')
            gt = [0, 0, 0, 0, 0, 0]
            gt[0] = float(splittedText[0])
            gt[1] = float(splittedText[1])
            gt[3] = float(splittedText[2])
            gt[4] = float(splittedText[3])
            gt[2] = float(splittedText[4])
            gt[5] = float(splittedText[5])
            pix_width = gt[0]
            ulX = gt[2]
            ulY = gt[5]
            rows = nc['lat'][:].shape[0]
            cols = nc['lon'][:].shape[0]
            lrX = ulX + gt[0] * cols
            lrY = ulY + gt[4] * rows
    
    elif profile == True:
        gt = ref['transform']
        pix_width = gt[0]
        ulX = gt[2]
        ulY = gt[5]
        rows = ref['height']
        cols = ref['width']
        lrX = ulX + gt[0] * cols
        lrY = ulY + gt[4] * rows
        
    else:
        gt = ref.rio.transform()
        pix_width = gt[0]
        ulX = gt[2]
        ulY = gt[5]
        rows = ref.rio.height
        cols = ref.rio.width
        lrX = ulX + gt[0] * cols
        lrY = ulY + gt[4] * rows

    lats = np.linspace(ulY, lrY, rows + 1)

    a = 6378137
    b = 6356752.3142

    # Degrees to radians
    lats = lats * np.pi / 180

    # Intermediate vars
    e = np.sqrt(1 - (b / a) ** 2)
    sinlats = np.sin(lats)
    zm = 1 - e * sinlats
    zp = 1 + e * sinlats

    # Distance between meridians
    q = pix_width / 360

    # Compute areas for each latitude in square m
    areas_to_equator = np.pi * b ** 2 * ((2 * np.arctanh(e * sinlats) / (2 * e) + sinlats / (zp * zm)))
    areas_between_lats = np.diff(areas_to_equator)
    areas_cells = np.abs(areas_between_lats) * q
    if area_1d:
        areagrid = areas_cells
    else:
        areagrid = np.transpose(matlib.repmat(areas_cells, cols, 1))
    return areagrid

def align_coords(ref, aligned):
    y = ref.coords['y']
    x = ref.coords['x']
    aligned = aligned.assign_coords({'y': y})
    aligned = aligned.assign_coords({'x': x})
    return aligned

def resh_to_grid(arr, scale):
    t = arr.reshape(arr.shape[0 ]//scale, scale, arr.shape[1 ]//scale, scale)
    data = t.transpose(0, 2, 1, 3).reshape(t.shape[0] * t.shape[2], t.shape[1] * t.shape[3])
    return data

def resample_by_ref(srcPath, dstPath, refPath, method='near'):
    with rio.open(srcPath) as src:
        srcSRS = src.crs
    with rio.open(refPath) as dst:
        dstSRS = dst.crs
        bounds = dst.bounds
        height = dst.height
        width = dst.width

    gdal.Warp(dstPath, srcPath, resampleAlg=method, width=width, height=height,
              srcSRS=srcSRS, dstSRS=dstSRS, outputBounds=bounds,
              creationOptions = ['COMPRESS=LZW', 'TILED=YES'])

def mosaic_gdal(naming, outPath, pattern=None, inputPath=None, vrtPath='/vsimem', files=None, refPath=None, 
                profile=None, srcNodata=None, dstNodata=None, resampleAlg='near', crs=None, xRes=None, yRes=None,
                n_threads=4, bounds=None, cutline=None):
    if files:
        ls = files
    else:
        ls = []
        for f in sorted(glob.glob(f'{inputPath}/**/*.tif', recursive=True)):
            if pattern.search(f):
                ls.append(f)
    print(len(ls), ls[0])
    vrt = gdal.BuildVRT(f'{vrtPath}/{naming}.vrt', ls, srcNodata=srcNodata)
    vrt.FlushCache()
    if refPath:
        with rio.open(refPath) as ref:
            profile = ref.profile
    if profile:       
        left = profile['transform'][2]
        bottom = profile['transform'][5] + profile['transform'][4] * profile['height']
        right = profile['transform'][2] + profile['transform'][0] * profile['width']
        top = profile['transform'][5]
        bounds = [left, bottom, right, top]
        xRes = profile['transform'][0]
        yRes = profile['transform'][4]
        crs = profile['crs'].to_wkt() 
    
    if not yRes:
        src = rio.open(ls[0])
        profile = src.profile
        xRes = profile['transform'][0]
        yRes = profile['transform'][4]          
        print("No Resolution was given, the the 1st list file is used")
    if not dstNodata:
        src = rio.open(ls[0])
        profile = src.profile
        dstNodata = profile['nodata']     
        print("No dstNodata was given, the the 1st list file is used")
    if not crs:
        src = rio.open(ls[0])
        profile = src.profile
        crs = profile['crs'].to_wkt()         
        print("No CRS was given, the the 1st list file is used")
    elif not isinstance(crs, str):
        crs = crs.ExportToWkt()    
    if cutline:
        crop = True
    else:
        crop = False
    gdal.SetConfigOption('GDAL_NUM_THREADS', str(n_threads))  
    os.makedirs(outPath, exist_ok=True)      
    gdal.Warp(f'{outPath}/{naming}.tif', f'{vrtPath}/{naming}.vrt', dstSRS=crs,
          resampleAlg=resampleAlg, outputBounds=bounds, xRes=xRes, yRes=yRes, 
          srcNodata=srcNodata, dstNodata=dstNodata, 
          cutlineDSName=cutline, cropToCutline=crop,
          multithread=True, creationOptions=['COMPRESS=LZW', 'TILED=YES'])

def convert_timedelta(duration):
    seconds = duration.total_seconds()
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return hours, minutes, seconds

def print_and_log(logFile, text):
    print(text)
    write_log(logFile, text)
    

def write_log(logFile, text):
    logFile.write('%s\n' %(text))
    logFile.flush()
    
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    distance = c * r
    return distance

def create_buffer(lon1, lat1, dist):
    """
    Calculate new lon, lat from great circle distance in kilometers 
    on the earth (specified in decimal degrees)
    """
    r = 6371
    dlon = dist / (r * math.cos(lat1 * math.pi / 180)) * 180 /  math.pi  
    dlat = dist / r * 180 /  math.pi
    lon2 = lon1 + dlon
    lat2 = lat1 + dlat
    return lon2, lat2

def replace_numpy_list(a, val_old, val_new):
    values = np.unique(a)
    val_old_orig = val_old.copy()
    val_old = np.array([i for i in val_old if i in values])
    idx = [np.argwhere(val_old_orig == i)[0][0] for i in val_old]
    val_new = val_new[idx]
    offset = max(-a.min(), 0) 
    arr = np.empty(int(a.max() + 1 + 2 * offset), dtype=val_new.dtype)
    arr[(val_old + offset).astype(np.int32)] = val_new
    out = arr[(a + offset).astype(np.int32)]
    np.place(out, ~np.isin(a, val_old), a[~np.isin(a, val_old)])
    return out

def append_to_targz(gzfile, files_to_add, out_path):
    # Decompress the existing archive
    with tarfile.open(gzfile, mode='r:gz') as tar:
        contents = {name: tar.extractfile(name).read() for name in tar.getnames()}
    for f in files_to_add:
        with open(f, 'rb') as file:
            file_content = file.read()
            filename = os.path.join(os.path.dirname(list(contents.keys())[0]), os.path.basename(f))  
            contents[filename] = file_content
            
    os.makedirs(out_path, exist_ok=True)
    if os.path.isfile(gzfile.replace(os.path.dirname(gzfile), out_path)):
        print('gz file exists')
        sys.exit()
    with gzip.open(gzfile.replace(os.path.dirname(gzfile), out_path), 'wb') as f_out:
        with tarfile.open(fileobj=f_out, mode='w') as tar:
            for i in contents.keys():
                file_io = io.BytesIO(contents[i])
                tarinfo = tarfile.TarInfo(name=i)
                tarinfo.size = len(contents[i])
                tar.addfile(tarinfo, file_io)



