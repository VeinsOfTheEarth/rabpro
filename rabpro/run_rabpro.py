# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 21:26:14 2020

@author: Jon

env: rabpro

This is a wrapper script for using RaBPro via command line.
"""

""" For Darwin only """
import argparse
import rabpro

def main(coords_file, da, name, years, force_merit=False, verbose=True):
    
       
    if len(years) == 0 or years[0] == 'a':
        years = 'all'
        
    if da == -1:
        da = None
                        
    rpo = rabpro.profiler(coords_file, da=da, name=name, force_merit=force_merit, verbose=verbose)
    rpo.delineate_basins()
    rpo.basins.to_file('/projects/rabpro/gaugebasin_shp/'+name+'.shp',driver='ESRI Shapefile')
    rpo.elev_profile()
    #rpo.profile.to_file('/projects/rabpro/eleprofile_merit/'+name+'.shp',driver='ESRI Shapefile')
    rpo.basin_stats(years=years)
    rpo.export()


if __name__ == '__main__':
    
    parser=argparse.ArgumentParser()
                
    parser.add_argument(
        '--metadata_file',
        nargs = 1,
        default = '',
        type = str)

    parser.add_argument(
        '--da',
        nargs = 1,
        default = [-1.],
        type = float)
    
    parser.add_argument(
        '--name',
        nargs = 1,
        default = 'unnamed',
        type = str)
    
    parser.add_argument(
        '--years',
        nargs = 1,
        default = ['all'],
        type = list)
    
    parser.add_argument(
        '--force_merit',
        nargs = 1,
        default = [False],
        type = bool)
    
    parser.add_argument(
        '--verbose',
        nargs = 1,
        default = [True],
        type = bool)
                
    args = vars(parser.parse_args())
    
    print(args)
    
    main(args['coords_file'][0], args['da'][0], args['name'][0], args['years'][0], args['force_merit'][0], args['verbose'][0])
