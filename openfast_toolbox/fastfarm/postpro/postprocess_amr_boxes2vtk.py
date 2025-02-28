#!/usr/bin/env python

#SBATCH --job-name=pp_vtk
#SBATCH --output amr2post_vtk.log.%j
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --account=<account>
#SBATCH --mem=160G
# #SBATCH --qos=high

# ----------------------------------------------------------------------- #
# postprocess_boxes2vtk.py                                                #
#                                                                         #
# Post process AMR-Wind output boxes for FAST.Farm and save vtk files for #
# each group (each individual sampling box). This script should be called #
# once for each output netcdf file and each group. If group is not given, #
# it expects only one (e.g. low box); otherwise, the script will stop and #
# warn the user. The output vtk is saved in $case/processedData/<group>.  #
#                                                                         #
# Usage:                                                                  #
# postprocess_boxes2vtk.py -p <fullpath> -f <ncfile> [-g <group>]         #
#                                                                         #
# Example usage:                                                          #
#    cd /full/path/to/amrwind/case                                        #
#    path=$(pwd -P)                                                       #
#    cd post_processing                                                   #
#    ls  # see what sampling box files are available.                     #
#    sbatch postprocess_boxes2vtk.py [-J <jobname>] -p $path              #
#                                   -f lowres40000.nc -g Low              #
#    sbatch postprocess_boxes2vtk.py [-J <jobname>] -p $path              #
#                    -f highres40000.nc -g HighT1_inflow0deg              #
#                                                                         #
# Regis Thedin                                                            #
# Mar 13, 2023                                                            #
# regis.thedin@nrel.gov                                                   #
# ----------------------------------------------------------------------- #

import argparse
import numpy as np
import os, time
from itertools import repeat
import multiprocessing
from windtools.amrwind.post_processing  import StructuredSampling

def main(samplingboxfile, pppath, pptag, requestedgroup, outpath, dt, t0, itime, ftime, steptime, offsetz, vtkstartind, terrain, ncores):

    if ncores is None:
        ncores = multiprocessing.cpu_count()

    # -------- CONFIGURE RUN
    s = StructuredSampling(pppath)

    outpathgroup = os.path.join(outpath, group)
    if not os.path.exists(outpathgroup):
        # Due to potential race condition, adding exist_ok flag as well
        os.makedirs(outpathgroup, exist_ok=True)

    # Redefine some variables as the user might call just this method
    s.pptag = pptag
    s.get_all_times_native()
    if ftime == -1:
        ftime = s.all_times[-1]
    available_time_indexes = [n for n in s.all_times if itime <= n <= ftime]

    ## Split all the time steps in arrays of roughly the same size
    chunks =  np.array_split(available_time_indexes, ncores)

    # Get rid of the empty chunks (happens when the number of boxes is lower than 96)
    chunks = [c for c in chunks if c.size > 0]

    # Now, get the beginning and end of each separate chunk
    itime_list = [i[0]    for i in chunks]
    ftime_list = [i[-1]+1 for i in chunks] 
    print(f'itime_list is {itime_list}')
    print(f'ftime_list is {ftime_list}')

    p = multiprocessing.Pool()
    ds_ = p.starmap(s.to_vtk, zip(repeat(group),          # group
                                  repeat(outpathgroup),   # outputPath
                                  repeat(samplingboxfile),# file
                                  repeat(pptag),          # pptag
                                  repeat(True),           # verbose
                                  repeat(offsetz),        # offset in z
                                  itime_list,             # itime
                                  ftime_list,             # ftime
                                  repeat(t0),             # t0
                                  repeat(dt),             # dt
                                  repeat(vtkstartind),    # vtkstartind
                                  repeat(terrain)         # terrain
                                 )
                              )
    print('Finished.')
    

if __name__ == '__main__':

    # ------------------------------------------------------------------------------
    # -------------------------------- PARSING INPUTS ------------------------------
    # ------------------------------------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument("--path", "-p",   type=str, default=os.getcwd(),
                        help="case full path (default cwd)")
    parser.add_argument("--ncfile", "-f", type=str, default=None,
                        help="netcdf sampling planes")
    parser.add_argument("--dt", "-dt", default=None,
                        help="time step for naming the boxes output")
    parser.add_argument("--initialtime", "-t0", default=None,
                        help="Time step related to first box output (optional)")
    parser.add_argument("--group", "-g",  type=str, default=None,
                        help="group within netcdf file to be read, if more than one is available")
    parser.add_argument("--itime", "-itime",  type=int, default=0,
                        help="sampling time step to start saving the data")
    parser.add_argument("--ftime", "-ftime",  type=int, default=-1,
                        help="sampling time step to end saving the data")
    parser.add_argument("--steptime", "-step",  type=int, default=1,
                        help="sampling time step increment to save the data")
    parser.add_argument("--offsetz", "-offsetz",  type=float, default=None,
                        help="Offset in the x direction, ensuring a point at hub height")
    parser.add_argument("--vtkstartind", "-vtkstartind", default=0,
                        help="Index by which the names of the vtk files will be shifted")
    parser.add_argument("--pptag", "-tag", type=str, default=None,
                        help="Post-processing tag (e.g. 'box_hr')")
    parser.add_argument("--terrain", "-terrain", type=bool, default=False,
                        help="Whether or not to add NaN for inside-terrain domain")
    parser.add_argument("--ncores", "-ncores", type=int, default=None,
                        help="Number of cores to be used. Default is machine's total")


    args = parser.parse_args()

    # Parse inputs
    path        = args.path
    ncfile      = args.ncfile
    dt          = args.dt
    t0          = args.initialtime
    group       = args.group
    itime       = args.itime
    ftime       = args.ftime
    steptime    = args.steptime
    offsetz     = args.offsetz
    vtkstartind = args.vtkstartind
    pptag       = args.pptag
    terrain     = args.terrain
    ncores      = args.ncores

    # ------------------------------------------------------------------------------
    # --------------------------- DONE PARSING INPUTS ------------------------------
    # ------------------------------------------------------------------------------

    if path == '.':
        path = os.getcwd()

    # We assume the case path was given, but maybe it was $case/post* or $case/proc*
    # Let's check for that and fix it
    if os.path.basename(path) == 'post_processing' or os.path.basename(path) == 'processedData':
        path = os.path.split(path)[0]

    pppath = os.path.join(path,'post_processing')
    outpath = os.path.join(path,'processedData')

    # ------- PERFORM CHECKS
    if not os.path.exists(path):
        parser.error(f"Path {path} does not exist.")

    if not os.path.exists(pppath):
        raise ValueError (f'Path {pppath} does not exist.')

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    if ncfile is not None:
        if isinstance(ncfile,str):
            if not ncfile.endswith('.nc'):
                raise ValueError (f"Received single string as ncfiles, but it does not end with .nc. Received {ncfile}")
            if not os.path.isfile(os.path.join(pppath,ncfile)):
                raise ValueError(f"File {ncfile} does not exist.")
        else:
            raise ValueError(f"the ncfile should be a string. Received {ncfile}.")

    if isinstance(dt,str):
        try:    dt=float(dt)
        except: pass
    if dt == 'None': dt=None
    if dt is not None and not isinstance(dt,(float,int)):
        raise ValueError(f'dt should be a scalar. Received {dt}.')

    if isinstance(t0,str):
        try:    t0=float(t0)
        except: pass
    if t0 == 'None': t0=None
    if t0 is not None and not isinstance(t0,(float,int)):
        raise ValueError(f't0 should be a scalar. Received {t0}, type {type(t0)}.')

    if steptime < 1:
        raise ValueError(f'The time step increment should be >= 1.')

    if itime < 0:
        raise ValueError(f'The initial time step should be >= 0.')

    if ftime != -1:
        if ftime < itime:
            raise ValueError(f'The final time step should be larger than the'\
                             f'initial. Received itime={itime} and ftime={ftime}.')

    if offsetz is None:
        print(f'!!! WARNING: no offset in z has been given. Ensure you will have a point at hub height.')
        offsetz=0

    if isinstance(vtkstartind,str):
        try:    vtkstartind=int(vtkstartind)
        except: pass
    if vtkstartind == 'None': vtkstartind=None
    if vtkstartind is not None and not isinstance(vtkstartind,int):
        raise ValueError(f'vtkstartind should be either None or an integer. Received {vtkstartind}.')

    # ------------------------------------------------------------------------------
    # ---------------------------- DONE WITH CHECKS --------------------------------
    # ------------------------------------------------------------------------------

    if ncfile:
        print(f'Reading {path}/{ncfile}')
    else:
        print(f'Reading {pppath}/{group}*')

    print(f'Starting job at {time.ctime()}')
    multiprocessing.freeze_support()
    main(ncfile, pppath, pptag, group, outpath, dt, t0, itime, ftime, steptime, offsetz, vtkstartind, terrain, ncores)
    print(f'Ending job at   {time.ctime()}')

