import numpy as np
import pandas as pd
import xarray as xr
import os, sys, re
from multiprocessing import Pool
from itertools import repeat

from openfast_toolbox.io import TurbSimFile, FASTOutputFile, VTKFile, FASTInputFile

def _get_fstf_filename(caseobj):
    if hasattr(caseobj, 'outputFFfilename'):
        return os.path.splitext(caseobj.outputFFfilename)[0]
    else:
        return 'FFarm_mod'

def _getCasePath(caseobj, cond, case):
    try:
        return caseobj.getCasePath(cond, case)
    except:
        print('>>> get CasePath failed')
        return os.path.join(caseobj.path, caseobj.condDirList[cond], caseobj.caseDirList[case]) 

def _getCaseSeedPath(caseobj, cond, case, seed):
    try:
        return caseobj.getCaseSeedPath(cond, case, seed)
    except:
        print('>>> get CaseSeedPath failed')
        return os.path.join(caseobj.path, caseobj.condDirList[cond], caseobj.caseDirList[case], f'Seed_{seed}')

def _getDT_Low(ff_file):
    '''
    Low-resolution (farm) time step actually used by FAST.Farm. The input file has one DT_Low entry per ambient-wind model and
    Mod_AmbWind selects it: 1 -> DT_Low-VTK, 2 and 3 -> DT_Low, 4 -> DT_Low-AMReX (older input files only have the first two).
    '''

    if ff_file['Mod_AmbWind'] == 1:
        return ff_file['DT_Low-VTK']
    elif ff_file['Mod_AmbWind'] == 4:
        return ff_file['DT_Low-AMReX']
    else:
        return ff_file['DT_Low']

def readTurbineOutputPar(caseobj, dt_openfast, dt_processing, saveOutput=True, output='nc',
                         iCondition=0, fCondition=-1, iCase=0, fCase=-1, iSeed=0, fSeed=-1, iTurbine=0, fTurbine=-1,
                         nCores=36):
    '''
    Inputs
    ------
    output: str
        Either 'nc' or 'zarr'. Determines the output format
    Zero-indexed initial and final values for conditions, cases, seeds, and turbines. 


    '''

    if fCondition==-1:
        fCondition = caseobj.nConditions
    if fCondition-iCondition <= 0:
        raise ValueError (f'Final condition to read needs to be larger than initial.')

    if fCase==-1:
        fCase = caseobj.nCases
    if fCase-iCase <= 0:
        raise ValueError (f'Final case to read needs to be larger than initial.')

    if fSeed==-1:
        fSeed = caseobj.nSeeds
    if fSeed-iSeed <= 0:
        raise ValueError (f'Final seed to read needs to be larger than initial.')

    if fTurbine==-1:
        fTurbine = caseobj.nTurbines
    if fTurbine-iTurbine <= 0:
        raise ValueError (f'Final turbine to read needs to be larger than initial.')

    if fCase-iCase < nCores:
        print(f'Total number of cases requested ({fCase-iCase}) is lower than number of cores {nCores}.')
        print(f'Changing the number of cores to {fCase-iCase}.')
        nCores = fCase-iCase

    if output not in ['zarr','nc']:
        raise ValueError (f'Output can only be zarr or nc')


    outfilename = f'ds_turbineOutput_temp_cond{iCondition}_{fCondition}_case{iCase}_{fCase}_seed{iSeed}_{fSeed}_turb{iTurbine}_{fTurbine}_dt{dt_processing}s'
    zarrstore = f'{outfilename}.zarr'
    ncfile = f'{outfilename}.nc'
    outputzarr = os.path.join(caseobj.path, zarrstore)
    outputnc   = os.path.join(caseobj.path, ncfile)

    if output=='zarr' and os.path.isdir(outputzarr):
       print(f'Output file {zarrstore} exists. Loading it.')
       comb_ds = xr.open_zarr(outputzarr)
       return comb_ds
    if output=='nc' and os.path.isfile(outputnc):
       print(f'Output file {ncfile} exists. Loading it.')
       comb_ds = xr.open_dataset(outputnc)
       return comb_ds
        

    print(f'Running readTurbineOutput in parallel using {nCores} workers')

    # Split all the cases in arrays of roughly the same size
    chunks =  np.array_split(range(iCase,fCase), nCores)
    # Now, get the beginning and end of each separate chunk
    iCase_list = [i[0]    for i in chunks]
    fCase_list = [i[-1]+1 for i in chunks] 
    print(f'iCase_list is {iCase_list}')
    print(f'fCase_list is {fCase_list}')

    p = Pool()
    ds_ = p.starmap(readTurbineOutput, zip(repeat(caseobj),          # caseobj
                                           repeat(dt_openfast),      # dt_openfast
                                           repeat(dt_processing),    # dt_processing
                                           repeat(False),            # saveOutput
                                           repeat(output),           # output
                                           repeat(iCondition),       # iCondition
                                           repeat(fCondition),       # fCondition
                                           iCase_list,               # iCase
                                           fCase_list,               # fCase
                                           repeat(iSeed),            # iSeed
                                           repeat(fSeed),            # fSeed
                                           repeat(iTurbine),         # iTurbine
                                           repeat(fTurbine),         # fTurbine
                                          )
                                       )
    p.close()
    p.terminate()
    p.join()

    print(f'Done reading all output. Concatenating the arrays')
  
    try:
        comb_ds = xr.combine_by_coords(ds_)
    except ValueError as e:
        if str(e) == "Coordinate variable case is neither monotonically increasing nor monotonically decreasing on all datasets":
            print('Concatenation using combine_by_coords failed. Concatenating using merge instead.')
            print('  WARNING: Indexes are _not_ monotonically increasing. Do not use `isel`.')
            print('           Try using `.sortby(<dimstr>)` to sort it.')
            comb_ds = xr.merge(ds_)
        else:
            raise

    if saveOutput:
        print(f'Done concatenating. Saving {output} file.')
        if output == 'zarr':  comb_ds.to_zarr(outputzarr)
        elif output == 'nc':  comb_ds.to_netcdf(outputnc)

    print('Finished.')

    return comb_ds


def readTurbineOutput(caseobj, dt_openfast, dt_processing=1, saveOutput=True, output='nc',
                      iCondition=0, fCondition=-1, iCase=0, fCase=-1, iSeed=0, fSeed=-1, iTurbine=0, fTurbine=-1):
    '''
    caseobj: FASTFarmCaseSetup object
        Object containing all the case information
    dt_openfast: scalar
        OpenFAST time step
    dt_processing: scalar
        Time step to which the processing will be saved. Default=1
    saveOutput: bool
        Whether or not to save the output to a zarr file
    output: str
        Format to save output. Only 'zarr' and 'nc' available
    '''
    
    if fCondition==-1:
        fCondition = caseobj.nConditions
    if fCondition-iCondition <= 0:
        raise ValueError (f'Final condition to read needs to be larger than initial.')

    if fCase==-1:
        fCase = caseobj.nCases
    if fCase-iCase <= 0:
        raise ValueError (f'Final case to read needs to be larger than initial.')

    if fSeed==-1:
        fSeed = caseobj.nSeeds
    if fSeed-iSeed <= 0:
        raise ValueError (f'Final seed to read needs to be larger than initial.')

    if fTurbine==-1:
        fTurbine = caseobj.nTurbines
    if fTurbine-iTurbine <= 0:
        raise ValueError (f'Final turbine to read needs to be larger than initial.')

    outfilename = f'ds_turbineOutput_temp_cond{iCondition}_{fCondition}_case{iCase}_{fCase}_seed{iSeed}_{fSeed}_turb{iTurbine}_{fTurbine}_dt{dt_processing}s'
    zarrstore = f'{outfilename}.zarr'
    ncfile = f'{outfilename}.nc'
    outputzarr = os.path.join(caseobj.path, zarrstore)
    outputnc   = os.path.join(caseobj.path, ncfile)

    # Read or process turbine output
    if os.path.isdir(outputzarr) or os.path.isfile(outputnc):
        # Data already processed. Reading output
        if output == 'zarr': 
            print(f'Output file {outputzarr} exists. Loading it.')
            turbs = xr.open_zarr(outputzarr) 
        elif output == 'nc': 
            print(f'Output file {outputnc} exists. Loading it.')
            turbs = xr.open_dataset(outputnc)
    else: 
        print(f'{outfilename}.{output} does not exist. Reading output data...')
        # Processed data not saved. Reading it
        dt_ratio = int(dt_processing/dt_openfast)

        turbs_cond = []
        for cond in np.arange(iCondition, fCondition, 1):
            turbs_case = []
            for case in np.arange(iCase, fCase, 1):
                turbs_seed = []
                for seed in np.arange(iSeed, fSeed, 1):
                    turbs_t=[]
                    for t in np.arange(iTurbine, fTurbine, 1):
                        print(f'Processing Condition {cond}, Case {case}, Seed {seed}, turbine {t+1}')

                        ff_file = os.path.join(caseobj.path, caseobj.condDirList[cond], caseobj.caseDirList[case], f'Seed_{seed}', f'{_get_fstf_filename(caseobj)}.T{t+1}')
                        if os.path.isfile(ff_file+'.outb'):
                            ff_file = ff_file + '.outb'
                        elif os.path.isfile(ff_file+'.out'):
                            ff_file = ff_file + '.out'
                        else:
                            raise ValueError(f'ERROR: neither output file {ff_file}.out[b] exists. ')
                            
                        df   = FASTOutputFile(ff_file).toDataFrame()
                        # Won't be able to send to xarray if columns are non-unique
                        if not df.columns.is_unique:
                            df = df.T.groupby(df.columns).first().T
                        ds_t = df.rename(columns={'Time_[s]':'time'}).set_index('time').to_xarray()
                        ds_t = ds_t.isel(time=slice(0,None,dt_ratio))
                        ds_t = ds_t.expand_dims(['cond','case','seed','turbine']).assign_coords({'cond': [caseobj.condDirList[cond]],
                                                                                                 'case':[caseobj.caseDirList[case]],
                                                                                                 'seed':[seed],
                                                                                                 'turbine': [t+1]})
                        turbs_t.append(ds_t)
                    turbs_t = xr.concat(turbs_t, dim='turbine', data_vars='all')
                    turbs_seed.append(turbs_t)
                turbs_seed = xr.concat(turbs_seed, dim='seed', data_vars='all')
                turbs_case.append(turbs_seed)
            turbs_case = xr.concat(turbs_case, dim='case', data_vars='all')
            turbs_cond.append(turbs_case)
        turbs_cond = xr.concat(turbs_cond, dim='cond', data_vars='all')

        # Rename variables to get rid of problematic characters ('-','/')
        varlist = list(turbs_cond.keys())
        varlistnew = [i.replace('/','_per_').replace('-','') for i in varlist]
        renameDict = dict(zip(varlist, varlistnew))
        turbs = turbs_cond.rename_vars(renameDict)

        if saveOutput:
            print(f'Saving output {outfilename}.{output}...')
            if output == 'zarr':  turbs.to_zarr(outputzarr)
            elif output == 'nc':  turbs.to_netcdf(outputnc)
            print(f'Saving output {outfilename}.{output}... Done.')
    
    return turbs



def _nOutputTimes(caseobj, cond, case, seed, sliceToRead, terrainSlice=None):
    '''
    Number of snapshots that readFFPlanes reads for one cond/case/seed when ftime=-1: the smaller of the number expected from TMax and
    the output time step, and the number of files on disk of the slice's first family (same rule as in readFFPlanes). Used by
    readFFPlanesPar to split the time range among the workers.
    '''
    seedPath = _getCaseSeedPath(caseobj, cond, case, seed)
    ff_file = FASTInputFile(os.path.join(seedPath,f'{_get_fstf_filename(caseobj)}.fstf'))
    vtkfiles = os.listdir( os.path.join(seedPath, 'vtk_ff'))

    if sliceToRead == 'terrain':
        if terrainSlice is None:  terrainSlice = _terrainSliceRow(ff_file['TerrainSlices'][0])[0]
        WrTerrainDT = ff_file['WrTerrainDT']
        if str(WrTerrainDT).strip('"').upper() == 'DEFAULT':  WrTerrainDT = _getDT_Low(ff_file)
        prefix = f'{_get_fstf_filename(caseobj)}.TerrSlice.{terrainSlice}.'
        ext = '.vtp'
        nOutputTimes_expected = int(np.floor(ff_file['TMax']/WrTerrainDT))
    else:
        prefix = f'{_get_fstf_filename(caseobj)}.Low.DisXY001.'
        ext = '.vtk'
        nOutputTimes_expected = int(np.floor(ff_file['TMax']/ff_file['WrDisDT']))

    nOutputTimes_available = len([f for f in vtkfiles if f.startswith(prefix) and f.endswith(ext)])

    return min(nOutputTimes_expected, nOutputTimes_available)



def readFFPlanesPar(caseobj, slicesToRead, verbose=False, saveOutput=True, iCondition=0, fCondition=-1, iCase=0, fCase=-1, iSeed=0, fSeed=-1, itime=0, ftime=-1, skiptime=1, nCores=36, outputformat='nc', terrainSlice=None):
    '''
    Parallel version of readFFPlanes for one slice ('x', 'y', 'z' or 'terrain'; see readFFPlanes for the inputs). The work is split
    among the workers by condition, by case, or, for a single condition and case, by time. Each worker calls readFFPlanes on its share,
    and the result is combined and saved under the same file name readFFPlanes uses, so a later call to either function loads it.
    '''

    if fCondition==-1:
        fCondition = caseobj.nConditions
    if fCondition-iCondition <= 0:
        raise ValueError (f'Final condition to read needs to be larger than initial.')

    if fCase==-1:
        fCase = caseobj.nCases
    if fCase-iCase <= 0:
        raise ValueError (f'Final case to read needs to be larger than initial.')

    if fSeed==-1:
        fSeed = caseobj.nSeeds
    if fSeed-iSeed <= 0:
        raise ValueError (f'Final seed to read needs to be larger than initial.')

    if outputformat not in ['zarr','nc']:
        raise ValueError (f'Output can only be zarr or nc')


    outfilename = f'ds_{slicesToRead}Slices_temp_cond{iCondition}_{fCondition}_case{iCase}_{fCase}_seed{iSeed}_{fSeed}'
    if slicesToRead == 'terrain' and terrainSlice is not None:  outfilename = f'ds_terrainSlices_{terrainSlice}_temp_cond{iCondition}_{fCondition}_case{iCase}_{fCase}_seed{iSeed}_{fSeed}'
    zarrstore = f'{outfilename}.zarr'
    ncfile = f'{outfilename}.nc'
    outputzarr = os.path.join(caseobj.path, zarrstore)
    outputnc   = os.path.join(caseobj.path, ncfile)

    if outputformat=='zarr' and os.path.isdir(outputzarr):
       print(f'Output file {zarrstore} exists. Loading it.')
       comb_ds = xr.open_zarr(outputzarr)
       return comb_ds
    if outputformat=='nc' and os.path.isfile(outputnc):
       print(f'Output file {ncfile} exists. Loading it.')
       comb_ds = xr.open_dataset(outputnc)
       return comb_ds


    # We will loop in the variable that we have more entries. E.g. if we have several condition, and only one case,
    # then we will loop on the conditions. Analogous, if we have single conditions with many cases, we will loop on
    # cases. A single condition and case is looped in time. Here we figure out where the loop will take place and set
    # the appropriate number of cores to be used.
    if fCondition-iCondition == 1 and fCase-iCase == 1:
        loopOn = 'time'
        print(f'Single condition, single case. Looping in time.')
        if ftime == -1:
            ftime = _nOutputTimes(caseobj, iCondition, iCase, iSeed, slicesToRead, terrainSlice)
        nTimes = len(range(itime,ftime,skiptime))
        if nTimes == 0:
            raise ValueError (f'No snapshots to read between itime={itime} and ftime={ftime} with skiptime={skiptime}.')
        if nTimes < nCores:
            print(f'Total number of times requested ({nTimes}) is lower than number of cores {nCores}.')
            print(f'Changing the number of cores to {nTimes}.')
            nCores = nTimes

    elif fCondition-iCondition >= fCase-iCase:
        loopOn = 'cond'
        print(f'Looping on conditions.')
        if fCondition-iCondition < nCores:
            print(f'Total number of condtions requested ({fCondition-iCondition}) is lower than number of cores {nCores}.')
            print(f'Changing the number of cores to {fCondition-iCondition}.')
            nCores = fCondition-iCondition

    else:
        loopOn = 'case'
        print(f'Looping on cases.')
        if fCase-iCase < nCores:
            print(f'Total number of cases requested ({fCase-iCase}) is lower than number of cores {nCores}.')
            print(f'Changing the number of cores to {fCase-iCase}.')
            nCores = fCase-iCase


    print(f'Running readFFPlanes in parallel using {nCores} workers')

    if loopOn == 'cond':
        # Split all the conditions in arrays of roughly the same size
        chunks =  np.array_split(range(iCondition,fCondition), nCores)
        # Now, get the beginning and end of each separate chunk
        iCond_list = [i[0]    for i in chunks]
        fCond_list = [i[-1]+1 for i in chunks] 
        print(f'iCond_list is {iCond_list}')
        print(f'fCond_list is {fCond_list}')

        # For generality, we create the list of cases and time as a repeat
        iCase_list = repeat(iCase)
        fCase_list = repeat(fCase)
        itime_list = repeat(itime)
        ftime_list = repeat(ftime)

    elif loopOn == 'case':
        # Split all the cases in arrays of roughly the same size
        chunks =  np.array_split(range(iCase,fCase), nCores)
        # Now, get the beginning and end of each separate chunk
        iCase_list = [i[0]    for i in chunks]
        fCase_list = [i[-1]+1 for i in chunks] 
        print(f'iCase_list is {iCase_list}')
        print(f'fCase_list is {fCase_list}')

        # For generality, we create the list of cond and time as a repeat
        iCond_list = repeat(iCondition)
        fCond_list = repeat(fCondition)
        itime_list = repeat(itime)
        ftime_list = repeat(ftime)

    elif loopOn == 'time':
        # Split all the times in arrays of roughly the same size
        chunks =  np.array_split(range(itime,ftime,skiptime), nCores)
        # Now, get the beginning and end of each separate chunk
        itime_list = [i[0]    for i in chunks]
        ftime_list = [i[-1]+1 for i in chunks] 
        print(f'itime_list is {itime_list}')
        print(f'ftime_list is {ftime_list}')

        # For generality, we create the list of cond and cases as a repeat
        iCond_list = repeat(iCondition)
        fCond_list = repeat(fCondition)
        iCase_list = repeat(iCase)
        fCase_list = repeat(fCase)


    p = Pool(nCores)
    ds_ = p.starmap(readFFPlanes, zip(repeat(caseobj),          # caseobj
                                      repeat([slicesToRead]),   # slicesToRead
                                      repeat(verbose),          # verbose
                                      repeat(False),            # saveOutput
                                      iCond_list,               # iCondition
                                      fCond_list,               # fCondition
                                      iCase_list,               # iCase
                                      fCase_list,               # fCase
                                      repeat(iSeed),            # iSeed
                                      repeat(fSeed),            # fSeed
                                      itime_list,               # itime
                                      ftime_list,               # ftime
                                      repeat(skiptime),         # skiptime
                                      repeat(outputformat),     # outformat
                                      repeat(terrainSlice)      # terrainSlice
                                     )
                                  )
    p.close()
    p.terminate()
    p.join()

    print(f'Done reading all output. Concatenating the arrays')

    try:
        comb_ds = xr.combine_by_coords(ds_)
    except ValueError as e:
        if 'neither monotonically increasing nor monotonically decreasing' in str(e):
            print('Concatenation using combine_by_coords failed. Concatenating using merge instead.')
            print('  WARNING: Indexes are _not_ monotonically increasing. Do not use `isel`.')
            print('           Try using `.sortby(<dimstr>)` to sort it.')
            comb_ds = xr.merge(ds_)
        else:
            raise


    if saveOutput:
        print(f'Done concatenating. Saving {outputformat} file.')
        if outputformat == 'zarr':  comb_ds.to_zarr(outputzarr)
        elif outputformat == 'nc':  comb_ds.to_netcdf(outputnc)


    print('Finished.')

    return comb_ds



def readFFPlanes(caseobj, slicesToRead=['x','y','z'], verbose=False, saveOutput=True, iCondition=0, fCondition=-1, iCase=0, fCase=-1, iSeed=0, fSeed=-1, itime=0, ftime=-1, skiptime=1, outformat='nc', terrainSlice=None):
    '''
    Read and process FAST.Farm planes into xarrays.

    INPUTS
    ======
    slicesToRead: list of str
        Slices to read: 'x', 'y', 'z' for the disturbed-wind planes normal to that axis (Low.DisYZ, Low.DisXZ, Low.DisXY .vtk files),
        and 'terrain' for the terrain-following point clouds (TerrSlice .vtp files). The terrain
        dataset has dimensions (time, seed, case, cond, slice, offset, x, y) with z(slice, offset, x, y) as a coordinate, or
        (time, seed, case, cond, slice, offset, point) with x, y, z as (slice, offset, point) coordinates when the cloud is not on a
        regular x-y lattice (e.g. sampled from an STL surface) or the offsets are not vertical. Its time is the snapshot number times WrTerrainDT.
    terrainSlice: str
        Name (SliceName in the FAST.Farm input file) of the single terrain slice to read. Default None reads all of them, concatenated along `slice`.
    i<quant>, f<quant>: int
        Initial and end index of <quant> to read
    itime: int
        Initial timestep to open and read
    ftime: int
        Final timestep to open and read
    skiptime: int
        Read at every skiptime timestep. Used when data is too fine and not needed.

    '''

    if isinstance(slicesToRead, str):  slicesToRead = [slicesToRead]

    if fCondition==-1:
        fCondition = caseobj.nConditions
    if fCondition-iCondition <= 0:
        raise ValueError (f'Final condition to read needs to be larger than initial.')

    if fCase==-1:
        fCase = caseobj.nCases
    if fCase-iCase <= 0:
        raise ValueError (f'Final case to read needs to be larger than initial.')

    if fSeed==-1:
        fSeed = caseobj.nSeeds
    if fSeed-iSeed <= 0:
        raise ValueError (f'Final seed to read needs to be larger than initial.')

    if skiptime<1:
        raise ValueError (f'Skiptime should be 1 or greater. If 1, no slices will be skipped.')

    if verbose: print(f'Requesting to save {slicesToRead} slices')

    # Read all VTK output for each plane and save an nc files for each normal. Load files if present.
    for slices in slicesToRead:
    
        storefile = f'ds_{slices}Slices_temp_cond{iCondition}_{fCondition}_case{iCase}_{fCase}_seed{iSeed}_{fSeed}'
        if slices == 'terrain' and terrainSlice is not None:
            storefile = f'ds_terrainSlices_{terrainSlice}_temp_cond{iCondition}_{fCondition}_case{iCase}_{fCase}_seed{iSeed}_{fSeed}'
        ftime_read = ftime   # resolved from the files of the first cond/case/seed read and then kept for the whole slice family

        if outformat == 'zarr':
            outputfile = os.path.join(caseobj.path, f'{storefile}.zarr')
        elif outformat == 'nc':
            outputfile = os.path.join(caseobj.path, f'{storefile}.nc')

        if os.path.isdir(outputfile): # zarr
            if len(slicesToRead) > 1:
                print(f"!! WARNING: Asked for multiple slices. Returning only the first one, {slices}\n",
                      f"           To load other slices, request `slicesToRead='y'`")
            print(f'Processed output file {outputfile} for slice {slices} exists. Loading it.')
            # Data already processed. Reading output
            Slices = xr.open_zarr(outputfile)
            return Slices

        elif os.path.isfile(outputfile): # nc
            if len(slicesToRead) > 1:
                print(f"!! WARNING: Asked for multiple slices. Returning only the first one, {slices}\n",
                      f"           To load other slices, request `slicesToRead='y'`")
            print(f'Processed output file {outputfile} for slice {slices} exists. Loading it.')
            # Data already processed. Reading output
            Slices = xr.open_dataset(outputfile)
            return Slices

            
        else:
            # This loop is due to memory allocation requirements
            Slices_cond = []
            for cond in np.arange(iCondition, fCondition, 1):
                Slices_case = []
                for case in np.arange(iCase, fCase, 1):
                    Slices_seed = []
                    for seed in np.arange(iSeed, fSeed, 1):
                        seedPath = _getCaseSeedPath(caseobj, cond, case, seed)

                        # Read FAST.Farm input to determine outputs
                        ff_file = FASTInputFile(os.path.join(seedPath,f'{_get_fstf_filename(caseobj)}.fstf'))

                        tmax          = ff_file['TMax']
                        NOutDisWindXY = ff_file['NOutDisWindXY']
                        OutDisWindZ   = ff_file['OutDisWindZ']
                        NOutDisWindYZ = ff_file['NOutDisWindYZ']
                        OutDisWindX   = ff_file['OutDisWindX']
                        NOutDisWindXZ = ff_file['NOutDisWindXZ']
                        WrDisDT       = ff_file['WrDisDT']
                        # Terrain-following slices. Older input files do not have the block.
                        if 'NumTerrainSlices' in ff_file.keys():
                            NumTerrainSlices = ff_file['NumTerrainSlices']
                            WrTerrainDT      = ff_file['WrTerrainDT']
                            TerrainSlices    = ff_file['TerrainSlices']
                            if str(WrTerrainDT).strip('"').upper() == 'DEFAULT':  WrTerrainDT = _getDT_Low(ff_file)   # DEFAULT means DT_Low in FAST.Farm
                        else:
                            NumTerrainSlices = 0

                        # Determine number of output VTKs based on succesful completion, and the number of actually available snapshots from the
                        # files on disk: the Low.DisXY001 family for the x, y, z planes, and the terrain slices themselves for the terrain point clouds
                        vtkfiles = os.listdir( os.path.join(seedPath, 'vtk_ff'))
                        if slices == 'terrain':
                            if NumTerrainSlices == 0:
                                raise ValueError(f'Terrain slices requested, but NumTerrainSlices=0 (or no terrain-following block) in the FAST.Farm input file.')
                            if WrTerrainDT < WrDisDT:
                                print(f'WARNING: WrTerrainDT ({WrTerrainDT}) < WrDisDT ({WrDisDT}). FAST.Farm names the terrain files with the disturbed-wind output counter,'
                                      f' so files overwrite each other and the times are unreliable.')
                            # Slices to read (the one requested, or all rows) and their files on disk, sorted by time. FAST.Farm writes no files for a
                            # slice it disabled at initialization (all points outside the low-resolution domain); such slices are skipped
                            deckSliceNames = []
                            terrainSliceNames = []
                            terrainSliceOffsets = []
                            terrainSliceFiles = []
                            for row in TerrainSlices:
                                name, offsets = _terrainSliceRow(row)
                                deckSliceNames.append(name)
                                if terrainSlice is not None and name != terrainSlice:
                                    continue
                                sliceprefix = f'{_get_fstf_filename(caseobj)}.TerrSlice.{name}.'
                                files = sorted([f for f in vtkfiles if f.startswith(sliceprefix) and f.endswith('.vtp')])
                                if len(files) == 0:
                                    print(f'WARNING: No files found for terrain slice {name} (FAST.Farm skips slices that fall entirely outside the low-resolution domain). Skipping it.')
                                    continue
                                terrainSliceNames.append(name)
                                terrainSliceOffsets.append(offsets)
                                terrainSliceFiles.append(files)
                            if terrainSlice is not None and terrainSlice not in deckSliceNames:
                                raise ValueError(f'Terrain slice {terrainSlice} is not in the FAST.Farm input file. Available terrain slices: {deckSliceNames}')
                            if len(terrainSliceNames) == 0:
                                raise ValueError(f'No terrain slice files found in {os.path.join(seedPath, "vtk_ff")}.')
                            firstfamily_files = terrainSliceFiles[0]
                            nOutputTimes_expected = int(np.floor(tmax/WrTerrainDT))
                            nOutputTimes_available = min([len(files) for files in terrainSliceFiles])
                        else:
                            disxy001_prefix = f'{_get_fstf_filename(caseobj)}.Low.DisXY001.'
                            firstfamily_files = [f for f in vtkfiles if f.startswith(disxy001_prefix) and f.endswith('.vtk')]
                            nOutputTimes_expected = int(np.floor(tmax/WrDisDT))
                            nOutputTimes_available = len(firstfamily_files)
                        if nOutputTimes_available != nOutputTimes_expected:
                            print(f'WARNING: Expected {nOutputTimes_expected} outputs from TMax and the output time step, but found {nOutputTimes_available}'
                                  f' files matching. Reading only available snapshots.')
                        nOutputTimes = min(nOutputTimes_expected, nOutputTimes_available)
                        

                        # Determine number of output digits for reading
                        ndigitsplane = max(len(str(max(NOutDisWindXY,NOutDisWindXZ,NOutDisWindYZ))), 3)
                        # Number of time digits is taken from the files actually on disk
                        ndigitstime = len(firstfamily_files[0].split('.')[-2]) if nOutputTimes_available > 0 else len(str(nOutputTimes_expected))


                        # Determine how many snapshots to read depending on input
                        if ftime_read==-1:
                            ftime_read=nOutputTimes
                        elif ftime_read>nOutputTimes:
                            raise ValueError (f'Final time step requested ({ftime_read}) is greater than the total available ({nOutputTimes})')

                        if verbose:
                            print(f'Processing {slices} slice: Condition {cond}, Case {case}, Seed {seed}, snapshot {itime} to {ftime_read} ({nOutputTimes} available)')

                        if slices == 'z':
                            # Read Low-res z-planes
                            Slices=[]
                            for zplane in range(NOutDisWindXY):
                                Slices_t=[]
                                #for t in range(nOutputTimes):
                                for t in np.arange(itime,ftime_read,skiptime):
                                    file = f'{_get_fstf_filename(caseobj)}.Low.DisXY{zplane+1:0{ndigitsplane}d}.{t:0{ndigitstime}d}.vtk'
                                    if verbose: print(f'Reading z plane {zplane} for time step {t}: \t {file}')

                                    vtk = VTKFile(os.path.join(seedPath, 'vtk_ff', file))
                                    ds = readAndCreateDataset(vtk, caseobj, cond=cond, case=case, seed=seed, t=t, WrDisDT=WrDisDT)

                                    Slices_t.append(ds)
                                Slices_t = xr.concat(Slices_t, dim='time', data_vars='all')
                                Slices.append(Slices_t)
                            Slices = xr.concat(Slices, dim='z', data_vars='all')

                        elif slices == 'y':
                            # Read Low-res y-planes
                            Slices=[]
                            for yplane in range(NOutDisWindXZ):
                                Slices_t=[]
                                #for t in range(nOutputTimes):
                                for t in np.arange(itime,ftime_read,skiptime):
                                    file = f'{_get_fstf_filename(caseobj)}.Low.DisXZ{yplane+1:0{ndigitsplane}d}.{t:0{ndigitstime}d}.vtk'
                                    if verbose: print(f'Reading y plane {yplane} for time step {t}: \t {file}')

                                    vtk = VTKFile(os.path.join(seedPath, 'vtk_ff', file))
                                    ds = readAndCreateDataset(vtk, caseobj, cond=cond, case=case, seed=seed, t=t, WrDisDT=WrDisDT)

                                    Slices_t.append(ds)
                                Slices_t = xr.concat(Slices_t, dim='time', data_vars='all')
                                Slices.append(Slices_t)
                            Slices = xr.concat(Slices, dim='y', data_vars='all')

                        elif slices == 'x':
                            # Read Low-res x-planes
                            Slices=[]
                            for xplane in range(NOutDisWindYZ):
                                Slices_t=[]
                                print(f'Processing {slices} slice: Condition {cond}, Case {case}, Seed {seed}, x plane {xplane}')
                                #for t in range(nOutputTimes):
                                for t in np.arange(itime,ftime_read,skiptime):
                                    file = f'{_get_fstf_filename(caseobj)}.Low.DisYZ{xplane+1:0{ndigitsplane}d}.{t:0{ndigitstime}d}.vtk'
                                    if verbose: print(f'Reading x plane {xplane} for time step {t}: \t {file}')

                                    vtk = VTKFile(os.path.join(seedPath, 'vtk_ff', file))
                                    ds = readAndCreateDataset(vtk, caseobj, cond=cond, case=case, seed=seed, t=t, WrDisDT=WrDisDT)

                                    Slices_t.append(ds)
                                Slices_t = xr.concat(Slices_t, dim='time', data_vars='all')
                                Slices.append(Slices_t)
                            Slices = xr.concat(Slices, dim='x', data_vars='all')

                        elif slices == 'terrain':
                            # Read terrain-following point clouds: one .vtp per slice and snapshot, with all offsets of the slice in the same file.
                            # FAST.Farm names these files with the disturbed-wind output counter (n/WrDisSkp1 in AWAE.f90), not with a terrain
                            # counter, so the file of a snapshot is looked up in the sorted list from disk and the time is the snapshot number
                            # times WrTerrainDT
                            Slices=[]
                            for terrslice in range(len(terrainSliceNames)):
                                sliceName = terrainSliceNames[terrslice]
                                offsets = terrainSliceOffsets[terrslice]
                                Slices_t=[]
                                for t in np.arange(itime,ftime_read,skiptime):
                                    file = terrainSliceFiles[terrslice][t]
                                    if verbose: print(f'Reading terrain slice {sliceName} for time step {t}: \t {file}')

                                    points, vel = readVTP_pointCloud(os.path.join(seedPath, 'vtk_ff', file))
                                    ds = readAndCreateDatasetTerrain(points, vel, sliceName, offsets, caseobj, cond=cond, case=case, seed=seed, t=t, WrTerrainDT=WrTerrainDT)

                                    Slices_t.append(ds)
                                Slices_t = xr.concat(Slices_t, dim='time', data_vars='all')
                                Slices.append(Slices_t)
                            for Slices_t in Slices:
                                if set(Slices_t.dims) != set(Slices[0].dims):
                                    raise ValueError(f'The terrain slices {terrainSliceNames} are not all on a regular lattice, so they cannot be combined in one dataset. Read them one at a time with terrainSlice=<name>.')
                            Slices = xr.concat(Slices, dim='slice', data_vars='all', join='outer')   # slices may be on different grids

                        else:
                            raise ValueError(f'Only slices x, y, z, terrain are available. Slice {slices} was requested. Stopping.')

                        Slices_seed.append(Slices)
                    Slices_seed = xr.concat(Slices_seed, dim='seed', data_vars='all')
                    Slices_case.append(Slices_seed)
                Slices_case = xr.concat(Slices_case, dim='case', data_vars='all')
                Slices_cond.append(Slices_case)

            Slices = xr.concat(Slices_cond, dim='cond', data_vars='all')

            if saveOutput:
                print(f'Saving {slices} slice file {outputfile}...')
                if outformat == 'zarr':
                    Slices.to_zarr(outputfile)
                elif outformat == 'nc':
                    Slices.to_netcdf(outputfile)

            if len(slicesToRead) == 1:
                # Single slice was requested
                if verbose: print(f'Since single slice was requested, returning it.')
                return Slices
            


def readAndCreateDataset(vtk, caseobj, cond=None, case=None, seed=None, t=None, WrDisDT=None):
    
    # Get info from VTK
    x = vtk.xp_grid
    y = vtk.yp_grid
    z = vtk.zp_grid
    u = vtk.point_data_grid['Velocity'][:,:,:,0]
    v = vtk.point_data_grid['Velocity'][:,:,:,1]
    w = vtk.point_data_grid['Velocity'][:,:,:,2]
    
    if t is None and WrDisDT is None:
        t=1
        WrDisDT = 1

    ds = xr.Dataset({
            'u': (['x', 'y', 'z'], u),
            'v': (['x', 'y', 'z'], v),
            'w': (['x', 'y', 'z'], w), },
           coords={
            'x': (['x'], x),
            'y': (['y'], y),
            'z': (['z'], z),
            'time': [t*WrDisDT] },
          )
    
    if cond is not None:  ds = ds.expand_dims('cond').assign_coords({'cond': [caseobj.condDirList[cond]]})
    if case is not None:  ds = ds.expand_dims('case').assign_coords({'case': [caseobj.caseDirList[case]]})
    if seed is not None:  ds = ds.expand_dims('seed').assign_coords({'seed': [seed]})           
        
    return ds        


def _terrainSliceRow(row):
    '''
    Name and offsets (m) of one row of the TERRAIN-FOLLOWING SAMPLING table, `"name"  off1 [off2 ...]  (Nx Ny Nz | default)  SourceType  "file"`.
    The name is the first quoted string. The numeric tokens after it are the offsets, unless the run of numbers ends at the SourceType instead of
    at `default`, in which case its last three numbers are the (Nx Ny Nz) normal.
    '''
    name = row.split('"')[1]
    tokens = row.split('"')[2].replace('(',' ').replace(')',' ').replace(',',' ').split()   # everything between the name and the file name
    numbers = []
    for token in tokens:
        try:
            numbers.append(float(token))
        except ValueError:
            break
    normalIsDefault = len(numbers) < len(tokens) and tokens[len(numbers)].lower() == 'default'
    if normalIsDefault:
        offsets = numbers
    else:
        offsets = numbers[:-3]
    if len(offsets) == 0:  offsets = [0.0]   # no offset given means sampling on the surface itself
    return name, offsets


def readAndCreateDatasetTerrain(points, vel, sliceName, offsets, caseobj, cond=None, case=None, seed=None, t=None, WrTerrainDT=None):
    '''
    Terrain-following counterpart of readAndCreateDataset. `points` and `vel` are the (N,3) arrays of one .vtp, which holds all the offsets of
    the slice back to back (the first N/nOffsets points are the first offset, and so on). When the offsets are vertical (`default` normal) all
    the sheets share their x, y, and a cloud on a regular x-y lattice is gridded to (offset, x, y) with z(offset, x, y) as a coordinate (lattice
    nodes not in the cloud become NaN). Any other cloud is kept on a `point` dimension, with x, y, z as (offset, point) coordinates.
    '''

    nOffsets = len(offsets)
    nPoints = points.shape[0]//nOffsets
    x = points[:,0].reshape(nOffsets, nPoints)
    y = points[:,1].reshape(nOffsets, nPoints)
    z = points[:,2].reshape(nOffsets, nPoints)
    u = vel[:,0].reshape(nOffsets, nPoints)
    v = vel[:,1].reshape(nOffsets, nPoints)
    w = vel[:,2].reshape(nOffsets, nPoints)

    if t is None and WrTerrainDT is None:
        t=1
        WrTerrainDT = 1

    # The sheets share their x, y only when the offsets are vertical; a tilted normal shifts every sheet sideways
    sheetsShareXY = np.all(x == x[0]) and np.all(y == y[0])
    # A cloud is a regular lattice when at most half of the unique-x by unique-y nodes are missing from it. A cloud sampled from an STL surface
    # has about as many distinct x and y values as points, so its lattice would be almost empty and it is kept as points
    xlattice = np.unique(x[0])
    ylattice = np.unique(y[0])
    isLattice = len(xlattice)*len(ylattice) <= 2*nPoints

    if sheetsShareXY and isLattice:
        # Scatter every point onto its lattice node
        ix = np.searchsorted(xlattice, x[0])
        iy = np.searchsorted(ylattice, y[0])
        z_grid = np.full((nOffsets, len(xlattice), len(ylattice)), np.nan, dtype=z.dtype)
        u_grid = np.full((nOffsets, len(xlattice), len(ylattice)), np.nan, dtype=u.dtype)
        v_grid = np.full((nOffsets, len(xlattice), len(ylattice)), np.nan, dtype=v.dtype)
        w_grid = np.full((nOffsets, len(xlattice), len(ylattice)), np.nan, dtype=w.dtype)
        z_grid[:,ix,iy] = z
        u_grid[:,ix,iy] = u
        v_grid[:,ix,iy] = v
        w_grid[:,ix,iy] = w

        ds = xr.Dataset({
                'u': (['offset', 'x', 'y'], u_grid),
                'v': (['offset', 'x', 'y'], v_grid),
                'w': (['offset', 'x', 'y'], w_grid),
                'z': (['offset', 'x', 'y'], z_grid), },
               coords={
                'offset': (['offset'], offsets),
                'x': (['x'], xlattice),
                'y': (['y'], ylattice),
                'time': [t*WrTerrainDT] },
              )
        coordVars = ['z']
    else:
        ds = xr.Dataset({
                'u': (['offset', 'point'], u),
                'v': (['offset', 'point'], v),
                'w': (['offset', 'point'], w),
                'x': (['offset', 'point'], x),
                'y': (['offset', 'point'], y),
                'z': (['offset', 'point'], z), },
               coords={
                'offset': (['offset'], offsets),
                'point': (['point'], np.arange(nPoints)),
                'time': [t*WrTerrainDT] },
              )
        coordVars = ['x', 'y', 'z']

    # The coordinates that vary per point are data variables up to here so that they also get the slice dimension
    ds = ds.expand_dims('slice').assign_coords({'slice': [sliceName]}).set_coords(coordVars)
    if cond is not None:  ds = ds.expand_dims('cond').assign_coords({'cond': [caseobj.condDirList[cond]]})
    if case is not None:  ds = ds.expand_dims('case').assign_coords({'case': [caseobj.caseDirList[case]]})
    if seed is not None:  ds = ds.expand_dims('seed').assign_coords({'seed': [seed]})

    return ds


def readVTP_pointCloud(vtppath):
    '''
    Read a VTK XML PolyData (.vtp) point cloud written by FAST.Farm's terrain-following sampling, either in its
    binary appended-raw form (the default) or in its ASCII form (FF_SLICE_ASCII environment variable set at run time).
    Returns the point coordinates (N,3) as float64 and the velocity vectors (N,3) as float32.
    '''

    with open(vtppath, 'rb') as f:
        raw = f.read()

    if b'format="appended"' in raw[:4096]:
        # Binary: each DataArray is a [uint32 byte count][payload] block at its `offset` after the underscore that opens the appended data
        vtktypes = {b'Float32': '<f4', b'Int64': '<i8'}
        header = raw[:raw.index(b'<AppendedData')]
        blob = raw[raw.index(b'_', raw.index(b'<AppendedData'))+1:]
        arrays = []
        for tag in re.findall(rb'<DataArray [^>]*>', header):
            offset = int(re.search(rb'offset="(\d+)"', tag).group(1))
            dtype = vtktypes[re.search(rb'type="(\w+)"', tag).group(1)]
            nbytes = int(np.frombuffer(blob[offset:offset+4], '<u4')[0])
            arrays.append(np.frombuffer(blob[offset+4:offset+4+nbytes], dtype))
    else:
        # ASCII: the values are written between the DataArray tags (NaN as the text NaN)
        arrays = []
        for text in re.findall(rb'<DataArray[^>]*>(.*?)</DataArray>', raw, re.S):
            arrays.append(np.array(text.split(), dtype=float))

    # The arrays are, in file order: points, verts connectivity, verts offsets, velocity
    points = arrays[0].reshape(-1,3).astype(float)
    vel    = arrays[3].reshape(-1,3).astype(np.float32)

    return points, vel


def readVTK_structuredPoints (vtkpath):
    '''
    Function to read the VTK written by utilities/postprocess_amr_boxes2vtk.py
    Input
    -----
    vtkpath: str
        Full path of the vtk, including its extension
    '''

    import vtk

    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(vtkpath)
    reader.Update()

    output = reader.GetOutput()

    dims = output.GetDimensions()
    spacing = output.GetSpacing()
    origin = output.GetOrigin()
    nx, ny, nz = dims

    data_type = output.GetScalarTypeAsString()
    point_data = output.GetPointData()
    vector_array = point_data.GetArray(0)
    num_components = vector_array.GetNumberOfComponents()


    # Convert vector array to a NumPy array
    vector_data = np.zeros((nx, ny, nz, num_components), dtype=np.float32)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                index = i + nx * (j + ny * k)
                vector = vector_array.GetTuple(index)
                vector_data[i, j, k, :] = vector

    # Create coordinates along x, y, and z dimensions
    x_coords = origin[0] + spacing[0] * np.arange(nx)
    y_coords = origin[1] + spacing[1] * np.arange(ny)
    z_coords = origin[2] + spacing[2] * np.arange(nz)

    # Create the xarray dataset
    ds = xr.Dataset(data_vars = {
                      'u': (['x', 'y', 'z'], vector_data[:,:,:,0]),
                      'v': (['x', 'y', 'z'], vector_data[:,:,:,1]),
                      'w': (['x', 'y', 'z'], vector_data[:,:,:,2]),
                    }, 
                    coords = {
                      'x': x_coords,
                      'y': y_coords,
                      'z': z_coords,
                      }
                  )

    return ds


def compute_load_rose(turbs, nSectors=18):
    
    channel_pairs = []                                                          
    channel_out   = []

    if 'TwrBsMxt_[kNm]' in turbs and 'TwrBsMyt_[kNm]' in turbs:
        channel_pairs += [['TwrBsMyt_[kNm]', 'TwrBsMxt_[kNm]']]  # fore-aft, side-to-side
        channel_out += ['TwrBsMt_[kNm]']

    if 'RootMxc1_[kNm]' in turbs and 'RootMyc1_[kNm]' in turbs:
        channel_pairs += [['RootMyc1_[kNm]', 'RootMxc1_[kNm]']]  # out-of-plane, in-plane
        channel_out   += ['RootMc1_[kNm]']

    if 'LSSGagMya_[kNm]' in turbs and 'LSSGagMza_[kNm]' in turbs:
        channel_pairs += [['LSSGagMya_[kNm]','LSSGagMza_[kNm]']]
        channel_out   += ['LSSGagMa_[kNm]']
    
    if nSectors%2 != 0:
        print(f'WARNING: it is recommended an even number of sectors')
        
    # Create the sector bins
    theta_bin = np.linspace(0,180, nSectors+1)

    # Bin the loads for each pair
    for p, curr_pair in enumerate(channel_pairs):
        print(f'Processing pair {curr_pair[0]}, {curr_pair[1]}.')
        
        load_0deg = turbs[curr_pair[0]]
        load_90deg = turbs[curr_pair[1]]
        all_load = []
        for i, theta in enumerate(theta_bin[:-1]):
            curr_theta = (theta_bin[i] + theta_bin[i+1])/2
            curr_load = load_0deg*np.cos(np.deg2rad(curr_theta)) + load_90deg*np.sin(np.deg2rad(curr_theta))
            all_load.append(curr_load.expand_dims('theta').assign_coords({'theta': [curr_theta]}))
                            
        all_load = xr.concat(all_load, dim='theta', data_vars='all').to_dataset(name=channel_out[p])
        
        turbs = xr.merge([turbs,all_load])
    
    return turbs



def compute_worst_performing_sector(ds=None, ds_mean=None, ds_std=None, vars=['TwrBsMt_[kNm]','RootMc1_[kNm]','LSSGagMa_[kNm]'], add_as='regvar'):
    '''
    From an dataset with sectors indicated by the `theta` variable (that is,
    output of `compute_load_rose`), get the sector of worst performance and
    add it as another variable. Takes either the regular ds or a pair of ds,
    one with mean and other with standard deviation. To find the sector with
    the worse performance, we look at the mean datasets, then find the same
    sector on the std ds and do the operations there. It returns the std and
    mean datasets with an additional variables `<var>_std_of_theta_of_maxmean`
    and `<var>_maxmean`, respectively.  Additionally, it adds a non-dimension
    coordinate with `theta_of_max_mean` indicating which sector the maximum
    mean occurred, which is also the same sector of the worse std. Perform all
    the operations in place, and returns both the mean and std and datasets.

    Inputs:
    -------
    ds: xr.Dataset
        Dataset output of compute_load_rose. Does not contain the mean nor
        the standard deviation data-- those will be computed internally.
        Has coordinate `turbines`.
    ds_mean: xr.Dataset
        Dataset output of compute_load_rose, followed by a `mean` operation.
        Contains the mean data. Has coordinate `turbines`.
    ds_std: xr.Dataset
        Dataset output of compute_load_rose, followed by a `std` operation.
        Contains the standard deviation data. Has coordinate `turbines`.
    vars: str
        Variable in which to perform the worst sector, if only a subset is 
        needed. By default, does in all three available (blade root, tower
        base, and LSS bending moments).
    add_as: str
        Option to add result as either coordinate or regular variable, given
        by 'coord' or 'regvar'

    Outputs
    -------
    ds_mean: xr.Dataseet
    ds_std: xr.Dataset

    Example call
    ------------
    ds_turbs_rose = compute_load_rose(ds_turbs_les)
    ds_turbs_mean = ds_turbs_les_rose.sel(time=slice(600,None)).mean(dim='time')
    ds_turbs_std  = ds_turbs_les_rose.sel(time=slice(600,None)).std(dim='time')

    ds_turbs_std = compute_worst_performing_sector(ds_mean=ds_turbs_mean, ds_std=ds_turbs_std)

    '''

    # Check input datasets
    if ds is None:
        if ds_mean is None or ds_std is None:
            raise ValueError(f'Either the regular ds or the pair of std and mean needs to be provided')
        dsst = ds_std.copy()
        dsme = ds_mean.copy()
    else:
        if ds_mean is not None or ds_std is not None:
            raise ValueError(f'Either the regular ds or the pair of std and mean needs to be provided')
        print(f'Only ds provided. Calculating the mean and std from `time` 600 until the end.')
        dsme = ds.sel(time=slice(600,None)).mean(dim='time').copy()
        dsst = ds.sel(time=slice(600,None)).std(dim='time').copy()
             
    # Check input
    if add_as not in ('coord', 'regvar'):
        raise ValueError(f"add_as can only be 'coord' or 'regvar'. See docstring")

    # Make vars iterable even if single string
    if isinstance(vars,str): vars = [vars]
        
    for var in vars:
        # Skip processing if such variable is not on dataset
        if var not in dsst:  continue

        print(f'Processing variable {var}')
        
        theta_idx_max = dsme[var].argmax(dim='theta') 
        max_stds      = dsst[var].isel(theta=theta_idx_max)
        max_mean      = dsme[var].isel(theta=theta_idx_max)
        theta_max = dsme['theta'].isel(theta=theta_idx_max)

        if add_as == 'coord':
            # Add the worst std value, related to the theta of worst mean
            dsst[f'{var}_of_theta_of_maxmean'] = xr.DataArray(
                             max_stds.data, dims=["turbine"],
                             coords={ "turbine"         : dsst["turbine"],
                                      f"{var}_theta_of_maxmean": ("turbine", theta_max.data), },
                         )
            # Add the worst mean value, related to the theta of worst mean
            dsme[f'{var}_of_theta_of_maxmean'] = xr.DataArray(
                             max_mean.data, dims=["turbine"],
                             coords={ "turbine"         : dsme["turbine"],
                                      f"{var}_theta_of_maxmean": ("turbine", theta_max.data), },
                         )
        elif add_as == 'regvar':
            # Add the actual worst std to array, related to the theta of worst mean
            dsst[f'{var}_of_theta_of_maxmean'] = ('turbine', max_stds.values)
            # Add the actual worst mean to array, related to the theta of worst mean
            dsme[f'{var}_of_theta_of_maxmean'] = ('turbine', max_mean.values)
            # Add the value of theta related to the maximum mean
            dsst[f'{var}_theta_of_maxmean'] = ('turbine', theta_max.values)
            dsme[f'{var}_theta_of_maxmean'] = ('turbine', theta_max.values)

    return dsme, dsst



def compute_del(ts, elapsed, lifetime, load2stress, slope, Sult, Sc=0.0, rainflow_bins=100, return_damage=False, goodman_correction=False):
    """
    Function from pCrunch.
    
    Computes damage equivalent load of input `ts`.

    Parameters
    ----------
    ts : np.array
        Time series to calculate DEL for.
    elapsed : int | float
        Elapsed time of the time series.
    lifetime : int | float
        Design lifetime of the component / material in years
    load2stress : float (optional)
        Linear scaling coefficient to convert an applied load to stress such that S = load2stress * L
    slope : int | float
        Slope of the fatigue curve.
    Sult : float (optional)
        Ultimate stress for use in Goodman equivalent stress calculation
    Sc : float (optional)
        Stress-axis intercept of log-log S-N Wohler curve. Taken as ultimate stress unless specified
    rainflow_bins : int
        Number of bins used in rainflow analysis.
        Default: 100
    return_damage: boolean
        Whether to compute both DEL and damage
        Default: False
    goodman_correction: boolean
        Whether to apply Goodman mean correction to loads and stress
        Default: False

    """
    
    import fatpack
    
    Scin = Sc if Sc > 0.0 else Sult

    try:
        F, Fmean = fatpack.find_rainflow_ranges(ts, return_means=True)
        fatpack_rainflow_successful = 1
    except:
        print(f'Fatpack call for find_rainflow_ranges did not work. Setting F=Fmean=0')
        fatpack_rainflow_successful = 0
        F = Fmean = np.zeros(1)

    if goodman_correction and np.abs(load2stress) > 0.0:
        F = fatpack.find_goodman_equivalent_stress(F, Fmean, Sult/np.abs(load2stress))

    Nrf, Frf = fatpack.find_range_count(F, rainflow_bins)
    DELs = Frf ** slope * Nrf / elapsed
    DEL = DELs.sum() ** (1.0 / slope)
    # With fatpack do:
    #curve = fatpack.LinearEnduranceCurve(1.)
    #curve.m = slope
    #curve.Nc = elapsed
    #DEL = curve.find_miner_sum(np.c_[Frf, Nrf]) ** (1 / slope)

    # Compute Palmgren/Miner damage using stress
    if not return_damage:
        return DEL
    
    D = np.nan # default return value
    if return_damage and np.abs(load2stress) > 0.0:
        try:
            S, Mrf = fatpack.find_rainflow_ranges(ts*load2stress, return_means=True)
        except:
            S = Mrf = np.zeros(1)
        if goodman_correction:
            S = fatpack.find_goodman_equivalent_stress(S, Mrf, Sult)
        Nrf, Srf = fatpack.find_range_count(S, rainflow_bins)
        curve = fatpack.LinearEnduranceCurve(Scin)
        curve.m = slope
        curve.Nc = 1
        D = curve.find_miner_sum(np.c_[Srf, Nrf])
        if lifetime > 0.0:
            D *= lifetime*365.0*24.0*60.0*60.0 / elapsed
            
    return DEL, D, fatpack_rainflow_successful






def calcDEL_theta (ds, var):
    
    # Set constants
    lifetime = 25        #  Design lifetime of the component / material in years
    #load2stress = 1     #  Linear scaling coefficient to convert an applied load to stress such that S = load2stress * L
    #slope = 10          #  Wohler exponent in the traditional SN-curve of S = A * N ^ -(1/m) (rthedin: 4 for tower, 10 for blades)
    #Sult=6e8            #  Ultimate stress for use in Goodman equivalent stress calculation
    Sc = 0               #  Stress-axis intercept of log-log S-N Wohler curve. Taken as ultimate stress unless specified
    rainflow_bins = 100
    
    # rotorse.rs.strains.axial_root_sparU_load2stress,m**2,[ 0.  0.  -0.06122281 -0.02535384  0.04190673  0. ],Linear conversion factors between loads [Fx-z; Mx-z] and axial stress in the upper spar cap at blade root
    # rotorse.rs.strains.axial_root_sparL_load2stress,m**2,[ 0.  0.  -0.06122281  0.02462415 -0.0423436   0. ],Linear conversion factors between loads [Fx-z; Mx-z] and axial stress in the lower spar cap at blade root
    # drivese.lss_axial_load2stress,m**2,[1.0976203  0.         0.         0.         1.56430446 1.56430446],
    # drivese.lss_shear_load2stress,m**2,[0.         1.77770979 1.77770979 0.78215223 0.         0.        ],
    # towerse.member.axial_load2stress,m**2,"[[0.         0.         0.80912515 0.32621673 0.32621673 0.        ]
    # towerse.member.shear_load2stress,m**2,"[[1.33022717 1.33022717 0.         0.         0.         0.16310837]
    # From Garrett 2023-11-20 for the blade root: [[0.        , 0.        , 0.61747941, 0.49381254, 0.49381254,        0.        ]]
    #
    # These are the values we are interested in (JJ pointed out the positions in the array, except for the blade bending)
    # blade bending 0.49381254  # from cylinder/blade axial
    # lss bending   1.56430446  # from lss axial
    # tower bending 0.32621673  # from tower axial
    # tower torsion 0.16310837  # from tower shear

    
    # Ultimate stress values from https://github.com/IEAWindTask37/IEA-15-240-RWT/blob/master/WT_Ontology/IEA-15-240-RWT.yaml#L746
    if var == 'RootMc1_[kNm]':  # blade root bending
        slope = 10
        Sult = 1047e6  # compression. Getting the lowest of compression/tension
        load2stress = 0.49381254
    elif var == 'TwrBsMt_[kNm]': # tower bending
        slope = 4
        Sult = 450e6
        load2stress = 0.32621673
    elif var == 'LSSGagMa_[kNm]': # lss bending 
        slope = 4
        Sult = 814e6
        load2stress = 1.56430446
    else:
        raise ValueError('Variable not recognized')

    # Initialize variable
    full_del_withgoodman = np.zeros((len(ds.theta), len(ds.seed), len(ds.turbine), len(ds.wdir), len(ds.yawCase)))
    full_del_woutgoodman = np.zeros((len(ds.theta), len(ds.seed), len(ds.turbine), len(ds.wdir), len(ds.yawCase)))
    full_damage          = np.zeros((len(ds.theta), len(ds.seed), len(ds.turbine), len(ds.wdir), len(ds.yawCase)))
    full_fatpack         = np.zeros((len(ds.theta), len(ds.seed), len(ds.turbine), len(ds.wdir), len(ds.yawCase)))

    # Loop through everything and compute DEL
    for i, theta in enumerate(ds.theta):
        for j, seed in enumerate(ds.seed):
            for k, turb in enumerate(ds.turbine):
                #print(f'Processing theta {i+1}/{len(ds.theta)}, seed {j+1}/{len(ds.seed)}, turb {k+1}/{len(ds.turbine)}, all {len(ds.wdir)} wdir, all {len(ds.yawCase)} yawCases.    ', end='\r',flush=True)
                for l, wdir in enumerate(ds.wdir):
                    print(f'Processing theta {i+1}/{len(ds.theta)}, seed {j+1}/{len(ds.seed)}, turb {k+1}/{len(ds.turbine)}, wdir {l+1}/{len(ds.wdir)}, all {len(ds.yawCase)} yawCases.    ', end='\r', flush=True)
                    for m, yaw in enumerate(ds.yawCase):
                        ts = ds.sel(wdir=wdir, yawCase=yaw, seed=seed, turbine=turb, theta=theta).squeeze()[var]*1e3  # convert kNm to Nm
                        elapsed = (ts.time[-1]-ts.time[0]).values
                        
                        DEL_withgoodman, damage, fatpack_rainflow_successful = compute_del(ts, elapsed, lifetime, load2stress, slope, Sult, Sc=Sc, rainflow_bins=rainflow_bins, return_damage=True, goodman_correction=True)
                        DEL_woutgoodman                                      = compute_del(ts, elapsed, lifetime, load2stress, slope, Sult, Sc=Sc, rainflow_bins=rainflow_bins, return_damage=False, goodman_correction=False)
                        
                        full_del_withgoodman[i,j,k,l,m] = DEL_withgoodman
                        full_del_woutgoodman[i,j,k,l,m] = DEL_woutgoodman
                        full_damage[i,j,k,l,m] = damage
                        full_fatpack[i,j,k,l,m] = fatpack_rainflow_successful
                        

    ds[f'DEL_withgoodman_[Nm]_{var}']       = (('theta', 'seed', 'turbine', 'wdir', 'yawCase'), full_del_withgoodman)
    ds[f'DEL_woutgoodman_[Nm]_{var}']       = (('theta', 'seed', 'turbine', 'wdir', 'yawCase'), full_del_woutgoodman)
    ds[f'damage_{var}']          = (('theta', 'seed', 'turbine', 'wdir', 'yawCase'), full_damage)
    ds[f'fatpack_success_{var}'] = (('theta', 'seed', 'turbine', 'wdir', 'yawCase'), full_fatpack)
    
    return ds




def calcDEL_nontheta (ds, var):
    
    # Set constants
    lifetime = 25       #  Design lifetime of the component / material in years
    #load2stress = 1     #  Linear scaling coefficient to convert an applied load to stress such that S = load2stress * L
    #slope = 10          #  Wohler exponent in the traditional SN-curve of S = A * N ^ -(1/m) (rthedin: 4 for tower, 10 for blades)
    #Sult=6e8           #  Ultimate stress for use in Goodman equivalent stress calculation
    Sc =0               #  Stress-axis intercept of log-log S-N Wohler curve. Taken as ultimate stress unless specified
    rainflow_bins = 100
    
    # Ultimate stress values from https://github.com/IEAWindTask37/IEA-15-240-RWT/blob/master/WT_Ontology/IEA-15-240-RWT.yaml#L746
    if var =='RootMzc1_[kNm]':
        raise NotImplementedError('Blade root torsional moment not implemented')
    elif var == 'TwrBsMzt_[kNm]': # tower torsional 
        slope = 4
        Sult = 450e6
        load2stress = 0.16310837
    else:
        raise ValueError('Variable not recognized')

    # Initialize variable
    full_del_withgoodman = np.zeros((len(ds.seed), len(ds.turbine), len(ds.wdir), len(ds.yawCase)))
    full_del_woutgoodman = np.zeros((len(ds.seed), len(ds.turbine), len(ds.wdir), len(ds.yawCase)))
    full_damage  = np.zeros((len(ds.seed), len(ds.turbine), len(ds.wdir), len(ds.yawCase)))
    full_fatpack = np.zeros((len(ds.seed), len(ds.turbine), len(ds.wdir), len(ds.yawCase)))

    # Loop through everything and compute DEL
    for i, seed in enumerate(ds.seed):
        for j, turb in enumerate(ds.turbine):
            for k, wdir in enumerate(ds.wdir):
                print(f'Processing seed {i+1}/{len(ds.seed)}, turb {j+1}/{len(ds.turbine)}, wdir {k+1}/{len(ds.wdir)}, all {len(ds.yawCase)} yawCases.    ', end='\r', flush=True)
                for l, yaw in enumerate(ds.yawCase):
                    ts = ds.sel(wdir=wdir, yawCase=yaw, seed=seed, turbine=turb).squeeze()[var]*1e3  # convert kNm to Nm
                    elapsed = (ts.time[-1]-ts.time[0]).values
                    
                    DEL_withgoodman, damage, fatpack_rainflow_successful = compute_del(ts, elapsed, lifetime, load2stress, slope, Sult, Sc=Sc, rainflow_bins=rainflow_bins, return_damage=True, goodman_correction=True)
                    DEL_woutgoodman                                      = compute_del(ts, elapsed, lifetime, load2stress, slope, Sult, Sc=Sc, rainflow_bins=rainflow_bins, return_damage=False, goodman_correction=False)

                    full_del_withgoodman[i,j,k,l] = DEL_withgoodman
                    full_del_woutgoodman[i,j,k,l] = DEL_woutgoodman
                    full_damage[i,j,k,l] = damage
                    full_fatpack[i,j,k,l] = fatpack_rainflow_successful
                    

    ds[f'DEL_withgoodman_[Nm]_{var}'] = (('seed', 'turbine', 'wdir', 'yawCase'), full_del_withgoodman)
    ds[f'DEL_woutgoodman_[Nm]_{var}'] = (('seed', 'turbine', 'wdir', 'yawCase'), full_del_woutgoodman)
    ds[f'damage_{var}']          = (('seed', 'turbine', 'wdir', 'yawCase'), full_damage)
    ds[f'fatpack_success_{var}'] = (('seed', 'turbine', 'wdir', 'yawCase'), full_fatpack)
    
    return ds



