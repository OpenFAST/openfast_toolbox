import numpy as np
import pandas as pd
import xarray as xr
import os, sys
from multiprocessing import Pool
from itertools import repeat

from openfast_toolbox.io import TurbSimFile, FASTOutputFile, VTKFile, FASTInputFile

def _get_fstf_filename(caseobj):
    if hasattr(caseobj, 'outputFFfilename'):
        return os.path.splitext(caseobj.outputFFfilename)[0]
    else:
        return 'FFarm_mod'


def readTurbineOutputPar(caseobj, dt_openfast, dt_processing, saveOutput=True, output='zarr',
                         iCondition=0, fCondition=-1, iCase=0, fCase=-1, iSeed=0, fSeed=-1, iTurbine=0, fTurbine=-1,
                         nCores=36):
    '''
    Inputs
    ------
    output: str
        Either 'nc' or 'zarr'. Determines the output format
    Zero-indexed initial and final values for conditions, cases, seeds, and turbines. 


    '''


    #from multiprocessing import set_start_method
    #try:
    #    set_start_method("spawn")
    #except RuntimeError:
    #    print(f'Fell into RunTime error on `set_start_method("spawn")`. Continuing..\n')


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

    if output=='zarr' and os.path.isdir(outputzarr) and saveOutput:
       print(f'Output file {zarrstore} exists. Loading it.')
       comb_ds = xr.open_zarr(outputzarr)
       return comb_ds
    if output=='nc' and os.path.isfile(outputnc) and saveOutput:
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
                                           repeat(False),       # saveOutput
                                           repeat(output),      # output
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

    # Trying this out
    print('trying to close the pool. does this seem to work better on notebooks?')
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

def readTurbineOutput(caseobj, dt_openfast, dt_processing=1, saveOutput=True, output='zarr',
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
    #else:
    #    fCondition += 1  # The user sets the last desired condition. This if for the np.arange.
    if fCondition-iCondition <= 0:
        raise ValueError (f'Final condition to read needs to be larger than initial.')

    if fCase==-1:
        fCase = caseobj.nCases
    #else:
        #fCase += 1  # The user sets the last desired case. This if for the np.arange.
    if fCase-iCase <= 0:
        raise ValueError (f'Final case to read needs to be larger than initial.')

    if fSeed==-1:
        fSeed = caseobj.nSeeds
    #else:
    #    fSeed += 1 # The user sets the last desired seed. This is for the np.arange
    if fSeed-iSeed <= 0:
        raise ValueError (f'Final seed to read needs to be larger than initial.')

    if fTurbine==-1:
        fTurbine = caseobj.nTurbines
    #else:
    #    fTurbine += 1  # The user sets the last desired turbine. This if for the np.arange.
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
        if output == 'zarr':  turbs = xr.open_zarr(outputzarr) 
        elif output == 'nc':  turbs = xr.open_dataset(outputnc)
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
                        ff_file = os.path.join(caseobj.path, caseobj.condDirList[cond], caseobj.caseDirList[case], f'Seed_{seed}', f'{_get_fstf_filename(caseobj)}.T{t+1}.outb')
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
                    turbs_t = xr.concat(turbs_t,dim='turbine')
                    turbs_seed.append(turbs_t)
                turbs_seed = xr.concat(turbs_seed,dim='seed')
                turbs_case.append(turbs_seed)
            turbs_case = xr.concat(turbs_case,dim='case')
            turbs_cond.append(turbs_case)
        turbs_cond = xr.concat(turbs_cond, dim='cond')

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






def readFFPlanesPar(caseobj, sliceToRead, verbose=False, saveOutput=True, iCondition=0, fCondition=-1, iCase=0, fCase=-1, iSeed=0, fSeed=-1, itime=0, ftime=-1, skiptime=1, nCores=36, outputformat='zarr'):

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


    zarrstore = f'ds_{sliceToRead}Slices_temp_cond{iCondition}_{fCondition}_case{iCase}_{fCase}_seed{iSeed}_{fSeed}.zarr'
    outputzarr = os.path.join(caseobj.path, zarrstore)

    if os.path.isdir(outputzarr) and saveOutput:
       print(f'Output file {zarrstore} exists. Attempting to read it..')
       comb_ds = xr.open_zarr(outputzarr)
       return comb_ds


    # We will loop in the variable that we have more entries. E.g. if we have several condition, and only one case,
    # then we will loop on the conditions. Analogous, if we have single conditions with many cases, we will loop on
    # cases. Here we figure out where the loop will take place and set the appropriate number of cores to be used.
    if fCondition-iCondition > fCase-iCase:
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

        # Split all the cases in arrays of roughly the same size
        chunks =  np.array_split(range(iCondition,fCondition), nCores)
        # Now, get the beginning and end of each separate chunk
        iCond_list = [i[0]    for i in chunks]
        fCond_list = [i[-1]+1 for i in chunks] 
        print(f'iCond_list is {iCond_list}')
        print(f'fCond_list is {fCond_list}')

        # For generality, we create the list of cases as a repeat
        iCase_list = repeat(iCase)
        fCase_list = repeat(fCase)

    elif loopOn == 'case':

        # Split all the cases in arrays of roughly the same size
        chunks =  np.array_split(range(iCase,fCase), nCores)
        # Now, get the beginning and end of each separate chunk
        iCase_list = [i[0]    for i in chunks]
        fCase_list = [i[-1]+1 for i in chunks] 
        print(f'iCase_list is {iCase_list}')
        print(f'fCase_list is {fCase_list}')

        # For generality, we create the list of cond as a repeat
        iCond_list = repeat(iCondition)
        fCond_list = repeat(fCondition)

    else:
        raise ValueError (f"This shouldn't occur. Not sure what went wrong.")


    p = Pool()
    ds_ = p.starmap(readFFPlanes, zip(repeat(caseobj),          # caseobj
                                      repeat(sliceToRead),      # slicesToRead
                                      repeat(verbose),          # verbose
                                      repeat(False),            # saveOutput
                                      iCond_list,               # iCondition
                                      fCond_list,               # fCondition
                                      iCase_list,               # iCase
                                      fCase_list,               # fCase
                                      repeat(iSeed),            # iSeed
                                      repeat(fSeed),            # fSeed
                                      repeat(itime),            # itime
                                      repeat(ftime),            # ftime
                                      repeat(skiptime),         # skiptime
                                      repeat(outputformat)      # outputformat
                                     )
                                  )

    print(f'Done reading all output. Concatenating the arrays')
    comb_ds = xr.combine_by_coords(ds_)

    if saveOutput:
        pass
        print('Done concatenating. Saving zarr file.')
        comb_ds.to_zarr(outputzarr)

    print('Finished.')

    return comb_ds









def readFFPlanes(caseobj, slicesToRead=['x','y','z'], verbose=False, saveOutput=True, iCondition=0, fCondition=-1, iCase=0, fCase=-1, iSeed=0, fSeed=-1, itime=0, ftime=-1, skiptime=1, outformat='zarr'):
    '''
    Read and process FAST.Farm planes into xarrays.

    INPUTS
    ======
    i<quant>, f<quant>: int
        Initial and end index of <quant> to read
    itime: int
        Initial timestep to open and read
    ftime: int
        Final timestep to open and read
    skiptime: int
        Read at every skiptime timestep. Used when data is too fine and not needed.

    '''

    if fCondition==-1:
        fCondition = caseobj.nConditions
    #else:
    #    fCondition += 1  # The user sets the last desired condition. This if for the np.arange.
    if fCondition-iCondition <= 0:
        raise ValueError (f'Final condition to read needs to be larger than initial.')

    if fCase==-1:
        fCase = caseobj.nCases
    #else:
        #fCase += 1  # The user sets the last desired case. This if for the np.arange.
    if fCase-iCase <= 0:
        raise ValueError (f'Final case to read needs to be larger than initial.')

    if fSeed==-1:
        fSeed = caseobj.nSeeds
    #else:
    #    fSeed += 1 # The user sets the last desired seed. This is for the np.arange
    if fSeed-iSeed <= 0:
        raise ValueError (f'Final seed to read needs to be larger than initial.')

    if skiptime<1:
        raise ValueError (f'Skiptime should be 1 or greater. If 1, no slices will be skipped.')

    print(f'Requesting to save {slicesToRead} slices')

    #if nConditions is None:
    #    nConditions = caseobj.nConditions
    #else:
    #    if nConditions > caseobj.nConditions:
    #        print(f'WARNING: Requested {nConditions} conditions, but only {caseobj.nConditions} are available. Reading {caseobj.nConditions} conditions')
    #        nConditions = caseobj.nConditions

    #if nCases is None:
    #    nCases = caseobj.nCases
    #else:
    #    if nCases > caseobj.nCases:
    #        print(f'WARNING: Requested {nCases} cases, but only {caseobj.nCases} are available. Reading {caseobj.nCases} cases')
    #        nCases = caseobj.nCases

    #if nSeeds is None:
    #    nSeeds = caseobj.nSeeds
    #else:
    #    if nSeeds > caseobj.nSeeds:
    #        print(f'WARNING: Requested {nSeeds} seeds, but only {caseobj.nSeeds} are available. Reading {caseobj.nSeeds} seeds')
    #        nSeeds = caseobj.nSeeds
    
    # Read all VTK output for each plane and save an nc files for each normal. Load files if present.
    for slices in slicesToRead:
    
        storefile = f'ds_{slices}Slices_temp_cond{iCondition}_{fCondition}_case{iCase}_{fCase}_seed{iSeed}_{fSeed}'
        if outformat == 'zarr':
            outputfile = os.path.join(caseobj.path, f'{storefile}.zarr')
        elif outformat == 'nc':
            outputfile = os.path.join(caseobj.path, f'{storefile}.nc')

        if os.path.isdir(outputfile): # it's zarr
            if len(slicesToRead) > 1:
                print(f"!! WARNING: Asked for multiple slices. Returning only the first one, {slices}\n",
                      f"           To load other slices, request `slicesToRead='y'`")
            print(f'Processed output file {outputfile} for slice {slices} exists. Loading it.')
            # Data already processed. Reading output
            Slices = xr.open_zarr(outputfile)
            return Slices

        elif os.path.isfile(outputfile):  # it's nc
            if len(slicesToRead) > 1:
                print(f"!! WARNING: Asked for multiple slices. Returning only the first one, {slices}\n",
                      f"           To load other slices, request `slicesToRead='y'`")
            print(f'Processed output file {outputfile} for slice {slices} exists. Loading it.')
            # Data already processed. Reading output
            Slices = xr.open_dataset(outputfile)
            return Slices

            
        else:
            
            # This for-loop is due to memory allocation requirements
            #print(f'Processing slices normal in the {slices} direction...')
            Slices_cond = []
            for cond in np.arange(iCondition, fCondition, 1):
                Slices_case = []
                for case in np.arange(iCase, fCase, 1):
                    Slices_seed = []
                    for seed in np.arange(iSeed, fSeed, 1):
                        seedPath = os.path.join(caseobj.path, caseobj.condDirList[cond], caseobj.caseDirList[case], f'Seed_{seed}')

                        # Read FAST.Farm input to determine outputs
                        ff_file = FASTInputFile(os.path.join(seedPath,f'{_get_fstf_filename(caseobj)}.fstf'))

                        tmax          = ff_file['TMax']
                        NOutDisWindXY = ff_file['NOutDisWindXY']
                        OutDisWindZ   = ff_file['OutDisWindZ']
                        NOutDisWindYZ = ff_file['NOutDisWindYZ']
                        OutDisWindX   = ff_file['OutDisWindX']
                        NOutDisWindXZ = ff_file['NOutDisWindXZ']
                        WrDisDT       = ff_file['WrDisDT']

                        # Determine number of output VTKs
                        nOutputTimes = int(np.floor(tmax/WrDisDT))

                        # Determine number of output digits for reading
                        ndigitsplane = max(len(str(max(NOutDisWindXY,NOutDisWindXZ,NOutDisWindYZ))), 3)
                        ndigitstime = len(str(nOutputTimes)) + 1  # this +1 is experimental. I had 1800 planes and got 5 digits.
                        # If this breaks again and I need to come here to fix, I need to ask Andy how the amount of digits is determined.


                        # Determine how many snapshots to read depending on input
                        if ftime==-1:
                            ftime=nOutputTimes
                        elif ftime>nOutputTimes:
                            raise ValueError (f'Final time step requested ({ftime}) is greater than the total available ({nOutputTimes})')


                        # Print info
                        print(f'Processing {slices} slice: Condition {cond}, Case {case}, Seed {seed}, snapshot {itime} to {ftime} ({nOutputTimes} available)')

                        if slices == 'z':
                            # Read Low-res z-planes
                            Slices=[]
                            for zplane in range(NOutDisWindXY):
                                Slices_t=[]
                                #for t in range(nOutputTimes):
                                for t in np.arange(itime,ftime,skiptime):
                                    file = f'{_get_fstf_filename(caseobj)}.Low.DisXY{zplane+1:0{ndigitsplane}d}.{t:0{ndigitstime}d}.vtk'
                                    if verbose: print(f'Reading z plane {zplane} for time step {t}: \t {file}')

                                    vtk = VTKFile(os.path.join(seedPath, 'vtk_ff', file))
                                    ds = readAndCreateDataset(vtk, caseobj, cond=cond, case=case, seed=seed, t=t, WrDisDT=WrDisDT)

                                    Slices_t.append(ds)
                                Slices_t = xr.concat(Slices_t,dim='time')
                                Slices.append(Slices_t)
                            Slices = xr.concat(Slices,dim='z')

                        elif slices == 'y':
                            # Read Low-res y-planes
                            Slices=[]
                            for yplane in range(NOutDisWindXZ):
                                Slices_t=[]
                                #for t in range(nOutputTimes):
                                for t in np.arange(itime,ftime,skiptime):
                                    file = f'{_get_fstf_filename(caseobj)}.Low.DisXZ{yplane+1:0{ndigitsplane}d}.{t:0{ndigitstime}d}.vtk'
                                    if verbose: print(f'Reading y plane {yplane} for time step {t}: \t {file}')

                                    vtk = VTKFile(os.path.join(seedPath, 'vtk_ff', file))
                                    ds = readAndCreateDataset(vtk, caseobj, cond=cond, case=case, seed=seed, t=t, WrDisDT=WrDisDT)

                                    Slices_t.append(ds)
                                Slices_t = xr.concat(Slices_t,dim='time')
                                Slices.append(Slices_t)
                            Slices = xr.concat(Slices,dim='y')

                        elif slices == 'x':
                            # Read Low-res x-planes
                            Slices=[]
                            for xplane in range(NOutDisWindYZ):
                                Slices_t=[]
                                print(f'Processing {slices} slice: Condition {cond}, Case {case}, Seed {seed}, x plane {xplane}')
                                #for t in range(nOutputTimes):
                                for t in np.arange(itime,ftime,skiptime):
                                    file = f'{_get_fstf_filename(caseobj)}.Low.DisYZ{xplane+1:0{ndigitsplane}d}.{t:0{ndigitstime}d}.vtk'
                                    if verbose: print(f'Reading x plane {xplane} for time step {t}: \t {file}')

                                    vtk = VTKFile(os.path.join(seedPath, 'vtk_ff', file))
                                    ds = readAndCreateDataset(vtk, caseobj, cond=cond, case=case, seed=seed, t=t, WrDisDT=WrDisDT)

                                    Slices_t.append(ds)
                                Slices_t = xr.concat(Slices_t,dim='time')
                                Slices.append(Slices_t)
                            Slices = xr.concat(Slices,dim='x')

                        else:
                            raise ValueError(f'Only slices x, y, z are available. Slice {slices} was requested. Stopping.')

                        Slices_seed.append(Slices)
                    Slices_seed = xr.concat(Slices_seed, dim='seed')
                    Slices_case.append(Slices_seed)
                Slices_case = xr.concat(Slices_case, dim='case')
                Slices_cond.append(Slices_case)

            Slices = xr.concat(Slices_cond, dim='cond')

            if saveOutput:
                print(f'Saving {slices} slice file {outputfile}...')
                if outformat == 'zarr':
                    Slices.to_zarr(outputfile)
                elif outformat == 'nc':
                    Slices.to_netcdf(outputfile)

            if len(slicesToRead) == 1:
                # Single slice was requested
                print(f'Since single slice was requested, returning it.')
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
    
    channel_pairs = [['TwrBsMxt_[kNm]', 'TwrBsMyt_[kNm]'],
                     ['RootMxc1_[kNm]', 'RootMyc1_[kNm]'],
                     ['LSSGagMya_[kNm]','LSSGagMza_[kNm]']]
    channel_out   =  ['TwrBsMt_[kNm]', 'RootMc1_[kNm]', 'LSSGagMa_[kNm]']
    
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
            curr_load = load_0deg*cosd(curr_theta) + load_90deg*sind(curr_theta)
            all_load.append(curr_load.expand_dims('theta').assign_coords({'theta': [curr_theta]}))
                            
        all_load = xr.concat(all_load, dim='theta').to_dataset(name=channel_out[p])
        
        turbs = xr.merge([turbs,all_load])
    
    return turbs



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



