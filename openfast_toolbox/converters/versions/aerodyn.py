from openfast_toolbox.io.fast_input_file import FASTInputFile

def ad30_to_ad40(fileold, filenew=None, verbose=True, overwrite=False):
    """
    Convert AD 3.0 file to AD 4.0
    NOTE: beta version.

    INPUTS:
     - fileold: filepath to old input file
     - filenew: filepath to new input file (can be the same as old)
                If None, and overwrite is false, a file with '_new' is written
     - overwrite: if True, filenew = fileold
    """
    if filenew is None:
        if overwrite:
            filenew = fileold
        else:
            filenew = fileold[:-4]+'_new.dat'
    if verbose:
        print('> Converting:', fileold)
        print('> to        :', filenew)
    # --- 
    f = FASTInputFile(fileold, IComment=[1])

    # --- Checking if conversion already done
    try:
        iExisting = [f.getID('Wake_Mod'), f.getID('BEM_Mod'), f.getID('Skew_Mod'), f.getID('SectAvg'), f.getID('AoA34')]
        print('[WARN] File already converted, skipping: ', fileold)
        # We still write
        f.write(filenew)
        return 
    except KeyError:
        pass

    # --- Extracting old data
    iSkewMod = f.getID('SkewMod')
    WakeMod       = f.pop('WakeMod')['value']
    SkewMod       = f.pop('SkewMod')['value']
    AFAeroMod     = f.pop('AFAeroMod')['value']
    UAMod         = f.pop('UAMod')['value']
    FrozenWake    = f.pop('FrozenWake')['value']
    DBEMTMod      = f.pop('DBEMT_Mod')['value']
    SkewModFactor = f.pop('SkewModFactor')['value']

    # --- Default
    BEM_Mod          = 1
    DBEMT_Mod        = 0
    AoA34            = True
    SectAvg          = False
    SectAvgWeighting = 1
    SectAvgNPoints   = 'default'
    Skew_Mod         = 0
    SkewMomCorr      = False
    SkewRedistr_Mod  = 1
    AoA34            = False
    UA_Mod           = 0

    # --- Logic
    Wake_Mod = {None:1, 0:0, 1:1, 2:1, 3:3}[WakeMod]
    Skew_Mod = {None:1, 0:-1, 1:0, 2:1}[SkewMod]
    UA_Mod = {None:0, 1:0, 2:UAMod}[AFAeroMod]
    if WakeMod==1:
        DBEMT_Mod=0
        BEM_Mod=1
    if WakeMod==2:
        DBEMT_Mod = DBEMTMod
    if FrozenWake is not None and FrozenWake:
        DBEMT_Mod=-1
    AoA34 = UA_Mod>0 # Legacy behavior if UA is on, use AoA34, if it's off use AoA QC


    # --- Changing file
    f.data[0]['value'] = '------- AERODYN INPUT FILE --------------------------------------------------------------------------'
#     f.data[1]['isComment'] = True
#     f.data[1]['value']     = (str(f.data[1]['value']) +' '+ str(f.data[1]['label']) + ' '+ str(f.data[1]['descr'])).strip()
    f.insertKeyValAfter('DTAero', 'Wake_Mod', Wake_Mod, 'Wake/induction model (switch) {0=none, 1=BEMT, 3=OLAF} [WakeMod cannot be 2 or 3 when linearizing]')

    #i = f.getID('Pvap') # NOTE might not exist for old files..
    i = f.getID('TipLoss')-2
    f.pop(i+1) # Remove 'BEMT comment'
    f.insertComment(i+1,'======  Blade-Element/Momentum Theory Options  ====================================================== [unused when WakeMod=0 or 3, except for BEM_Mod]')
    f.insertKeyVal (i+2, 'BEM_Mod', BEM_Mod, 'BEM model {1=legacy NoSweepPitchTwist, 2=polar} (switch) [used for all Wake_Mod to determine output coordinate system]')
    f.insertComment(i+3, '--- Skew correction')
    f.insertKeyVal (i+4, 'Skew_Mod'         , Skew_Mod     ,'Skew model {0=No skew model, -1=Remove non-normal component for linearization, 1=skew model active}')
    f.insertKeyVal (i+5, 'SkewMomCorr'      , False        ,'Turn the skew momentum correction on or off [used only when Skew_Mod=1]')
    f.insertKeyVal (i+6, 'SkewRedistr_Mod'  , 'default'    ,'Type of skewed-wake correction model (switch) {0=no redistribution, 1=Glauert/Pitt/Peters, default=1} [used only when Skew_Mod=1]')
    f.insertKeyVal (i+7, 'SkewRedistrFactor', SkewModFactor,'Constant used in Pitt/Peters skewed wake model {or "default" is 15/32*pi} (-) [used only when Skew_Mod=1 and SkewRedistr_Mod=1]')
    f.insertComment(i+8, '--- BEM algorithm ')

    i = f.getID('MaxIter')
    f.pop(i+1) # Remove 'DBEMT comment'
    f.insertComment(i+1, '--- Shear correction')
    f.insertKeyVal(i+2, 'SectAvg'          , SectAvg          , 'Use sector averaging (flag)')
    f.insertKeyVal(i+3, 'SectAvgWeighting' , SectAvgWeighting , 'Weighting function for sector average  {1=Uniform, default=1} within a sector centered on the blade (switch) [used only when SectAvg=True] ')
    f.insertKeyVal(i+4, 'SectAvgNPoints'   , SectAvgNPoints   , 'Number of points per sectors (-) {default=5} [used only when SectAvg=True] ')
    f.insertKeyVal(i+5, 'SectAvgPsiBwd'    , 'default'        , 'Backward azimuth relative to blade where the sector starts (<=0) {default=-60} (deg) [used only when SectAvg=True]')
    f.insertKeyVal(i+6, 'SectAvgPsiFwd'    , 'default'        , 'Forward azimuth relative to blade where the sector ends (>=0) {default=60} (deg) [used only when SectAvg=True]')
    f.insertComment(i+7, '--- Dynamic wake/inflow')
    f.insertKeyVal (i+8, 'DBEMT_Mod', DBEMT_Mod, 'Type of dynamic BEMT (DBEMT) model {0=No Dynamic Wake, -1=Frozen Wake for linearization, 1:constant tau1, 2=time-dependent tau1, 3=constant tau1 with continuous formulation} (-)')
    f.data[f.getID('tau1_const')]['descr']= 'Time constant for DBEMT (s) [used only when DBEMT_Mod=1 or 3]'

    #i = f.getID('OLAFInputFileName')
    i = f.getID('FLookup')-2
    f.pop(i+1) # Remove 'BEDDOES comment'
    f.insertComment(i+1, '======  Unsteady Airfoil Aerodynamics Options  ====================================================')
    f.insertKeyVal (i+2, 'AoA34' , AoA34,  'Sample the angle of attack (AoA) at the 3/4 chord or the AC point {default=True} [always used]')
    f.insertKeyVal (i+3, 'UA_Mod', UA_Mod, 'Unsteady Aero Model Switch (switch) {0=Quasi-steady (no UA), 2=B-L Gonzalez, 3=B-L Minnema/Pierce, 4=B-L HGM 4-states, 5=B-L HGM+vortex 5 states, 6=Oye, 7=Boeing-Vertol}')
    if verbose:
        print(' -------------- New AeroDyn inputs (with new meaning):')
        print('Wake_Mod         : {}'.format(Wake_Mod         ))
        print('BEM_Mod          : {}'.format(BEM_Mod          ))
        print('SectAvg          : {}'.format(SectAvg          ))
        print('SectAvgWeighting : {}'.format(SectAvgWeighting ))
        print('SectAvgNPoints   : {}'.format(SectAvgNPoints   ))
        print('DBEMT_Mod        : {}'.format(DBEMT_Mod        ))
        print('Skew_Mod         : {}'.format(Skew_Mod         ))
        print('SkewMomCorr      : {}'.format(SkewMomCorr      ))
        print('SkewRedistr_Mod  : {}'.format(SkewRedistr_Mod  ))
        print('AoA34            : {}'.format(AoA34            ))
        print('UA_Mod           : {}'.format(UA_Mod           ))
        print('--------------- Old AeroDyn inputs:')
        print('WakeMod:   {}'.format(WakeMod))
        print('SkewMod:   {}'.format(SkewMod))
        print('AFAeroMod: {}'.format(AFAeroMod))
        print('FrozenWake:{}'.format(FrozenWake))
        print('DBEMT_Mod  {}'.format(DBEMTMod))
        print('UAMod:     {}'.format(UAMod))
        print('-----------------------------------------------------')

    # --- Write new file
    f.write(filenew)
    return f


if __name__ == '__main__':
    fileold='C:/W0/Work-Old/2018-NREL/BAR-Cone-BEM/openfast-ad-neo/reg_tests/r-test/modules/aerodyn/py_ad_B1n2_OLAF/AD_old.dat'
    filenew='C:/W0/Work-Old/2018-NREL/BAR-Cone-BEM/openfast-ad-neo/reg_tests/r-test/modules/aerodyn/py_ad_B1n2_OLAF/AD_conv.dat'
    ad30_to_ad40(fileold, filenew)
