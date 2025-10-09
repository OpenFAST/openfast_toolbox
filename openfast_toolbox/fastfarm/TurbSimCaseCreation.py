import numpy as np
from openfast_toolbox.io import FASTInputFile

class TSCaseCreation:
    
    def __init__(self, D, HubHt, Vhub, TI, PLexp, x, y, z=None, zbot=1.0, cmax=5.0, fmax=5.0,
                 Cmeander=1.9, boxType='highres', extent=None,
                 ds_low=None, ds_high=None, dt_low=None, dt_high=None, mod_wake=1):
        """
        Instantiate the object. 
        
        Parameters
        ==========
        D: float
            Rotor diameter (m)
        HubHt: float
            Turbine hub height (m)
        Vhub: float
            Mean wind speed at hub height (m/s)
        TI: float
            Turbulence intensity at hub height
        PLexp: float
            Power law exponent for shear (-)
        x, y, z: float,
            x-, y-, and z-location of turbine, respectively. If z is None, z is set to 0
        cmax: float
            Maximum blade chord (m). If not specified, set to NREL 5MW value.
        fmax: float
            Maximum excitation frequency (Hz). If not specified set to NREL 5MW tower value.
        boxType: str
            Box type, either 'lowres' or 'highres'. Sets the appropriate dt, dy, and
            dz for discretization. Defaults to `highres` for backward compatibility
        extent: float or list of floats [xmin, xmax, ymin, ymax, zabovehub]
            Extent of the box. If high-res, extent is the total length around individual
            turbines (in D), typically 1.2D. If low-res, a list of each indidual extent
            is expected. All values should be positive.
        ds: float
            Spatial resolution of the requested box. Optional, and defaults to the value
            given by the modeling guidances
        dt: float
            Temporal resolution of the requested box. Optional, and defaults to the value
            given by the modeling guidances
        mod_wake: int
            The wake model to be used. Recommendations for dt_low changes across models.
            Defaults to polar wake for backward compatibility.
        """

        # Perform some checks on the input
        if boxType not in ['lowres', 'highres']:
            raise ValueError(f"Box type can only be 'lowres' or 'highres'. Received {boxType}.")
        if boxType == 'lowres':
            if extent is None or len(extent) != 5:
                raise ValueError(f'extent not defined properly for the low-res box. It should '\
                                 f'be [xmin, xmax, ymin, ymax, zabovehub]. Received {extent}.')
        elif boxType == 'highres':
            if extent is None or not isinstance(extent, (int,float)) or extent<1:
                raise ValueError(f'extent not defined properly for the high-res box. It should '\
                                 f'be given as a scalar larger than the rotor diameter. Received {extent}')

        # Set parameters for convenience
        self.D        = D
        self.RefHt    = HubHt
        self.URef     = Vhub
        self.TI       = TI
        self.PLexp    = PLexp
        self.cmax     = cmax
        self.fmax     = fmax
        self.Cmeander = Cmeander
        self.boxType  = boxType
        self.extent   = extent
        self.ds_low   = ds_low
        self.ds_high  = ds_high
        self.dt_low   = dt_low
        self.dt_high  = dt_high
        self.mod_wake = mod_wake

        # Turbine location
        self.turbLocs(x,y,z)
        # Discretization
        self.discretization()
        # Setup domain size
        self.domainSize(zbot=zbot)
        # Determine origin
        # self.originLoc()

    def turbLocs(self,x,y,z=None):
        """
        Specify turbine locations
        
        Parameters
        ==========
        x, y, z: float
            x-, y-, and z-location of turbine, respectively
        """
        self.x     = np.asarray(x)
        self.y     = np.asarray(y)
        if z is None:
            self.z     = np.asarray(y)*0
        else:
            self.z     = np.asarray(z)

    def discretization(self):
        '''
        Specify discretization for both the high-res and low-res boxes. Follows guidelines present at
        https://openfast.readthedocs.io/en/main/source/user/fast.farm/ModelGuidance.html#low-resolution-domain

        '''
        
        if self.boxType == 'lowres':
            self._discretization_lowres()
        elif self.boxType == 'highres':
            self._discretization_highres()
        else:
            raise ValueError("boxType can only be 'lowres' or 'highres'. Stopping.")

    def _discretization_lowres(self):

        # Temporal resolution for low-res
        if self.mod_wake == 1 and self.dt_low is None:
            self.dt_low = self.Cmeander*self.D/(10*self.URef)
        elif self.mod_wake == 2 and self.dt_low is None:
            self.dt_low = (self.D/15)/(2*self.URef)

        # Spatial resolution for low-res
        if self.ds_low is None:
            self.ds_low = self.Cmeander*self.D*self.URef/150

        self.dy = np.floor(self.ds_low/self.ds_high)*self.ds_high
        self.dz = np.floor(self.ds_low/self.ds_high)*self.ds_high
        self.dt = self.dt_low

    def _discretization_highres(self):

        # Temporal resolution for high-res
        if self.dt_high is None:
            self.dt_high = 1.0/(2.0*self.fmax)
        
        # Spatial resolution for high-res
        if self.ds_high is None:
            self.ds_high = self.cmax

        self.dy = self.ds_high
        self.dz = self.ds_high
        self.dt = self.dt_high


    def domainSize(self, zbot):
    
        # Set default
        self.ymin = None
        self.ymax = None

        if self.boxType == 'lowres':
            self.ymin = min(self.y) - self.extent[2]*self.D
            self.ymax = max(self.y) + self.extent[3]*self.D
            Zdist_Low = self.RefHt + self.extent[4]*self.D
            Ydist_Low = self.ymax - self.ymin

            self.ny = np.ceil(Ydist_Low/self.dy)+1
            self.nz = np.ceil(Zdist_Low/self.dz)+1

        elif self.boxType=='highres':
            Ydist_high = self.extent*self.D
            Zdist_high = self.RefHt + self.extent*self.D/2 - zbot
           
            self.ny = np.ceil(Ydist_high/self.dy)+1
            self.nz = np.ceil(Zdist_high/self.dz)+1
           
        else:
            raise ValueError("boxType can only be 'lowres' or 'highres'. Stopping.")

        # We need to make sure the number of points is odd.
        if self.ny%2 == 0:
            self.ny += 1
        if self.nz%2 == 0:
            self.nz += 1
        
        self.Width  = self.dy*(self.ny-1)
        self.Height = self.dz*(self.nz-1)
        Dgrid=min(self.Height,self.Width)

        # Set the hub height using half of the total grid height 
        self.HubHt_for_TS = zbot - 0.5*Dgrid + self.Height


    def originLoc(self):
        raise NotImplementedError
        


    def plotSetup(self, fig=None, ax=None):
        """
        Plot a figure showing the turbine locations and the extent of the turbulence box
        """
        if fig is None:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(6,5))
            ax  = fig.add_subplot(111,aspect="equal")
        xmin = min(self.x)-self.D
        xmax = max(self.x)+self.D

        # high-res boxes
        for wt in range(len(self.x)):
            ax.plot(self.x[wt],self.y[wt],'x',ms=8,mew=2,label=f"WT{wt+1}")

        # low-res box
        #         ax.plot([xmin,xmax,xmax,xmin,xmin],
        #                 [ymin,ymin,ymax,ymax,ymin],'--k',lw=2,label='Low')
        if self.ymin is not None:
            ax.axhline(self.ymin, ls='--', c='k', lw=2, label='Low')
        if self.ymax is not None:
            ax.axhline(self.ymax, ls='--', c='k', lw=2)

        ax.legend(bbox_to_anchor=(1.05,1.015),frameon=False)
        ax.set_xlabel("x-location [m]")
        ax.set_ylabel("y-location [m]")
        fig.tight_layout
        return fig, ax

    def writeTSFile(self, fileOut, fileIn=None, NewFile=False, tpath=None, tmax=50, turb=None, verbose=0):
        """
        Write a TurbSim primary input file.
        See WriteTSFile below.
        """
        if fileIn is None or fileIn in ['unused', '"unused"']:
            NewFile = True
            if verbose>0: print('[INFO] No TurbSim template provided, using a Dummy TurbSim file')
        WriteTSFile(fileIn=fileIn, fileOut=fileOut, params=self, NewFile=NewFile, tpath=tpath, tmax=tmax, turb=turb, verbose=verbose)


        
def WriteTSFile(fileOut, fileIn, params, NewFile=True, tpath=None, tmax=50, turb=None, verbose=0):
    """
    Write a TurbSim primary input file, 

    Parameters
    ==========
    tpath: string
        Path to base turbine location (.fst). Only used if NewFile is False
    boxType: string
        Box type, either 'lowres' or 'highres'. Writes the proper `TurbModel`
        If boxType=='highres', `turb` needs to be specified
    turb: int
        Turbine number to be printed on the time series file. Only needed
        if boxType='highres'

    params:
       object with fields:
            highres
            lowres
            nz
            ny
            dt
            [tmax]
            HubHt_for_TS
            Height
            Width
            TI
            RefHt
            URef
            PLexp
    """

    if params.boxType=='highres' and not isinstance(turb, int):
        raise ValueError("turb needs to be an integer when boxType is 'highres'")
    if params.boxType=='lowres' and turb is not None:
        print("WARNING: `turb` is not used when boxType is 'lowres'. Remove `turb` to dismiss this warning.")

    if NewFile:
        if verbose>1:  print(f'[INFO] Writing a new {fileOut} file from scratch')
        WriteDummyTSFile(fileOut)
        fileIn=fileOut

        # --- Writing FFarm input file from scratch
#         with open(fileOut, 'w') as f:
#             f.write( '--------TurbSim v2.00.* Input File------------------------\n')
#             f.write( 'for Certification Test #1 (Kaimal Spectrum, formatted FF files).\n')
#             f.write( '---------Runtime Options-----------------------------------\n')
#             f.write( 'False\tEcho\t\t- Echo input data to <RootName>.ech (flag)\n')
#             f.write( '123456\tRandSeed1\t\t- First random seed  (-2147483648 to 2147483647)\n')
#             f.write( 'RanLux\tRandSeed2\t\t- Second random seed (-2147483648 to 2147483647) for intrinsic pRNG, or an alternative pRNG: "RanLux" or "RNSNLW"\n')
#             f.write( 'False\tWrBHHTP\t\t- Output hub-height turbulence parameters in binary form?  (Generates RootName.bin)\n')
#             f.write( 'False\tWrFHHTP\t\t- Output hub-height turbulence parameters in formatted form?  (Generates RootName.dat)\n')
#             f.write( 'False\tWrADHH\t\t- Output hub-height time-series data in AeroDyn form?  (Generates RootName.hh)\n')
#             f.write( 'True\tWrADFF\t\t- Output full-field time-series data in TurbSim/AeroDyn form? (Generates RootName.bts)\n')
#             f.write( 'False\tWrBLFF\t\t- Output full-field time-series data in BLADED/AeroDyn form?  (Generates RootName.wnd)\n')
#             f.write( 'False\tWrADTWR\t\t- Output tower time-series data? (Generates RootName.twr)\n')
#             f.write( 'False\tWrHAWCFF\t\t- Output full-field time-series data in HAWC form?  (Generates RootName-u.bin, RootName-v.bin, RootName-w.bin, RootName.hawc)\n')
#             f.write( 'False\tWrFMTFF\t\t- Output full-field time-series data in formatted (readable) form?  (Generates RootName.u, RootName.v, RootName.w)\n')
#             f.write( 'False\tWrACT\t\t- Output coherent turbulence time steps in AeroDyn form? (Generates RootName.cts)\n')
#             #f.write(f'True\tClockwise\t\t- Clockwise rotation looking downwind? (used only for full-field binary files - not necessary for AeroDyn)\n')
#             f.write( '0\tScaleIEC\t\t- Scale IEC turbulence models to exact target standard deviation? [0=no additional scaling; 1=use hub scale uniformly; 2=use individual scales]\n')
#             f.write( '\n')
#             f.write( '--------Turbine/Model Specifications-----------------------\n')
#             f.write(f'{params.nz:.0f}\tNumGrid_Z\t\t- Vertical grid-point matrix dimension\n')
#             f.write(f'{params.ny:.0f}\tNumGrid_Y\t\t- Horizontal grid-point matrix dimension\n')
#             f.write(f'{params.dt:.6f}\tTimeStep\t\t- Time step [seconds]\n')
#             f.write(f'{tmax:.4f}\tAnalysisTime\t\t- Length of analysis time series [seconds] (program will add time if necessary: AnalysisTime = MAX(AnalysisTime, UsableTime+GridWidth/MeanHHWS) )\n')
#             f.write( '"ALL"\tUsableTime\t\t- Usable length of output time series [seconds] (program will add GridWidth/MeanHHWS seconds unless UsableTime is "ALL")\n')
#             f.write(f'{params.HubHt_for_TS:.3f}\tHubHt\t\t- Hub height [m] (should be > 0.5*GridHeight)\n')
#             f.write(f'{params.Height:.3f}\tGridHeight\t\t- Grid height [m]\n')
#             f.write(f'{params.Width:.3f}\tGridWidth\t\t- Grid width [m] (should be >= 2*(RotorRadius+ShaftLength))\n')
#             f.write( '0\tVFlowAng\t\t- Vertical mean flow (uptilt) angle [degrees]\n')
#             f.write( '0\tHFlowAng\t\t- Horizontal mean flow (skew) angle [degrees]\n')
#             f.write( '\n')
#             f.write( '--------Meteorological Boundary Conditions-------------------\n')
#             if params.boxType=='lowres':
#                 f.write( '"IECKAI"\tTurbModel\t\t- Turbulence model ("IECKAI","IECVKM","GP_LLJ","NWTCUP","SMOOTH","WF_UPW","WF_07D","WF_14D","TIDAL","API","IECKAI","TIMESR", or "NONE")\n')
#                 f.write( '"unused"\tUserFile\t\t- Name of the file that contains inputs for user-defined spectra or time series inputs (used only for "IECKAI" and "TIMESR" models)\n')
#             elif params.boxType=='highres':
#                 f.write( '"TIMESR"\tTurbModel\t\t- Turbulence model ("IECKAI","IECVKM","GP_LLJ","NWTCUP","SMOOTH","WF_UPW","WF_07D","WF_14D","TIDAL","API","USRINP","TIMESR", or "NONE")\n')
#                 f.write(f'"USRTimeSeries_T{turb}.txt"\tUserFile\t\t- Name of the file that contains inputs for user-defined spectra or time series inputs (used only for "IECKAI" and "TIMESR" models)\n')
#             else:
#                 raise ValueError("boxType can only be 'lowres' or 'highres'. Stopping.")
#             f.write( '1\tIECstandard\t\t- Number of IEC 61400-x standard (x=1,2, or 3 with optional 61400-1 edition number (i.e. "1-Ed2") )\n')
#             f.write(f'"{params.TI:.3f}\t"\tIECturbc\t\t- IEC turbulence characteristic ("A", "B", "C" or the turbulence intensity in percent) ("KHTEST" option with NWTCUP model, not used for other models)\n')
#             f.write( '"NTM"\tIEC_WindType\t\t- IEC turbulence type ("NTM"=normal, "xETM"=extreme turbulence, "xEWM1"=extreme 1-year wind, "xEWM50"=extreme 50-year wind, where x=wind turbine class 1, 2, or 3)\n')
#             f.write( '"default"\tETMc\t\t- IEC Extreme Turbulence Model "c" parameter [m/s]\n')
#             f.write( '"PL"\tWindProfileType\t\t- Velocity profile type ("LOG";"PL"=power law;"JET";"H2L"=Log law for TIDAL model;"API";"PL";"TS";"IEC"=PL on rotor disk, LOG elsewhere; or "default")\n')
#             f.write( '"unused"\tProfileFile\t\t- Name of the file that contains input profiles for WindProfileType="USR" and/or TurbModel="USRVKM" [-]\n')
#             f.write(f'{params.RefHt:.3f}\tRefHt\t\t- Height of the reference velocity (URef) [m]\n')
#             f.write(f'{params.URef:.3f}\tURef\t\t- Mean (total) velocity at the reference height [m/s] (or "default" for JET velocity profile) [must be 1-hr mean for API model; otherwise is the mean over AnalysisTime seconds]\n')
#             f.write( '350\tZJetMax\t\t- Jet height [m] (used only for JET velocity profile, valid 70-490 m)\n')
#             f.write(f'"{params.PLexp:.3f}"\tPLExp\t\t- Power law exponent [-] (or "default")\n')
#             f.write( '"default"\tZ0\t\t- Surface roughness length [m] (or "default")\n')
#             f.write( '\n')
#             f.write( '--------Non-IEC Meteorological Boundary Conditions------------\n')
#             f.write( '"default"\tLatitude\t\t- Site latitude [degrees] (or "default")\n')
#             f.write( '0.05\tRICH_NO\t\t- Gradient Richardson number [-]\n')
#             f.write( '"default"\tUStar\t\t- Friction or shear velocity [m/s] (or "default")\n')
#             f.write( '"default"\tZI\t\t- Mixing layer depth [m] (or "default")\n')
#             f.write( '"default"\tPC_UW\t\t- Hub mean u\'w\' Reynolds stress [m^2/s^2] (or "default" or "none")\n')
#             f.write( '"default"\tPC_UV\t\t- Hub mean u\'v\' Reynolds stress [m^2/s^2] (or "default" or "none")\n')
#             f.write( '"default"\tPC_VW\t\t- Hub mean v\'w\' Reynolds stress [m^2/s^2] (or "default" or "none")\n')
#             f.write( '\n')
#             f.write( '--------Spatial Coherence Parameters----------------------------\n')
#             f.write( '"IEC"\tSCMod1\t\t- u-component coherence model ("GENERAL","IEC","API","NONE", or "default")\n')
#             f.write( '"IEC"\tSCMod2\t\t- v-component coherence model ("GENERAL","IEC","NONE", or "default")\n')
#             f.write( '"IEC"\tSCMod3\t\t- w-component coherence model ("GENERAL","IEC","NONE", or "default")\n')
#             f.write(f'"12.0 {0.12/(8.1*42):.8f}"\tInCDec1\t- u-component coherence parameters for general or IEC models [-, m^-1] (e.g. "10.0  0.3e-3" in quotes) (or "default")\n')
#             f.write(f'"12.0 {0.12/(8.1*42):.8f}"\tInCDec2\t- v-component coherence parameters for general or IEC models [-, m^-1] (e.g. "10.0  0.3e-3" in quotes) (or "default")\n')
#             f.write(f'"12.0 {0.12/(8.1*42):.8f}"\tInCDec3\t- w-component coherence parameters for general or IEC models [-, m^-1] (e.g. "10.0  0.3e-3" in quotes) (or "default")\n')
#             f.write( '"0.0"\tCohExp\t\t- Coherence exponent for general model [-] (or "default")\n')
#             f.write(f'\n')
#             f.write(f'--------Coherent Turbulence Scaling Parameters-------------------\n')
#             f.write( '".\\unused\t\t- Name of the path where event data files are located\n')
#             f.write( '"random"\tCTEventFile\t\t- Type of event files ("LES", "DNS", or "RANDOM")\n')
#             f.write( 'true\tRandomize\t\t- Randomize the disturbance scale and locations? (true/false)\n')
#             f.write( '1\tDistScl\t\t- Disturbance scale [-] (ratio of event dataset height to rotor disk). (Ignored when Randomize = true.)\n')
#             f.write( '0.5\tCTLy\t\t- Fractional location of tower centerline from right [-] (looking downwind) to left side of the dataset. (Ignored when Randomize = true.)\n')
#             f.write( '0.5\tCTLz\t\t- Fractional location of hub height from the bottom of the dataset. [-] (Ignored when Randomize = true.)\n')
#             f.write( '30\tCTStartTime\t\t- Minimum start time for coherent structures in RootName.cts [seconds]\n')
#             f.write( '\n')
#             f.write( '====================================================\n')
#             f.write( '! NOTE: Do not add or remove any lines in this file!\n')
#             f.write( '====================================================\n')
    if verbose>1: print(f'[INFO] Modifying {fileIn} to be {fileOut}')
    wt=0
    ts = FASTInputFile(fileIn)
    ts['NumGrid_Z']  = int(params.nz)
    ts['NumGrid_Y']  = int(params.ny)
    ts['TimeStep']   = round(params.dt, 6)
    ts['AnalysisTime'] = round(tmax, 4)
    ts['HubHt']      = round(params.HubHt_for_TS, 3)
    ts['GridHeight'] = round(params.Height, 3)
    ts['GridWidth']  = round(params.Width , 3)
    if params.boxType=='lowres':
        ts['TurbModel'] = '"IECKAI"'
        ts['UserFile'] = '"unused"'
    elif params.boxType=='highres':
        ts['TurbModel'] = '"TIMSR"'
        ts['UserFile'] = f'"USRTimeSeries_T{turb}.txt"'
    ts['IECTurbc']    = round(float(params.TI   ) , 3)
    ts['RefHt']       = round(float(params.RefHt) , 3)
    ts['URef']        = round(float(params.URef ) , 3)
    ts['PLExp']       = round(float(params.PLexp) , 3)
    ts.write(fileOut)



def WriteDummyTSFile(fileOut):
    with open(fileOut, 'w') as f:
        f.write('--------TurbSim v2.00.* Input File------------------------\n')
        f.write('for Certification Test #1 (Kaimal Spectrum, formatted FF files).\n')
        f.write('---------Runtime Options-----------------------------------\n')
        f.write('False\tEcho\t\t- Echo input data to <RootName>.ech (flag)\n')
        f.write('123456\tRandSeed1\t\t- First random seed  (-2147483648 to 2147483647)\n')
        f.write('RanLux\tRandSeed2\t\t- Second random seed (-2147483648 to 2147483647) for intrinsic pRNG, or an alternative pRNG: "RanLux" or "RNSNLW"\n')
        f.write('False\tWrBHHTP\t\t- Output hub-height turbulence parameters in binary form?  (Generates RootName.bin)\n')
        f.write('False\tWrFHHTP\t\t- Output hub-height turbulence parameters in formatted form?  (Generates RootName.dat)\n')
        f.write('False\tWrADHH\t\t- Output hub-height time-series data in AeroDyn form?  (Generates RootName.hh)\n')
        f.write('True\tWrADFF\t\t- Output full-field time-series data in TurbSim/AeroDyn form? (Generates RootName.bts)\n')
        f.write('False\tWrBLFF\t\t- Output full-field time-series data in BLADED/AeroDyn form?  (Generates RootName.wnd)\n')
        f.write('False\tWrADTWR\t\t- Output tower time-series data? (Generates RootName.twr)\n')
        f.write('False\tWrHAWCFF\t\t- Output full-field time-series data in HAWC form?  (Generates RootName-u.bin, RootName-v.bin, RootName-w.bin, RootName.hawc)\n')
        f.write('False\tWrFMTFF\t\t- Output full-field time-series data in formatted (readable) form?  (Generates RootName.u, RootName.v, RootName.w)\n')
        f.write('False\tWrACT\t\t- Output coherent turbulence time steps in AeroDyn form? (Generates RootName.cts)\n')
        f.write('0\tScaleIEC\t\t- Scale IEC turbulence models to exact target standard deviation? [0=no additional scaling; 1=use hub scale uniformly; 2=use individual scales]\n')
        f.write('\n')
        f.write('--------Turbine/Model Specifications-----------------------\n')
        f.write('10\tNumGrid_Z\t\t- Vertical grid-point matrix dimension\n')
        f.write('10\tNumGrid_Y\t\t- Horizontal grid-point matrix dimension\n')
        f.write('0.1\tTimeStep\t\t- Time step [seconds]\n')
        f.write('50.0\tAnalysisTime\t\t- Length of analysis time series [seconds] (program will add time if necessary: AnalysisTime = MAX(AnalysisTime, UsableTime+GridWidth/MeanHHWS) )\n')
        f.write('"ALL"\tUsableTime\t\t- Usable length of output time series [seconds] (program will add GridWidth/MeanHHWS seconds unless UsableTime is "ALL")\n')
        f.write('100.0\tHubHt\t\t- Hub height [m] (should be > 0.5*GridHeight)\n')
        f.write('50.0\tGridHeight\t\t- Grid height [m]\n')
        f.write('50.0\tGridWidth\t\t- Grid width [m] (should be >= 2*(RotorRadius+ShaftLength))\n')
        f.write('0\tVFlowAng\t\t- Vertical mean flow (uptilt) angle [degrees]\n')
        f.write('0\tHFlowAng\t\t- Horizontal mean flow (skew) angle [degrees]\n')
        f.write('\n')
        f.write('--------Meteorological Boundary Conditions-------------------\n')
        f.write('"IECKAI"\tTurbModel\t\t- Turbulence model ("IECKAI","IECVKM","GP_LLJ","NWTCUP","SMOOTH","WF_UPW","WF_07D","WF_14D","TIDAL","API","IECKAI","TIMESR", or "NONE")\n')
        f.write('"unused"\tUserFile\t\t- Name of the file that contains inputs for user-defined spectra or time series inputs (used only for "IECKAI" and "TIMESR" models)\n')
        f.write('1\tIECstandard\t\t- Number of IEC 61400-x standard (x=1,2, or 3 with optional 61400-1 edition number (i.e. "1-Ed2") )\n')
        f.write('"A"\t\tIECturbc\t\t- IEC turbulence characteristic ("A", "B", "C" or the turbulence intensity in percent) ("KHTEST" option with NWTCUP model, not used for other models)\n')
        f.write('"NTM"\tIEC_WindType\t\t- IEC turbulence type ("NTM"=normal, "xETM"=extreme turbulence, "xEWM1"=extreme 1-year wind, "xEWM50"=extreme 50-year wind, where x=wind turbine class 1, 2, or 3)\n')
        f.write('"default"\tETMc\t\t- IEC Extreme Turbulence Model "c" parameter [m/s]\n')
        f.write('"PL"\tWindProfileType\t\t- Velocity profile type ("LOG";"PL"=power law;"JET";"H2L"=Log law for TIDAL model;"API";"PL";"TS";"IEC"=PL on rotor disk, LOG elsewhere; or "default")\n')
        f.write('"unused"\tProfileFile\t\t- Name of the file that contains input profiles for WindProfileType="USR" and/or TurbModel="USRVKM" [-]\n')
        f.write('100\tRefHt\t\t- Height of the reference velocity (URef) [m]\n')
        f.write('10.0\tURef\t\t- Mean (total) velocity at the reference height [m/s] (or "default" for JET velocity profile) [must be 1-hr mean for API model; otherwise is the mean over AnalysisTime seconds]\n')
        f.write('350\tZJetMax\t\t- Jet height [m] (used only for JET velocity profile, valid 70-490 m)\n')
        f.write('"0.2"\tPLExp\t\t- Power law exponent [-] (or "default")\n')
        f.write('"default"\tZ0\t\t- Surface roughness length [m] (or "default")\n')
        f.write('\n')
        f.write('--------Non-IEC Meteorological Boundary Conditions------------\n')
        f.write('"default"\tLatitude\t\t- Site latitude [degrees] (or "default")\n')
        f.write('0.05\tRICH_NO\t\t- Gradient Richardson number [-]\n')
        f.write('"default"\tUStar\t\t- Friction or shear velocity [m/s] (or "default")\n')
        f.write('"default"\tZI\t\t- Mixing layer depth [m] (or "default")\n')
        f.write('"default"\tPC_UW\t\t- Hub mean u\'w\' Reynolds stress [m^2/s^2] (or "default" or "none")\n')
        f.write('"default"\tPC_UV\t\t- Hub mean u\'v\' Reynolds stress [m^2/s^2] (or "default" or "none")\n')
        f.write('"default"\tPC_VW\t\t- Hub mean v\'w\' Reynolds stress [m^2/s^2] (or "default" or "none")\n')
        f.write('\n')
        f.write('--------Spatial Coherence Parameters----------------------------\n')
        f.write('"IEC"\tSCMod1\t\t- u-component coherence model ("GENERAL","IEC","API","NONE", or "default")\n')
        f.write('"IEC"\tSCMod2\t\t- v-component coherence model ("GENERAL","IEC","NONE", or "default")\n')
        f.write('"IEC"\tSCMod3\t\t- w-component coherence model ("GENERAL","IEC","NONE", or "default")\n')
        f.write(f'"12.0 {0.12/(8.1*42):.8f}"\tInCDec1\t- u-component coherence parameters for general or IEC models [-, m^-1] (e.g. "10.0  0.3e-3" in quotes) (or "default")\n')
        f.write(f'"12.0 {0.12/(8.1*42):.8f}"\tInCDec2\t- v-component coherence parameters for general or IEC models [-, m^-1] (e.g. "10.0  0.3e-3" in quotes) (or "default")\n')
        f.write(f'"12.0 {0.12/(8.1*42):.8f}"\tInCDec3\t- w-component coherence parameters for general or IEC models [-, m^-1] (e.g. "10.0  0.3e-3" in quotes) (or "default")\n')
        f.write('"0.0"\tCohExp\t\t- Coherence exponent for general model [-] (or "default")\n')
        f.write('\n')
        f.write('--------Coherent Turbulence Scaling Parameters-------------------\n')
        f.write('"unused"\tCTEventPath\t\t- Name of the path where event data files are located\n')
        f.write('"random"\tCTEventFile\t\t- Type of event files ("LES", "DNS", or "RANDOM")\n')
        f.write('true\tRandomize\t\t- Randomize the disturbance scale and locations? (true/false)\n')
        f.write('1\tDistScl\t\t- Disturbance scale [-] (ratio of event dataset height to rotor disk). (Ignored when Randomize = true.)\n')
        f.write('0.5\tCTLy\t\t- Fractional location of tower centerline from right [-] (looking downwind) to left side of the dataset. (Ignored when Randomize = true.)\n')
        f.write('0.5\tCTLz\t\t- Fractional location of hub height from the bottom of the dataset. [-] (Ignored when Randomize = true.)\n')
        f.write('30\tCTStartTime\t\t- Minimum start time for coherent structures in RootName.cts [seconds]\n')
        f.write('\n')
        f.write('====================================================\n')
        f.write('! NOTE: Do not add or remove any lines in this file!\n')
        f.write('====================================================\n')



def writeTimeSeriesFile(fileOut,yloc,zloc,u,v,w,time, verbose=0):
    """
    Write a TurbSim primary input file, 
    """

    if verbose>0: print(f'Writing {fileOut}')
    # --- Writing TurbSim user-defined time series file
    with open(fileOut, 'w') as f:
        f.write( '--------------TurbSim v2.00.* User Time Series Input File-----------------------\n')
        f.write( '     Time series input from low-res turbsim run\n')
        f.write( '--------------------------------------------------------------------------------\n')
        f.write( '          3 nComp - Number of velocity components in the file\n')
        f.write( '          1 nPoints - Number of time series points contained in this file (-)\n')
        f.write( '          1 RefPtID - Index of the reference point (1-nPoints)\n')
        f.write( '     Pointyi Pointzi ! nPoints listed in order of increasing height\n')
        f.write( '       (m)     (m)\n')
        f.write(f'       {yloc:.5f}   {zloc:.5f}\n')
        f.write( '--------Time Series-------------------------------------------------------------\n')
        f.write( 'Elapsed Time         Point01u             Point01v           Point01w\n')
        f.write( '         (s)            (m/s)                (m/s)              (m/s)\n')
        for i in range(len(time)):
            f.write(f'\t{time[i]:.2f}\t\t\t  {u[i]:.5f}\t\t\t  {v[i]:.5f}\t\t\t {w[i]:.5f}\n')

