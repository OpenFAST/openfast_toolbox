import numpy as np
import os
import subprocess

from openfast_toolbox.fastfarm.FASTFarmCaseCreation import getMultipleOf
from openfast_toolbox.tools.strings import INFO, FAIL, OK, WARN, print_bold

class AMRWindSimulation:
    '''
    This class is used to help prepare sampling planes for an AMR-Wind
      simulation. The sampling planes will be used to generate inflow
      data for FAST.Farm simulations.
    Specifically, this class contains info from the AMR-Wind input file,
      and it carries out simple calculations about the AMR-Wind
      simulation
    For reference, see https://openfast.readthedocs.io/en/dev/source/user/fast.farm/ModelGuidance.html
    '''

    def __init__(self, wts:dict,
                    dt: float, prob_lo: tuple, prob_hi: tuple, 
                    n_cell: tuple, max_level: int, 
                    incflo_velocity_hh: tuple, 
                    postproc_name='sampling',
                    buffer_lr = [3,6,3,3,2],
                    buffer_hr = 0.6,
                    ds_hr = None, ds_lr = None,
                    dt_hr = None, dt_lr = None,
                    mod_wake = None,
                    level_lr = 0,
                    level_hr = -1, 
                    verbose=1):
        '''
        Values from the AMR-Wind input file
        Inputs:
        ------
          dt: scalar
              This should be a fixed dt value from the LES run
          prob_lo, prob_hi: tuple
              Extents of the AMR-Wind domain
          n_cell: tuple of integers
              Number of cells from AMR-Wind input file
          max_level: int
              Max level on the AMR-Wind grid. Used for high-res box placement
          incflo_velocity_hh: tuple
              Velocity vector, specifically at hub height
          buffer_lr: 5-position vector
              Buffer for [xmin, xmax, ymin, ymax, zmax] in low-res box, in D
          buffer_hr: scalar
              Buffer for all directions (constant) in high-res box, in D
          ds_hr, ds_lr: scalar
              Spatial resolution of the high-res and low-res boxes. No need to set
              ahead of time, but optional to fix when you know the resolution you need
          dt_hr, dt_lr: scalar
              Same as above, but temporal resolution
          mod_wake: int
              Wake formulations within FAST.Farm. 1:polar; 2:curl; 3:cartesian.
          level_lr: int
              Grid level used for low-resolution box placement. Useful if staggered
              grid approach is used for background mesh
          level_hr: int
              Grid level used for high-resolution boxes placement. Defaults to 
              highest level available

        '''

        # Process inputs
        self.wts                = wts
        self.dt                 = dt
        self.prob_lo            = prob_lo
        self.prob_hi            = prob_hi
        self.n_cell             = n_cell
        self.max_level          = max_level
        self.incflo_velocity_hh = incflo_velocity_hh
        self.postproc_name_lr   = f"{postproc_name}_lr"
        self.postproc_name_hr   = f"{postproc_name}_hr"
        self.buffer_lr          = buffer_lr
        self.buffer_hr          = buffer_hr
        self.ds_hr              = ds_hr
        self.ds_lr              = ds_lr
        self.dt_hr              = dt_hr
        self.dt_lr              = dt_lr
        self.mod_wake           = mod_wake
        self.level_lr           = level_lr
        self.level_hr           = level_hr
        self.verbose            = verbose

        # Placeholder variables, to be calculated by FFCaseCreation
        self.output_interval_lr = None
        self.output_interval_hr = None
        self.sampling_labels_lr = None
        self.sampling_labels_hr = None
        self.nx_lr = None
        self.ny_lr = None
        self.nz_lr = None
        self.xlow_lr = None
        self.xhigh_lr = None
        self.ylow_lr = None
        self.yhigh_lr = None
        self.zlow_lr = None
        self.zhigh_lr = None
        self.zoffsets_lr = None
        self.hr_domains = None

        # Run extra functions
        self._checkInputs()
        self._calc_simple_params()
        self._calc_sampling_params()

        # Get execution time 
        from datetime import datetime
        self.curr_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.commit = self._get_git_commit_hash()


    def _get_git_commit_hash(self):
        '''
        Return current git commit hash for this repository.
        '''
        try:
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            commit_hash = subprocess.check_output(['git', '-C', repo_root, 'rev-parse', '--short', 'HEAD'],
                                                  stderr=subprocess.DEVNULL, text=True).strip()
            return commit_hash if commit_hash else 'unknown'
        except Exception:
            return 'unknown'


    def __repr__(self):
        s  = f'<{type(self).__name__} object>\n\n'
        s += f'Requested parameters:\n'
        s += f' - Wake model: {self.mod_wake} (1:Polar; 2:Curl; 3:Cartesian)\n'
        s += f' - Extent of high-res boxes: {self.extent_high} D to each side\n'
        s += f' - Extent of low-res box: xmin={self.extent_low[0]} D, xmax={self.extent_low[1]} D, ymin={self.extent_low[2]} D, ymax={self.extent_low[3]} D, zmax={self.extent_low[4]} D\n'

        s += f'\n'
        s += f'LES parameters:\n'
        s += f' - velocity hub height: {self.incflo_velocity_hh} m/s\n'
        s += f' - ds LES at low-res level (level {self.level_lr}): ({self.dx_at_lr_level}, {self.dy_at_lr_level}, {self.dz_at_lr_level}) m\n'
        s += f' - ds LES at high-res level (level {self.level_hr}): ({self.dx_at_hr_level}, {self.dy_at_hr_level}, {self.dz_at_hr_level}) m\n'
        s += f' - dt LES: {self.dt} s\n'
        s += f' - Extents: ({self.prob_hi[0]-self.prob_lo[0]}, {self.prob_hi[1]-self.prob_lo[1]}, {self.prob_hi[2]-self.prob_lo[2]}) m\n'
        s += f'   - x: {self.prob_lo[0]}:{self.dx0}:{self.prob_hi[0]} m,\t  ({self.n_cell[0]} points at level 0)\n'
        s += f'   - y: {self.prob_lo[1]}:{self.dy0}:{self.prob_hi[1]} m,\t  ({self.n_cell[1]} points at level 0)\n'
        s += f'   - z: {self.prob_lo[2]}:{self.dz0}:{self.prob_hi[2]} m,\t  ({self.n_cell[2]} points at level 0)\n'

        s += f'\n'
        s += f'Low-res domain: \n'
        s += f' - ds low: {self.ds_low_les} m\n'
        s += f' - dt low: {self.dt_low_les} s (with LES dt = {self.dt} s, output interval is {self.output_interval_lr})\n'
        s += f' - Sampling labels: {self.sampling_labels_lr}\n'
        s += f' - Extents: ({self.xdist_lr}, {self.ydist_lr}, {self.zdist_lr}) m\n'
        s += f'   - x: {self.xlow_lr}:{self.ds_low_les}:{self.xhigh_lr} m,\t  ({self.nx_lr} points at level {self.level_lr})\n'
        s += f'   - y: {self.ylow_lr}:{self.ds_low_les}:{self.yhigh_lr} m,\t  ({self.ny_lr} points at level {self.level_lr})\n'
        s += f'   - z: {self.zlow_lr}:{self.ds_low_les}:{self.zhigh_lr} m,\t  ({self.nz_lr} points at level {self.level_lr})\n'

        s += f'\n'
        s += f'High-res domain: \n'
        s += f' - ds high: {self.ds_high_les} m\n'
        s += f' - dt high: {self.dt_high_les} s (with LES dt = {self.dt} s, output interval is {self.output_interval_hr})\n'
        s += f' - Sampling labels: {self.sampling_labels_hr}\n'
        for t in np.arange(len(self.hr_domains)):
            s += f" - Turbine {t}, base located at ({self.wts[t]['x']:.2f}, {self.wts[t]['y']:.2f}, {self.wts[t]['z']:.2f}), hub height of {self.wts[t]['zhub']} m\n"
            s += f"   - Extents: ({self.hr_domains[t]['xdist_hr']}, {self.hr_domains[t]['ydist_hr']}, {self.hr_domains[t]['zdist_hr']}) m\n"
            s += f"     - x: {self.hr_domains[t]['xlow_hr']}:{self.ds_high_les}:{self.hr_domains[t]['xhigh_hr']} m,\t  ({self.hr_domains[t]['nx_hr']} points at level {self.level_hr})\n"
            s += f"     - y: {self.hr_domains[t]['ylow_hr']}:{self.ds_high_les}:{self.hr_domains[t]['yhigh_hr']} m,\t  ({self.hr_domains[t]['ny_hr']} points at level {self.level_hr})\n"
            s += f"     - z: {self.hr_domains[t]['zlow_hr']}:{self.ds_high_les}:{self.hr_domains[t]['zhigh_hr']} m,\t  ({self.hr_domains[t]['nz_hr']} points at level {self.level_hr})\n"
        return s


    def _checkInputs(self):
        '''
        Check that the AMR-Wind inputs make sense
        '''
        if len(self.prob_lo) != 3:
            raise ValueError(f"prob_lo must contain 3 elements, but it has {len(self.prob_lo)}")
        if len(self.prob_hi) != 3:
            raise ValueError(f"prob_hi must contain 3 elements, but it has {len(self.prob_hi)}")
        if len(self.incflo_velocity_hh) != 3:
            raise ValueError(f"incflo_velocity_hh must contain 3 elements, but it has {len(self.incflo_velocity_hh)}")
        if (self.prob_lo[0] >= self.prob_hi[0]):
            raise ValueError("x-component of prob_lo larger than x-component of prob_hi")
        if (self.prob_lo[1] >= self.prob_hi[1]):
            raise ValueError("y-component of prob_lo larger than y-component of prob_hi")
        if (self.prob_lo[2] >= self.prob_hi[2]):
            raise ValueError("z-component of prob_lo larger than z-component of prob_hi")
        if self.mod_wake not in [1,2,3]:
            raise ValueError (f'mod_wake parameter can only be 1 (polar), 2 (curl), or 3 (cartesian). Received {self.mod_wake}.')

        if self.level_hr == -1:
            self.level_hr = self.max_level

        # fmax and cmax from input farm should be >0
        # Check that level_lr is >= 0
        # check that level_hr is <=self.max_level

        # For convenience, the turbines should not be zero-indexed
        if 'name' in self.wts[0]:
            if self.wts[0]['name'] != 'T1':
                if self.verbose>0: WARN(f"Recommended turbine numbering should start at 1. Currently it is zero-indexed.")


        # Flags of given/calculated spatial resolution for warning/error printing purposes
        self.given_ds_hr = False
        self.given_ds_lr = False
        warn_msg = ""
        if self.ds_hr is not None:
            warn_msg += f"HIGH-RES SPATIAL RESOLUTION GIVEN. CONVERTING FATAL ERRORS ON HIGH-RES BOXES CHECKS TO WARNINGS."
            self.given_ds_hr = True
        if self.ds_lr is not None:
            warn_msg += f"LOW-RES SPATIAL RESOLUTION GIVEN. CONVERTING FATAL ERRORS ON LOW-RES BOX CHECKS TO WARNINGS."
            self.given_ds_lr = True
        if self.verbose>0 and len(warn_msg)>0: WARN(f'{warn_msg}')
        a=1


    def _calc_simple_params(self):
        '''
        Calculate simulation parameters, given only AMR-Wind inputs
        '''
        # Grid resolution at Level 0
        self.dx0 = (self.prob_hi[0] - self.prob_lo[0]) / self.n_cell[0]
        self.dy0 = (self.prob_hi[1] - self.prob_lo[1]) / self.n_cell[1]
        self.dz0 = (self.prob_hi[2] - self.prob_lo[2]) / self.n_cell[2]

        # Create grid resolutions at every level. Each array position corresponds to one level
        # Access using res[1]['dx'], where `1` is the desired level
        res = {}
        for lvl in np.arange(0,self.max_level+1):
            res[lvl] = {'dx':self.dx0/(2**lvl), 'dy':self.dy0/(2**lvl), 'dz':self.dz0/(2**lvl)}

        # Get grid resolutions at the levels of each box
        self.dx_at_lr_level = res[self.level_lr]['dx']
        self.dy_at_lr_level = res[self.level_lr]['dy']
        self.dz_at_lr_level = res[self.level_lr]['dz']

        self.dx_at_hr_level = res[self.level_hr]['dx']
        self.dy_at_hr_level = res[self.level_hr]['dy']
        self.dz_at_hr_level = res[self.level_hr]['dz']

        # Get the maximum ds at each boxes' level
        self.ds_max_at_lr_level = max(res[self.level_lr]['dx'], res[self.level_lr]['dy'], res[self.level_lr]['dz'])
        self.ds_max_at_hr_level = max(res[self.level_hr]['dx'], res[self.level_hr]['dy'], res[self.level_hr]['dz'])

        # Hub height wind speed
        self.vhub = np.sqrt(self.incflo_velocity_hh[0]**2 + self.incflo_velocity_hh[1]**2)


    def _calc_sampling_params(self):
        '''
        Calculate parameters for sampling planes
        '''

        self._calc_sampling_labels()
        self._calc_sampling_time()
        self._calc_grid_resolution()
        self._calc_grid_placement()


    def _calc_sampling_labels(self):
        '''
        Calculate labels for AMR-Wind sampling
        '''
        sampling_labels_lr = ["Low"]
        self.sampling_labels_lr = sampling_labels_lr

        sampling_labels_hr = []
        for turbkey in self.wts:
            if 'name' in self.wts[turbkey].keys():
                wt_name = self.wts[turbkey]['name']
            else:
                wt_name = f'T{turbkey+1}'
            sampling_labels_hr.append(f"High{wt_name}_inflow0deg")
        
        self.sampling_labels_hr = sampling_labels_hr


    def _calc_sampling_time(self):
        '''
        Calculate timestep values and AMR-Wind plane sampling frequency
        '''

        # Get some paramters from farm input
        self.fmax_max     = max(turb['fmax']     for turb in self.wts.values())
        self.cmeander_min = min(turb['Cmeander'] for turb in self.wts.values())
        self.Dwake_min    = min(turb['D']        for turb in self.wts.values()) # Approximate D_wake as D_rotor
        self.cmax_min     = min(turb['cmax']     for turb in self.wts.values())


        # High resolution domain, dt_high_les
        if self.dt_hr is None:
            # Calculate dt of high-res per guidelines
            dt_hr_max = 1 / (2 * self.fmax_max)
            self.dt_high_les = getMultipleOf(dt_hr_max, multipleof=self.dt) # Ensure dt_hr is a multiple of the AMR-Wind timestep
            if self.dt_high_les ==0:
                raise ValueError(f"AMR-Wind timestep dt={self.dt} is too coarse for high resolution domain! The time step based on `fmax` is {dt_hr_max}, which is too small to be rounded as a multiple of dt.")
        else:
            # The dt of high-res is given
            self.dt_high_les = self.dt_hr

        if self.dt_high_les < self.dt:
            raise ValueError(f"AMR-Wind timestep {self.dt} too coarse for high resolution domain! AMR-Wind timestep must be at least {self.dt_high_les} sec.")


        # Low resolution domain, dt_low_les
        if self.mod_wake == 1: # Polar
            self.dr = self.cmax_min
            dt_lr_max = self.cmeander_min * self.Dwake_min / (10 * self.vhub)
        else: # mod_wake == 2 or 3
            self.dr = self.Dwake_min/15
            dt_lr_max = self.dr / (2* self.vhub)


        if self.dt_lr is None:
            # Calculate dt of low-res per guidelines
            self.dt_low_les = getMultipleOf(dt_lr_max, multipleof=self.dt_high_les)  # Ensure that dt_lr is a multiple of the high res sampling timestep
        else:
            # dt of low-res is given
            self.dt_low_les = self.dt_lr


        if self.dt_low_les < self.dt:
            raise ValueError(f"AMR-Wind timestep {self.dt} too coarse for low resolution domain! AMR-Wind timestep must be at least {self.dt_low_les} sec.")
        if self.dt_high_les > self.dt_low_les:
            raise ValueError(f"Low resolution timestep ({self.dt_low_les}) is finer than high resolution timestep ({self.dt_high_les})!")


        # Sampling interval
        self.output_interval_hr = int(np.floor(round(self.dt_high_les/self.dt,4)))
        self.output_interval_lr = getMultipleOf(self.dt_low_les/self.dt, multipleof=self.output_interval_hr)

        if self.output_interval_lr % self.output_interval_hr != 0:
            raise ValueError(f"Low resolution output interval of {self.output_interval_lr} not a multiple of the high resolution interval {self.output_interval_hr}!")

        # Convert sampling interval to sampling time interval
        self.output_time_interval_hr = float(np.round(self.output_interval_hr * self.dt, decimals=10))
        self.output_time_interval_lr = float(np.round(self.output_interval_lr * self.dt, decimals=10))


    def _calc_grid_resolution(self):
        '''
        Calculate sampling grid resolutions

        Uses level_lr and level_hr to determine which AMR-Wind level is to be used to sample these boxes
        '''

        # Calculate ds_hr and ds_lr if not specified as inputs
        if self.ds_hr is None:
            self.ds_hr = self._calc_grid_resolution_hr()

        if self.ds_lr is None:
            self.ds_lr = self._calc_grid_resolution_lr() 

        # Set the computed or speficied spatial resolutions
        self.ds_high_les = self.ds_hr
        self.ds_low_les  = self.ds_lr


        # Perform some checks
        if self.ds_high_les < self.ds_max_at_hr_level:
            error_msg = f"AMR-Wind grid spacing of {self.ds_max_at_hr_level} m at the high-res box level of {self.level_hr} is too coarse for "\
                        f"the high resolution domain. AMR-Wind grid spacing at level {self.level_hr} must be at least {self.ds_high_les} m."
            if self.given_ds_hr:
                if self.verbose>0 and len(error_msg)>0: WARN(f'{error_msg}')
            else:
                raise ValueError(error_msg)


        if self.ds_low_les < self.ds_max_at_lr_level:
            error_msg = f"AMR-Wind grid spacing of {self.ds_max_at_lr_level} at the low-res box level of {self.level_lr} is too coarse for "\
                        f"the low resolution domain! AMR-Wind grid spacing at level {self.level_lr} must be at least {self.ds_low_les} m. "\
                        f"If you can afford to have {self.ds_low_les} m on AMR-Wind for the low-res box, do so. If you cannot, add `ds_lr={self.ds_max_at_lr_level}` " \
                        f"to the call to `AMRWindSimulation`. Note that sampled values will no longer be at the cell centers, as you will be requesting "\
                        f"sampling at {self.ds_low_les} m while the underlying grid will be at {self.ds_max_at_lr_level} m.\n --- SUPRESSING FURTHER ERRORS ---"
            if self.given_ds_lr:
                if self.verbose>0 and len(error_msg)>0: WARN(f'{error_msg}')
            else:
                raise ValueError(error_msg)

        if self.ds_low_les % self.ds_high_les != 0:
            raise ValueError(f"Low resolution grid spacing of {self.ds_low_les} m not a multiple of the high resolution grid spacing {self.ds_high_les} m.")



    def _calc_grid_resolution_hr(self):

        ds_hr_max = self.cmax_min

        if ds_hr_max < self.ds_max_at_hr_level:
            raise ValueError(f"AMR-Wind grid spacing of {self.ds_max_at_hr_level} m at the high-res box level of {self.level_hr} is too coarse for "\
                             f"high resolution domain! The high-resolution domain requires AMR-Wind grid spacing to be at least {ds_hr_max} m. If a "\
                             f"coarser high-res domain is acceptable, then manually specify the high-resolution grid spacing to be at least "\
                             f"{self.ds_max_at_hr_level} with ds_hr = {self.ds_max_at_hr_level}.")
        ds_high_les = getMultipleOf(ds_hr_max, multipleof=self.ds_max_at_hr_level)  # Ensure that ds_hr is a multiple of the refined AMR-Wind grid spacing

        return ds_high_les


    def _calc_grid_resolution_lr(self):

        ds_lr_max = self.cmeander_min * self.Dwake_min * self.vhub / 150
        # The expression above is the same as
        # For polar wake model:  ds_lr_max = self.dt_low_les * self.vhub**2 / 15
        # For curled wake model: ds_lr_max = self.cmeander_max * self.dt_low_les * self.vhub**2 / 5

        ds_low_les = getMultipleOf(ds_lr_max, multipleof=self.ds_hr) 
        if self.verbose>0: INFO(f"Low-res spatial resolution (ds_low) should be >={ds_lr_max:.2f} m.\nTo be a multiple of ds_high"\
              f"={self.ds_hr} m, we pick ds_low={ds_low_les} m")

        #self.ds_lr = self.ds_low_les
        return ds_low_les




    def _calc_grid_placement(self):
        '''
        Calculate placement of sampling grids
        '''

        self._calc_grid_placement_hr()
        self._calc_grid_placement_lr()


    def _calc_grid_placement_hr(self):
        '''
        Calculate placement of high resolution grids based on desired level level_hr

        '''

        # Calculate high resolution grid placement
        hr_domains = {} 
        for turbkey in self.wts:
            wt_x = self.wts[turbkey]['x']
            wt_y = self.wts[turbkey]['y']
            wt_z = self.wts[turbkey]['z']
            wt_h = self.wts[turbkey]['zhub']
            wt_D = self.wts[turbkey]['D']

            # Calculate minimum/maximum HR domain extents
            xlow_hr_min  = wt_x - self.buffer_hr * wt_D 
            xhigh_hr_max = wt_x + self.buffer_hr * wt_D 
            ylow_hr_min  = wt_y - self.buffer_hr * wt_D 
            yhigh_hr_max = wt_y + self.buffer_hr * wt_D 
            zlow_hr_min  = wt_z
            zhigh_hr_max = wt_z + wt_h + self.buffer_hr * wt_D 

            # Snap the requested extent outward onto the LES sampling grid: floor the low edge and ceil the
            # high edge (both relative to prob_lo, where the grid lines actually sit), then derive the extent
            # and point count from what came out. This guarantees the box contains [wt - buffer, wt + buffer]
            # on every side regardless of where the turbine falls on the grid, at the cost of at most one
            # extra cell per direction.
            tol = 1e-6
            xlow_hr_aligned  = self.prob_lo[0] + self.ds_high_les*np.floor((xlow_hr_min  - self.prob_lo[0])/self.ds_high_les + tol)
            xhigh_hr_aligned = self.prob_lo[0] + self.ds_high_les*np.ceil( (xhigh_hr_max - self.prob_lo[0])/self.ds_high_les - tol)
            xdist_hr = xhigh_hr_aligned - xlow_hr_aligned
            nx_hr = int(round(xdist_hr/self.ds_high_les)) + 1

            ylow_hr_aligned  = self.prob_lo[1] + self.ds_high_les*np.floor((ylow_hr_min  - self.prob_lo[1])/self.ds_high_les + tol)
            yhigh_hr_aligned = self.prob_lo[1] + self.ds_high_les*np.ceil( (yhigh_hr_max - self.prob_lo[1])/self.ds_high_les - tol)
            ydist_hr = yhigh_hr_aligned - ylow_hr_aligned
            ny_hr = int(round(ydist_hr/self.ds_high_les)) + 1

            # Calculate actual HR domain extent
            #  NOTE: Sampling planes should measure at AMR-Wind cell centers, not cell edges
            xlow_hr = xlow_hr_aligned - 0.5*self.dx_at_hr_level
            xhigh_hr = xlow_hr + xdist_hr

            ylow_hr = ylow_hr_aligned - 0.5*self.dy_at_hr_level
            yhigh_hr = ylow_hr + ydist_hr

            # The z floor is snapped UP (ceil), unlike x/y: the requested floor is the tower base, and the
            # per-turbine refinement region does not extend below it, so the sample box must start at or
            # above it to be contained in that level. The top is snapped up as well (below) so the requested
            # zhigh_hr_max is always covered.
            zlow_hr_snapped = self.prob_lo[2] + self.ds_high_les*np.ceil((zlow_hr_min - self.prob_lo[2])/self.ds_high_les - tol)
            zlow_hr = zlow_hr_snapped + 0.5*self.dz_at_hr_level

            # Recompute z-extent from the (now snapped-up) floor so we still cover zhigh_hr_max
            zdist_hr_min = zhigh_hr_max - zlow_hr
            zdist_hr = self.ds_high_les * np.ceil(zdist_hr_min/self.ds_high_les)
            nz_hr = int(zdist_hr/self.ds_high_les) + 1

            zhigh_hr = zlow_hr + zdist_hr
            zoffsets_hr = np.arange(zlow_hr, zhigh_hr+self.ds_high_les, self.ds_high_les) - zlow_hr


            # Check domain extents
            if xhigh_hr > self.prob_hi[0]:
                raise ValueError(f"Turbine {turbkey}: HR domain point {xhigh_hr} extends beyond maximum AMR-Wind x-extent!")
            if xlow_hr < self.prob_lo[0]:
                raise ValueError(f"Turbine {turbkey}: HR domain point {xlow_hr} extends beyond minimum AMR-Wind x-extent!")
            if yhigh_hr > self.prob_hi[1]:
                raise ValueError(f"Turbine {turbkey}: HR domain point {yhigh_hr} extends beyond maximum AMR-Wind y-extent!")
            if ylow_hr < self.prob_lo[1]:
                raise ValueError(f"Turbine {turbkey}: HR domain point {ylow_hr} extends beyond minimum AMR-Wind y-extent!")
            if zhigh_hr > self.prob_hi[2]:
                raise ValueError(f"Turbine {turbkey}: HR domain point {zhigh_hr} extends beyond maximum AMR-Wind z-extent!")
            if zlow_hr < self.prob_lo[2]:
                raise ValueError(f"Turbine {turbkey}: HR domain point {zlow_hr} extends beyond minimum AMR-Wind z-extent!")

            # Save out info for FFCaseCreation
            self.extent_high = self.buffer_hr*2

            hr_turb_info = {'nx_hr': nx_hr, 'ny_hr': ny_hr, 'nz_hr': nz_hr,
                            'xdist_hr': xdist_hr, 'ydist_hr': ydist_hr, 'zdist_hr': zdist_hr,
                            'xlow_hr': xlow_hr, 'ylow_hr': ylow_hr, 'zlow_hr': zlow_hr,
                            'xhigh_hr': xhigh_hr, 'yhigh_hr': yhigh_hr, 'zhigh_hr': zhigh_hr,
                            'zoffsets_hr': zoffsets_hr}
            hr_domains[turbkey] = hr_turb_info
        self.hr_domains = hr_domains


    def _calc_grid_placement_lr(self):
        '''
        Calculate placement of low resolution grid based on desired level level_lr
        '''

        ### ~~~~~~~~~ Calculate low resolution grid placement ~~~~~~~~~ 
        # Calculate minimum/maximum LR domain extents
        wt_all_x_min = min(turb['x'] for turb in self.wts.values())
        wt_all_x_max = max(turb['x'] for turb in self.wts.values())
        wt_all_y_min = min(turb['y'] for turb in self.wts.values())
        wt_all_y_max = max(turb['y'] for turb in self.wts.values())
        wt_all_z_min = min(turb['z'] for turb in self.wts.values())
        wt_all_z_max = max(turb['z']+turb['zhub']+0.5*turb['D'] for turb in self.wts.values())
        D_max = max(turb['D'] for turb in self.wts.values())
            
        xlow_lr_min  = wt_all_x_min - self.buffer_lr[0] * D_max
        xhigh_lr_max = wt_all_x_max + self.buffer_lr[1] * D_max 
        ylow_lr_min  = wt_all_y_min - self.buffer_lr[2] * D_max 
        yhigh_lr_max = wt_all_y_max + self.buffer_lr[3] * D_max 
        zlow_lr_min  = wt_all_z_min - 0.5               * D_max
        zhigh_lr_max = wt_all_z_max + self.buffer_lr[4] * D_max 

        # Carve out exception for flat terrain
        if wt_all_z_min == 0:
            zlow_lr_min = wt_all_z_min

        # Calculate the minimum/maximum LR domain coordinate lengths & number of grid cells
        xdist_lr_min = xhigh_lr_max - xlow_lr_min  # Minumum possible length of x-extent of LR domain
        ydist_lr_min = yhigh_lr_max - ylow_lr_min
        zdist_lr_min = zhigh_lr_max - zlow_lr_min

        self.xdist_lr = self.ds_low_les * np.ceil(xdist_lr_min/self.ds_low_les)  # The `+ ds_lr` comes from the +1 to NS_LOW in Sec. 4.2.15.6.4.1.1
        self.ydist_lr = self.ds_low_les * np.ceil(ydist_lr_min/self.ds_low_les)
        self.zdist_lr = self.ds_low_les * np.ceil(zdist_lr_min/self.ds_low_les)

        self.nx_lr = int(self.xdist_lr/self.ds_low_les) + 1  # TODO: adjust xdist_lr calculation by also using `inflow_deg`
        self.ny_lr = int(self.ydist_lr/self.ds_low_les) + 1  # TODO: adjust ydist_lr calculation by also using `inflow_deg`
        self.nz_lr = int(self.zdist_lr/self.ds_low_les) + 1

        ## Calculate actual LR domain extent
        #   NOTE: Sampling planes should measure at AMR-Wind cell centers, not cell edges
        #   NOTE: Should we use dx/dy/dz values here or ds_lr?
        #           - AR: I think it's correct to use ds_lr to get to the xlow values,
        #               but then offset by 0.5*amr_dx0 if need be
        self.xlow_lr = getMultipleOf(xlow_lr_min, multipleof=self.ds_low_les) - 0.5*self.dx_at_lr_level + self.prob_lo[0]%self.ds_low_les
        self.ylow_lr = getMultipleOf(ylow_lr_min, multipleof=self.ds_low_les) - 0.5*self.dy_at_lr_level + self.prob_lo[1]%self.ds_low_les
        self.zlow_lr = getMultipleOf(zlow_lr_min, multipleof=self.ds_low_les) + 0.5*self.dz_at_lr_level + self.prob_lo[2]%self.ds_low_les

        self.xhigh_lr = self.xlow_lr + self.xdist_lr
        self.yhigh_lr = self.ylow_lr + self.ydist_lr
        self.zhigh_lr = self.zlow_lr + self.zdist_lr
        self.zoffsets_lr = np.arange(self.zlow_lr, self.zhigh_lr+self.ds_low_les, self.ds_low_les) - self.zlow_lr

        # Check domain extents
        if self.xhigh_lr > self.prob_hi[0]:
            raise ValueError(f"LR domain point {self.xhigh_lr} extends beyond maximum AMR-Wind x-extent!")
        if self.xlow_lr < self.prob_lo[0]:
            raise ValueError(f"LR domain point {self.xlow_lr} extends beyond minimum AMR-Wind x-extent!")
        if self.yhigh_lr > self.prob_hi[1]:
            raise ValueError(f"LR domain point {self.yhigh_lr} extends beyond maximum AMR-Wind y-extent!")
        if self.ylow_lr < self.prob_lo[1]:
            raise ValueError(f"LR domain point {self.ylow_lr} extends beyond minimum AMR-Wind y-extent!")
        if self.zhigh_lr > self.prob_hi[2]:
            raise ValueError(f"LR domain point {self.zhigh_lr} extends beyond maximum AMR-Wind z-extent!")
        if self.zlow_lr < self.prob_lo[2]:
            raise ValueError(f"LR domain point {self.zlow_lr} extends beyond minimum AMR-Wind z-extent!")

        # Check grid placement
        self._check_grid_placement()

        # Save out info for FFCaseCreation
        self.extent_low = self.buffer_lr


    def _check_grid_placement(self):
        '''
        Check the values of parameters that were calculated by _calc_sampling_params
        '''

        # Calculate parameters of the low-res grid to allow checking
        amr_xgrid_at_lr_level = np.arange(self.prob_lo[0], self.prob_hi[0], self.dx_at_lr_level)
        amr_ygrid_at_lr_level = np.arange(self.prob_lo[1], self.prob_hi[1], self.dy_at_lr_level)
        amr_zgrid_at_lr_level = np.arange(self.prob_lo[2], self.prob_hi[2], self.dz_at_lr_level)

        amr_xgrid_at_lr_level_cc = amr_xgrid_at_lr_level + 0.5*self.dx_at_lr_level  # Cell-centered AMR-Wind x-grid
        amr_ygrid_at_lr_level_cc = amr_ygrid_at_lr_level + 0.5*self.dy_at_lr_level
        amr_zgrid_at_lr_level_cc = amr_zgrid_at_lr_level + 0.5*self.dz_at_lr_level

        sampling_xgrid_lr = self.xlow_lr + self.ds_lr*np.arange(self.nx_lr)
        sampling_ygrid_lr = self.ylow_lr + self.ds_lr*np.arange(self.ny_lr)
        sampling_zgrid_lr = self.zlow_lr + self.zoffsets_lr

        # Check the low-res grid placement
        self._check_grid_placement_single(sampling_xgrid_lr, amr_xgrid_at_lr_level_cc, boxstr='Low', dirstr='x')
        self._check_grid_placement_single(sampling_ygrid_lr, amr_ygrid_at_lr_level_cc, boxstr='Low', dirstr='y')
        self._check_grid_placement_single(sampling_zgrid_lr, amr_zgrid_at_lr_level_cc, boxstr='Low', dirstr='z')



        # High resolution grids (span the entire domain to make this check easier)
        amr_xgrid_at_hr_level = np.arange(self.prob_lo[0], self.prob_hi[0], self.dx_at_hr_level)
        amr_ygrid_at_hr_level = np.arange(self.prob_lo[1], self.prob_hi[1], self.dy_at_hr_level)
        amr_zgrid_at_hr_level = np.arange(self.prob_lo[2], self.prob_hi[2], self.dz_at_hr_level)

        amr_xgrid_at_hr_level_cc = amr_xgrid_at_hr_level + 0.5*self.dx_at_hr_level
        amr_ygrid_at_hr_level_cc = amr_ygrid_at_hr_level + 0.5*self.dy_at_hr_level
        amr_zgrid_at_hr_level_cc = amr_zgrid_at_hr_level + 0.5*self.dz_at_hr_level

        for turbkey in self.hr_domains:
            nx_hr   = self.hr_domains[turbkey]['nx_hr']
            ny_hr   = self.hr_domains[turbkey]['ny_hr']
            xlow_hr = self.hr_domains[turbkey]['xlow_hr']
            ylow_hr = self.hr_domains[turbkey]['ylow_hr']

            sampling_xgrid_hr = xlow_hr + self.ds_hr*np.arange(nx_hr)
            sampling_ygrid_hr = ylow_hr + self.ds_hr*np.arange(ny_hr)
            sampling_zgrid_hr = self.hr_domains[turbkey]['zlow_hr'] + self.hr_domains[turbkey]['zoffsets_hr']

            # Check the high-res grid placement
            self._check_grid_placement_single(sampling_xgrid_hr, amr_xgrid_at_hr_level_cc, boxstr='High', dirstr='x')
            self._check_grid_placement_single(sampling_ygrid_hr, amr_ygrid_at_hr_level_cc, boxstr='High', dirstr='y')
            self._check_grid_placement_single(sampling_zgrid_hr, amr_zgrid_at_hr_level_cc, boxstr='High', dirstr='z')



    def _check_grid_placement_single(self, sampling_xyzgrid_lhr, amr_xyzgrid_at_lhr_level_cc, boxstr, dirstr):
        '''
        Generic function to check placement of x, y, z grids in low, high-res boxes

        Inputs
        ------
        sampling_xyzgrid_lhr: np array
            Actual sampling locations of {x,y,z}grid on either low or high res (`lhr`) 
        amr_xyzgrid_at_lhr_level_cc: np array
            Actual AMR-Wind grid cell-center location of {x,y,z}grid at either low- or high-res levels
        boxstr: str
            Either 'Low', or 'High'
        distr: str
            Either 'x', 'y', or 'z'

        '''
        
	# Check if all values in sampling grid sampling_{x,y,z}grid_lr are part of AMR-Wind cell-centered values amr_{x,y,z}grid_at_lr_level_cc
        is_sampling_xyzgrid_subset = set(sampling_xyzgrid_lhr).issubset(set(amr_xyzgrid_at_lhr_level_cc))

        if is_sampling_xyzgrid_subset is False:
            amr_index = np.argmin(np.abs(amr_xyzgrid_at_lhr_level_cc - sampling_xyzgrid_lhr[0]))
            error_msg = f"{boxstr} resolution {dirstr}-sampling grid is not cell-centered with AMR-Wind's grid. \n    "\
        		f"{dirstr}-sampling grid:        {sampling_xyzgrid_lhr[0]}, {sampling_xyzgrid_lhr[1]}, "\
                        f"{sampling_xyzgrid_lhr[2]}, {sampling_xyzgrid_lhr[3]}, ... \n    "\
        		f"AMR-Wind grid (subset): {amr_xyzgrid_at_lhr_level_cc[amr_index  ]}, {amr_xyzgrid_at_lhr_level_cc[amr_index+1]}, "\
        		f"{amr_xyzgrid_at_lhr_level_cc[amr_index+2]}, {amr_xyzgrid_at_lhr_level_cc[amr_index+3]}, ..."
            if self.given_ds_lr:
                if self.verbose>0 and len(error_msg)>0: WARN(f'{error_msg}')
            else:
                raise ValueError(error_msg)



    def write_sampling_params(self, out=None, format='netcdf', terrain=None, overwrite=False, chunk_size_vec_lowres=(32,32,'auto')):
        '''
        Write out text that can be used for the sampling planes in an 
          AMR-Wind input file

        out: str
            Output path or full filename for Input file to be written
            to. If None, result is written to screen.
        format: str
            Format requested for the output to be saved from AMR-Wind. Options are
            native, netcdf, amrex_erf, or amrex_amrwind
        terrain: bool (required)
            If the case contains terrain. If it does, mu_turb will also be requested. This
            is necessary to process terrain files and set NaN inside the terrain.
        overwrite: bool
            If saving to a file, whether or not to overwrite potentially
            existing file
        chunk_size_vec_lowres: vector of int
            Size of AMReX-based chunk for the low-res only, used with the sub-volume sampling.
            This is a *maximum* box size, so the box count in each direction is
            ceil(num_points/chunk_size). The vertical entry may be given as 'auto' (default),
            in which case it is set to the full vertical extent of the low-res box.
            High-res boxes have no chunking and each high-res box is given as a single chunk.

        '''

        if format not in ['netcdf','native','amrex_erf','amrex_amrwind']:
            raise ValueError(f'format should be one of: native, netcdf, amrex_erf, amrex_amrwind')
        self.sampling_format = format

        if terrain == True:
            # The version where mu_turb is saved is only for netcdf/native
            # The amrex version includes the terrain mask through the use of derived fields
            fields = 'velocity mu_turb'
        elif terrain == False:
            fields = 'velocity'
        else:
            raise ValueError(f'The `terrain` input should be explicitly set to True or False')

        if self.sampling_format == 'amrex_erf':
            s  = f"#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#\n"
            s += f"#         ERF SUBVOLUME OUTPUT          #\n"
            s += f"#.......................................#\n"
            s += f"# Sampling info generated by openfast_toolbox on {self.curr_datetime}, commit {self.commit}.\n"
            s += f"erf.fixed_dt          = {self.dt}\n\n"
            s += self._write_sampling_params_amrex_erf(fields, chunk_size_vec_lowres)
        elif self.sampling_format == 'amrex_amrwind':
            # Write time step information for consistency with sampling interval
            s = f"time.fixed_dt    = {self.dt}\n\n"
            # Write flow velocity info for consistency
            s += f"incflo.velocity = {self.incflo_velocity_hh[0]} {self.incflo_velocity_hh[1]} {self.incflo_velocity_hh[2]}\n\n\n"

            s += f"#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#\n"
            s += f"#       AMRWIND AMREX SAMPLING          #\n"
            s += f"#.......................................#\n"
            s += f"# Sampling info generated by openfast_toolbox on {self.curr_datetime}, commit {self.commit}.\n"
            s += f"incflo.post_processing                = {self.postproc_name_lr} {self.postproc_name_hr}\n\n\n"
            s += self._write_sampling_params_amrex_amrwind(terrain, chunk_size_vec_lowres)
        else:
            # Write time step information for consistency with sampling interval
            s = f"time.fixed_dt    = {self.dt}\n\n"
            # Write flow velocity info for consistency
            s += f"incflo.velocity = {self.incflo_velocity_hh[0]} {self.incflo_velocity_hh[1]} {self.incflo_velocity_hh[2]}\n\n\n"

            # Write high-level info for sampling
            s += f"#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#\n"
            s += f"#          POST-PROCESSING              #\n"
            s += f"#.......................................#\n"
            s += f"# Sampling info generated by openfast_toolbox on {self.curr_datetime}, commit {self.commit}.\n"
            s += f"incflo.post_processing                = {self.postproc_name_lr} {self.postproc_name_hr}\n\n\n"
            s += self._write_sampling_params_netcdfnative(fields)

        if out is None:
            return s
        elif os.path.isdir(out):
            outfile = os.path.join(out, 'sampling_config.i')
        else: 
            # full file given
            outfile = out
            if not os.path.exists(os.path.dirname(outfile)):
                os.makedirs(os.path.dirname(outfile))
            if not overwrite:
                if os.path.isfile(outfile):
                    raise FileExistsError(f"{str(outfile)} already exists! Aborting...")

        with open(outfile,"w") as out:
            out.write(s)



    def _write_sampling_params_netcdfnative(self, fields):

        sampling_labels_lr_str = " ".join(str(item) for item in self.sampling_labels_lr)
        sampling_labels_hr_str = " ".join(str(item) for item in self.sampling_labels_hr)
        output_time_interval_lr_str = f"{self.output_time_interval_lr:.10f}".rstrip('0').rstrip('.')
        output_time_interval_hr_str = f"{self.output_time_interval_hr:.10f}".rstrip('0').rstrip('.')

        s  = f"# ---- Low-res sampling parameters ----\n"
        s += f"{self.postproc_name_lr}.output_format           = {self.sampling_format}\n"
        s += f"{self.postproc_name_lr}.output_after_final_step = false\n"
        s += f"{self.postproc_name_lr}.output_from_restart     = true\n"
        s += f"{self.postproc_name_lr}.output_time_interval    = {output_time_interval_lr_str}\n"
        s += f"{self.postproc_name_lr}.fields                  = {fields}\n"
        s += f"{self.postproc_name_lr}.labels                  = {sampling_labels_lr_str}\n\n"

        # Write out low resolution sampling plane info
        zoffsets_lr_str = " ".join(str(item) for item in self.zoffsets_lr)

        s += f"# Low sampling grid spacing = {self.ds_lr} m\n"
        s += f"{self.postproc_name_lr}.Low.type          = PlaneSampler\n"
        s += f"{self.postproc_name_lr}.Low.num_points    = {self.nx_lr} {self.ny_lr}\n"
        s += f"{self.postproc_name_lr}.Low.origin        = {self.xlow_lr:.4f} {self.ylow_lr:.4f} {self.zlow_lr:.4f}\n"  # Round the float output
        s += f"{self.postproc_name_lr}.Low.axis1         = {self.xdist_lr:.4f} 0.0 0.0\n"  # Assume: axis1 oriented parallel to AMR-Wind x-axis
        s += f"{self.postproc_name_lr}.Low.axis2         = 0.0 {self.ydist_lr:.4f} 0.0\n"  # Assume: axis2 oriented parallel to AMR-Wind y-axis
        s += f"{self.postproc_name_lr}.Low.offset_vector = 0.0 0.0 1.0\n"
        s += f"{self.postproc_name_lr}.Low.offsets       = {zoffsets_lr_str}\n\n\n"

        s += f"# ---- High-res sampling parameters ----\n"
        s += f"{self.postproc_name_hr}.output_format           = {self.sampling_format}\n"
        s += f"{self.postproc_name_hr}.output_after_final_step = false\n"
        s += f"{self.postproc_name_hr}.output_from_restart     = true\n"
        s += f"{self.postproc_name_hr}.output_time_interval    = {output_time_interval_hr_str}\n"
        s += f"{self.postproc_name_hr}.fields                  = {fields}\n"
        s += f"{self.postproc_name_hr}.labels                  = {sampling_labels_hr_str}\n"

        # Write out high resolution sampling plane info
        for turbkey in self.hr_domains:
            wt_x = self.wts[turbkey]['x']
            wt_y = self.wts[turbkey]['y']
            wt_z = self.wts[turbkey]['z']
            wt_h = self.wts[turbkey]['zhub']
            wt_D = self.wts[turbkey]['D']
            if 'name' in self.wts[turbkey].keys():
                wt_name = self.wts[turbkey]['name']
            else:
                wt_name = f'T{turbkey+1}'
            sampling_name = f"High{wt_name}_inflow0deg"
            nx_hr = self.hr_domains[turbkey]['nx_hr']
            ny_hr = self.hr_domains[turbkey]['ny_hr']
            xlow_hr = self.hr_domains[turbkey]['xlow_hr']
            ylow_hr = self.hr_domains[turbkey]['ylow_hr']
            zlow_hr = self.hr_domains[turbkey]['zlow_hr']
            xdist_hr = self.hr_domains[turbkey]['xdist_hr']
            ydist_hr = self.hr_domains[turbkey]['ydist_hr']
            zoffsets_hr = self.hr_domains[turbkey]['zoffsets_hr']
            zoffsets_hr_str = " ".join(str(item) for item in zoffsets_hr)

            s += f"\n# Turbine {wt_name} with base at (x,y,z) = ({wt_x:.4f}, {wt_y:.4f}, {wt_z:.4f}), with hh = {wt_h}, D = {wt_D}, grid spacing = {self.ds_hr} m\n"
            s += f"{self.postproc_name_hr}.{sampling_name}.type          = PlaneSampler\n"
            s += f"{self.postproc_name_hr}.{sampling_name}.num_points    = {nx_hr} {ny_hr}\n"
            s += f"{self.postproc_name_hr}.{sampling_name}.origin        = {xlow_hr:.4f} {ylow_hr:.4f} {zlow_hr:.4f}\n"  # Round the float output
            s += f"{self.postproc_name_hr}.{sampling_name}.axis1         = {xdist_hr:.4f} 0.0 0.0\n"  # Assume: axis1 oriented parallel to AMR-Wind x-axis
            s += f"{self.postproc_name_hr}.{sampling_name}.axis2         = 0.0 {ydist_hr:.4f} 0.0\n"  # Assume: axis2 oriented parallel to AMR-Wind y-axis
            s += f"{self.postproc_name_hr}.{sampling_name}.offset_vector = 0.0 0.0 1.0\n"
            s += f"{self.postproc_name_hr}.{sampling_name}.offsets       = {zoffsets_hr_str}\n"

        return s


    def _write_sampling_params_amrex_erf(self, fields, chunk_size_vec_lowres):
        def _to_erf_sampling_vars(fields_in):
            vars_out = []
            for field in fields_in.split():
                if field == 'velocity':
                    vars_out.extend(['x_velocity', 'y_velocity', 'z_velocity'])
                else:
                    vars_out.append(field)
            return " ".join(vars_out)

        # ERF subvolume sampling uses cell-corner origins.
        # Internal placement in this class stores cell-centered origins, so shift by half
        # of the AMR-Wind cell size at the corresponding level.
        xlow_lr_corner = self.xlow_lr - 0.5 * self.dx_at_lr_level
        ylow_lr_corner = self.ylow_lr - 0.5 * self.dy_at_lr_level
        zlow_lr_corner = self.zlow_lr - 0.5 * self.dz_at_lr_level

        chunk_z_lr = chunk_size_vec_lowres[2]
        if chunk_z_lr == 'auto':
            # 'auto' gives chunks spanning the full vertical extent, i.e. one box per column
            chunk_z_lr = self.nz_lr

        origins = [f"{xlow_lr_corner:.4f} {ylow_lr_corner:.4f} {zlow_lr_corner:.4f}"]
        nxnynz = [f"{self.nx_lr} {self.ny_lr} {self.nz_lr}"]
        dxdydz = [f"{self.ds_low_les} {self.ds_low_les} {self.ds_low_les}"]
        chunks = [f"{chunk_size_vec_lowres[0]} {chunk_size_vec_lowres[1]} {chunk_z_lr}"]

        s  = f"# ---- ERF subvolume sampling parameters ----\n"
        s += f"# Order of boxes in origin/nxnynz/dxdydz/chunk_size: low-res first, then each high-res turbine box\n"

        for turbkey in self.hr_domains:
            xlow_hr_corner = self.hr_domains[turbkey]['xlow_hr'] - 0.5 * self.dx_at_hr_level
            ylow_hr_corner = self.hr_domains[turbkey]['ylow_hr'] - 0.5 * self.dy_at_hr_level
            zlow_hr_corner = self.hr_domains[turbkey]['zlow_hr'] - 0.5 * self.dz_at_hr_level

            origins.append(f"{xlow_hr_corner:.4f} {ylow_hr_corner:.4f} {zlow_hr_corner:.4f}")
            nxnynz.append(
                f"{self.hr_domains[turbkey]['nx_hr']} {self.hr_domains[turbkey]['ny_hr']} {self.hr_domains[turbkey]['nz_hr']}"
            )
            dxdydz.append(f"{self.ds_high_les} {self.ds_high_les} {self.ds_high_les}")
            # Each high-res box is a single chunk, sized to this turbine's own grid
            chunks.append(
                f"{self.hr_domains[turbkey]['nx_hr']} {self.hr_domains[turbkey]['ny_hr']} {self.hr_domains[turbkey]['nz_hr']}"
            )

        erf_sampling_vars = _to_erf_sampling_vars(fields)
        subvol_file = os.path.join('post_processing',f"{self.postproc_name_lr}_{self.postproc_name_hr}") # TODO fix

        s += f"erf.subvol_file          = \"{subvol_file}\"\n"
        s += f"erf.subvol_int           = {self.output_interval_lr} {self.output_interval_hr}\n"
        s += f"erf.subvol_sampling_vars = {erf_sampling_vars}\n"
        s += f"erf.subvol.origin        = {'       '.join(origins)}\n"
        s += f"erf.subvol.nxnynz        = {'       '.join(nxnynz)}\n"
        s += f"erf.subvol.dxdydz        = {'       '.join(dxdydz)}\n"
        s += f"erf.subvol.chunk_size    = {'       '.join(chunks)}\n"

        return s


    def _write_sampling_params_amrex_amrwind(self, terrain, chunk_size_vec_lowres):

        # Here we need to modify the labels so that is consistent with ERF-based AMReX samping
        sampling_labels_lr = ['0_']
        sampling_labels_hr = [f"{i+1}_" for i, s in enumerate(self.sampling_labels_hr)]

        sampling_labels_lr_str = " ".join(str(item) for item in sampling_labels_lr)
        sampling_labels_hr_str = " ".join(str(item) for item in sampling_labels_hr)
        output_time_interval_lr_str = f"{self.output_time_interval_lr:.10f}".rstrip('0').rstrip('.')
        output_time_interval_hr_str = f"{self.output_time_interval_hr:.10f}".rstrip('0').rstrip('.')

        is_lr_aligned = (
            np.isclose(self.dx_at_lr_level, self.ds_low_les)
            and np.isclose(self.dy_at_lr_level, self.ds_low_les)
            and np.isclose(self.dz_at_lr_level, self.ds_low_les)
        )
        if not is_lr_aligned:
            WARN(
                "amrex_amrwind low-res sampling not perfectly matching a certain level. "
                f"Received LES dx/dy/dz at level {self.level_lr} = ({self.dx_at_lr_level}, {self.dy_at_lr_level}, {self.dz_at_lr_level}) "
                f"and ds_low_les = {self.ds_low_les}. Skipping cells to get the desired sampling at {self.ds_low_les} m uniform resolution.")

        is_hr_aligned = (
            np.isclose(self.dx_at_hr_level, self.ds_high_les)
            and np.isclose(self.dy_at_hr_level, self.ds_high_les)
            and np.isclose(self.dz_at_hr_level, self.ds_high_les)
        )
        if not is_hr_aligned:
            WARN(
                "amrex_amrwind high-res sampling not perfectly matching a certain level."
                f"Received LES dx/dy/dz at level {self.level_hr} = ({self.dx_at_hr_level}, {self.dy_at_hr_level}, {self.dz_at_hr_level}) "
                f"and ds_high_les = {self.ds_high_les}. Skipping cells to get the desired sampling at {self.ds_high_les} m uniform resolution.")

        #sampleAt = 'cell_centers'
        sampleAt = 'cell_corners'

        if sampleAt == 'cell_centers':
            xlow_lr_out = self.xlow_lr
            ylow_lr_out = self.ylow_lr
            zlow_lr_out = self.zlow_lr
            xshift_hr = 0.0
            yshift_hr = 0.0
            zshift_hr = 0.0
            origin_comment = 'cell centers'
        elif sampleAt == 'cell_corners':
            xlow_lr_out = self.xlow_lr - 0.5 * self.dx_at_lr_level
            ylow_lr_out = self.ylow_lr - 0.5 * self.dy_at_lr_level
            zlow_lr_out = self.zlow_lr - 0.5 * self.dz_at_lr_level
            xshift_hr = -0.5 * self.dx_at_hr_level
            yshift_hr = -0.5 * self.dy_at_hr_level
            zshift_hr = -0.5 * self.dz_at_hr_level
            origin_comment = 'cell corners'
        else:
            raise ValueError(f"Unknown sampleAt='{sampleAt}'. Expected 'cell_centers' or 'cell_corners'.")

        s  = f"# ---- Low-res sampling parameters ----\n"
        s += f"{self.postproc_name_lr}.type                    = Subvolume\n"
        s += f"{self.postproc_name_lr}.output_rename           = ffboxes\n"
        s += f"{self.postproc_name_lr}.output_after_final_step = false\n"
        s += f"{self.postproc_name_lr}.output_from_restart     = true\n"
        s += f"{self.postproc_name_lr}.output_time_interval    = {output_time_interval_lr_str}\n"
        if terrain:
            s += f"{self.postproc_name_lr}.derived_fields          = mask_terrain(velocity)\n"
            s += f"{self.postproc_name_lr}.fields                  = none\n"
        else:
            s += f"{self.postproc_name_lr}.fields                  = velocity\n"
        s += f"{self.postproc_name_lr}.labels                  = {sampling_labels_lr_str}\n\n"

        s += f"# Low sampling grid spacing = {self.ds_lr} m\n"
        s += f"{self.postproc_name_lr}.{sampling_labels_lr_str}.type           = Rectangular\n"
        s += f"{self.postproc_name_lr}.{sampling_labels_lr_str}.origin         = {xlow_lr_out:.4f} {ylow_lr_out:.4f} {zlow_lr_out:.4f} # {origin_comment}\n"
        s += f"{self.postproc_name_lr}.{sampling_labels_lr_str}.num_points     = {self.nx_lr} {self.ny_lr} {self.nz_lr}\n"
        s += f"{self.postproc_name_lr}.{sampling_labels_lr_str}.dx_vec         = {self.ds_low_les} {self.ds_low_les} {self.ds_low_les}\n"
        chunk_z_lr = chunk_size_vec_lowres[2]
        if chunk_z_lr == 'auto':
            # 'auto' gives chunks spanning the full vertical extent, i.e. one box per column
            chunk_z_lr = self.nz_lr

        s += f"{self.postproc_name_lr}.{sampling_labels_lr_str}.chunk_size_vec = {chunk_size_vec_lowres[0]} {chunk_size_vec_lowres[1]} {chunk_z_lr}\n\n\n"

        s += f"# ---- High-res sampling parameters ----\n"
        s += f"{self.postproc_name_hr}.type                    = Subvolume\n"
        s += f"{self.postproc_name_hr}.output_rename           = ffboxes\n"
        s += f"{self.postproc_name_hr}.output_after_final_step = false\n"
        s += f"{self.postproc_name_hr}.output_from_restart     = true\n"
        s += f"{self.postproc_name_hr}.output_time_interval    = {output_time_interval_hr_str}\n"
        if terrain:
            s += f"{self.postproc_name_hr}.derived_fields          = mask_terrain(velocity)\n"
            s += f"{self.postproc_name_hr}.fields                  = none\n"
        else:
            s += f"{self.postproc_name_hr}.fields                  = velocity\n"
        s += f"{self.postproc_name_hr}.labels                  = {sampling_labels_hr_str}\n"

        for turbkey in self.hr_domains:
            wt_x = self.wts[turbkey]['x']
            wt_y = self.wts[turbkey]['y']
            wt_z = self.wts[turbkey]['z']
            wt_h = self.wts[turbkey]['zhub']
            wt_D = self.wts[turbkey]['D']
            if 'name' in self.wts[turbkey].keys():
                wt_name = self.wts[turbkey]['name']
            else:
                wt_name = f'T{turbkey+1}'
            sampling_name = sampling_labels_hr[turbkey]
            nx_hr = self.hr_domains[turbkey]['nx_hr']
            ny_hr = self.hr_domains[turbkey]['ny_hr']
            nz_hr = self.hr_domains[turbkey]['nz_hr']
            xlow_hr = self.hr_domains[turbkey]['xlow_hr']
            ylow_hr = self.hr_domains[turbkey]['ylow_hr']
            zlow_hr = self.hr_domains[turbkey]['zlow_hr']
            xlow_hr_out = xlow_hr + xshift_hr
            ylow_hr_out = ylow_hr + yshift_hr
            zlow_hr_out = zlow_hr + zshift_hr

            s += f"\n# Turbine {wt_name} with base at (x,y,z) = ({wt_x:.4f}, {wt_y:.4f}, {wt_z:.4f}), with hh = {wt_h}, D = {wt_D}, grid spacing = {self.ds_hr} m\n"
            s += f"{self.postproc_name_hr}.{sampling_name}.type           = Rectangular\n"
            s += f"{self.postproc_name_hr}.{sampling_name}.origin         = {xlow_hr_out:.4f} {ylow_hr_out:.4f} {zlow_hr_out:.4f}  # {origin_comment}\n"
            s += f"{self.postproc_name_hr}.{sampling_name}.num_points     = {nx_hr} {ny_hr} {nz_hr}\n"
            s += f"{self.postproc_name_hr}.{sampling_name}.dx_vec         = {self.ds_high_les} {self.ds_high_les} {self.ds_high_les}\n"
            s += f"{self.postproc_name_hr}.{sampling_name}.chunk_size_vec = {nx_hr} {ny_hr} {nz_hr} # Entire box\n"

        return s




