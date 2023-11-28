#%% Import modules, define constants, functions and classes

########## Constants ###########

TRANSPORT = 1
ACTIVATION = 3
SHORT_STEP = 1

########## Modules ###########

# Generic
import pandas as pd
import os
import shutil
from glob import glob
import numpy as np
from copy import deepcopy

# Plotting
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LogNorm

# Cerberus
from cerberus.solvers import CodeInput, Solver
import cerberus

# Change current directory to this script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

########## Classes ###########

class Simulation:
    """
    Manages the initialization and execution of a Serpent/Cerberus simulation.
    Handles file parsing, simulation time-stepping, and data monitoring/control for solver interaction.
    """
    ########## Simulation analysis ###########
    def start(self, case_name, main_input_file, pbed_pos_file, materials, fracs, is_fuel, pbed_universe, division, N_passes, thermal_E=1e-5):
        """Initialize the simulation with the main input file."""
        # Recursively search for all related input files
        self.main_input_file = main_input_file
        self.input_files = self.rec_Serpent_file_search(self.main_input_file)
        # Store pebbles positions
        pos = pd.read_csv(pbed_pos_file, delim_whitespace=True, header=None, names=["x", "y", "z", "r", "uni"])
        # Create separate folder with case files
        os.makedirs(case_name)
        for input_file in self.input_files:
            shutil.copy(input_file, case_name)
        os.chdir(case_name)
        # Modify main input file with materials division
        N_radial = division['radial']['Nzones']
        radial_bounds = division['radial']['bounds']
        N_axial = division['axial']['Nzones']
        axial_bounds = division['axial']['bounds']
        radial_grid =  np.array([np.sqrt(radial_bounds[0]**2 + i * np.pi * (radial_bounds[1]**2 - radial_bounds[0]**2) / N_radial / np.pi) for i in range(N_radial + 1)]).round(6)
        axial_grid = np.linspace(axial_bounds[0], axial_bounds[1], N_axial+1).round(6)
        with open(main_input_file, 'a') as f:
            for i,mat in enumerate(materials):
                if is_fuel[i]:
                    f.write(f'\ndiv {mat} peb {pbed_universe} 1 {materials[mat]} subr {N_radial} {radial_bounds[0]} {radial_bounds[1]} subz {N_axial} {axial_bounds[0]} {axial_bounds[1]} submod {N_passes}\n')
            f.write('dep daystep 10\n') # just to make it a depletion calculation
            f.write('set depout 2 2\n')
            f.write('set Cerberus_verbosity 0\n')
            f.write('set pcc 0\n')
            f.write(f'det pebble_flux dl {pbed_universe} de E\n')
            f.write(f'det zone_flux dn 4 {N_radial} 1 {N_axial} {" ".join(radial_grid.astype(str))} 0 360 {" ".join(axial_grid.astype(str))} de E\n')
            f.write(f'det zone_power dr -8 void dn 4 {N_radial} 1 {N_axial} {" ".join(radial_grid.astype(str))} 0 360 {" ".join(axial_grid.astype(str))} de E\n')
            f.write(f'ene E 2 1 1e-10 {thermal_E}\n')
        # Modify pos file with universes break down
        pos['uni'] = assign_random_array(len(pos), [materials[mat] for mat in materials], fracs) 
        pos.to_csv(pbed_pos_file, sep='\t', header=False, index=False)

        # Prepare empty directories
        os.makedirs(f'./Plots/')
        os.makedirs(f'./Data/')

        # Prepare the input for the solver
        self.input = CodeInput(self.input_files)
        # Initialize the solver with necessary configurations
        self.solver = Solver("Serpent", sss_exe, ["-port", "-omp", str(ncores)])
        self.solver.input = self.input
        self.solver.initialize(use_MPI=MPI_mode)

    def deplete(self, curr_time, time_step, mode, switch_mode):
        """Advance the simulation by a given time step with the right depletion mode."""
        # Change depletion mode (1: transport, 2: decay, 3: activation)
        self.set_values(switch_mode, mode)
        # Increment the current time and deplete
        curr_time += time_step
        self.solver.advance_to_time(curr_time)
        return curr_time

    def rec_Serpent_file_search(self, main_input):
        files_to_read = [main_input]
        if main_input.split('.')[-1] in ['stl','ifc']:
            return files_to_read
        elif '.wrk' in main_input:
            return files_to_read
        path = os.path.dirname(main_input)
        with open(main_input) as f:
            lines = f.readlines()
        for i_line in range(len(lines)):
            line = lines[i_line].strip().split('%')[0]
            if len(line) == 0:
                continue
            fields = line.split()
            cmd = fields[0]
            if cmd == 'include':
                field_spot = 1
            elif cmd == 'pbed':
                field_spot = 3
            elif cmd == 'file':
                field_spot = 2
            elif cmd == 'set' and len(fields)>1:
                if fields[1] == 'dd' and fields[2] =='5':
                    field_spot = 3
                else:
                    continue
            elif cmd == 'ifc':
                field_spot = 1
            else:
                continue

            if isinstance(field_spot, int):
                field_spot = [field_spot]

            if len(fields) > max(field_spot):
                for i in field_spot:
                    file_name = fields[i].replace('"', '')
                    files_to_read += self.rec_Serpent_file_search(os.path.normpath(os.path.join(path, file_name)))
            else:
                spots = list(fields)
                while len(spots) <= max(field_spot):
                    i_line += 1
                    line = lines[i_line].strip().split('%')[0]
                    while len(line) == 0:
                        i_line += 1
                        line = lines[i_line].strip().split('%')[0]
                        if i_line > len(lines):
                            raise Exception('Lost here')
                    fields = line.split()
                    for j in fields:
                        spots.append(j)

                for j in field_spot:
                    file_name = spots[j].replace('"', '')
                    files_to_read += simulation.rec_Serpent_file_search(os.path.normpath(os.path.join(path, file_name)))
        return files_to_read

    
    ########## TRANSFERRABLES USE ###########
    def get_tra(self, transferrable, input_parameter=False):
        """Retrieve a transferrable object, adding a prefix if necessary."""
        prefix = 'sss_iv_' if input_parameter else 'sss_ov_'
        if isinstance(transferrable, str) and not transferrable.startswith('sss'):
            transferrable = f'{prefix}{transferrable}'
        return self.solver.get_transferrable(transferrable) if isinstance(transferrable, str) else transferrable

    def get(self, transferrable, input_parameter=False):
        """Get a transferrable object and perform communication with Serpent."""
        tra = self.get_tra(transferrable, input_parameter=input_parameter)
        tra.communicate()
        return tra

    def get_values(self, transferrable, input_parameter=False, return_singles=True, communicate=True):
        """Retrieve values from a transferrable object."""
        tra = self.get(transferrable, input_parameter) if communicate else self.get_tra(transferrable, input_parameter)
        values = tra.value_vec
        return values[0] if return_singles and len(values) == 1 else values

    def get_multiple_values(self, transferrables_matrix, input_parameter=False, return_singles=True, communicate=True, ignore_None=True):
        """Retrieve multiple values from a matrix of transferrable objects."""
        if isinstance(transferrables_matrix, list) or isinstance(transferrables_matrix, np.ndarray):
            return [self.get_multiple_values(sub_matrix, input_parameter, return_singles, communicate, ignore_None) 
                    for sub_matrix in transferrables_matrix]
        elif not ignore_None or transferrables_matrix is not None:
            return self.get_values(transferrables_matrix, input_parameter, return_singles, communicate)

    def set_values(self, transferrable, values, communicate=True):
        """Set values to a transferrable object and optionally communicate."""
        tra = self.get_tra(transferrable, input_parameter=True)
        # Handle single numeric values by converting them to a list
        if isinstance(values, (int, float, np.number)):
            values = [values]
        tra.value_vec = np.array(values)
        if communicate:
            tra.communicate()
        return tra


#%% Pebble bed class object
class Pebble_bed:
    def read_pbed_file(self, pbed_file_path, radial_center=[0, 0]):
        """Reads pebble bed data from a file and calculates relevant metrics."""
        # Load and preprocess pebble bed data from the specified file
        data = pd.read_csv(pbed_file_path, delim_whitespace=True, header=None, names=["x", "y", "z", "r", "uni"])
        self.data = data
        
        # Assign IDs and calculate central coordinates and distances
        self.data['id'] = self.data.index.to_numpy()
        self.center = np.array(radial_center + [data.z.mean()])
        self.data["r_dist"] = np.linalg.norm(self.data[['x', 'y']] - self.center[:2], axis=1)
        self.data["dist"] = np.linalg.norm(self.data[['x', 'y', 'z']] - self.center, axis=1)
        
        # Determine the bounding box of the pebble distribution and identify unique universes
        self.box = self.data[['x', 'y', 'z']].agg(['min', 'max']).to_numpy().T
        self.universes_list = self.data.uni.unique()

    def add_field(self, name, array):
        """Adds a new field to the pebble bed data."""
        self.data[name] = array

    def slice(self, dir_id='x', val='middle', tol=None):
        """Slices the pebble bed data along a specified direction and value."""
        # Determine the slicing value
        val = np.nanmean(self.data[dir_id]) if isinstance(val, str) and val == 'middle' else val
        sub_pbed = deepcopy(self)

        # Apply slicing conidtion
        condition = self.data[dir_id] - val
        sub_pbed.data = self.data[abs(condition) <= (self.data["r"] if tol is None else tol)]
        return sub_pbed

    def clip(self, dir_id='x', val='middle', direction=+1, logical_and=True):
        """Clips the pebble bed data based on specified conditions."""
        # Configure clipping parameters
        if not isinstance(dir_id, (list, np.ndarray, tuple)):
            dir_id = [dir_id]
        val = [val] * len(dir_id) if not isinstance(val, (list, np.ndarray, tuple)) else val
        direction = [direction] * len(dir_id) if not isinstance(direction, (list, np.ndarray, tuple)) else direction
        
        # Apply clipping conditions
        conditions = [self._apply_clip_condition(d_id, v, d) for d_id, v, d in zip(dir_id, val, direction)]
        gathered_conditions = np.logical_and.reduce(conditions) if logical_and else np.logical_or.reduce(conditions)
        sub_pbed = deepcopy(self)
        sub_pbed.data = self.data[gathered_conditions]
        return sub_pbed

    def _apply_clip_condition(self, dir_id, val, direction):
        """Helper function to apply clipping conditions."""
        # Determine and apply a single clip condition based on direction and value
        dir_id = str(dir_id).lower()
        if dir_id in ["x", "y", "z", 'dist', 'r_dist']:
            val = np.nanmean(self.data[dir_id]) if isinstance(val, str) and val == 'middle' else val
            return (self.data[dir_id] * direction >= val * direction)
        raise ValueError(f"Unknown direction id: {dir_id}")

    def projection(self, dir_id, val):
        """Calculates the projected radius for each pebble in a given direction."""
        # Calculate the projected radius based on the relative position in the specified direction
        rel_pos = val - self.data[dir_id]
        tmp = self.data.r**2 - rel_pos**2
        r_projected = np.full_like(tmp, np.nan)
        valid_indices = tmp >= 0
        r_projected[valid_indices] = np.sqrt(tmp[valid_indices])
        return r_projected

    def plot2D(self, field='id', dir_id='x', val='middle', colormap='turbo', xlim=None, ylim=None, tol=None, equal=True, field_title=None, clim=None, log_color=False, vertical_cbar=False, shrink_cbar=1, pad_cbar=0.15, translation=None, no_ax=False, superimpose_Serpent=False, Serpent_xsize=None, Serpent_ysize=None, Serpent_geom_path=None, fig_size=None, new_fig=True, save_fig=False, fig_folder='./', fig_name=None, fig_suffix='', fig_dpi=400, plot_title=None, patch_clip=None, unit='cm'):
        """Generates a 2D plot of the pebble bed."""
        # Determine the plotting axes based on the specified direction
        if dir_id == 'x':
            xdir = 'y'
            ydir = 'z'
            dir_id = 'x'
        elif dir_id == 'y':
            xdir = 'x'
            ydir = 'z'
            dir_id = 'y'
        elif dir_id == 'z':
            xdir = 'x'
            ydir = 'y'
            dir_id = 'z'
        # Prepare data for plotting, including slicing and projecting radii
        if isinstance(val, str) and val == 'middle':
            val = np.nanmean(self.data[dir_id])
        if isinstance(field_title, type(None)):
            field_title = field
        if isinstance(tol, type(None)):
            sub_pbed = self.slice(dir_id, val)
            r = sub_pbed.projection(dir_id, val)
        else:
            sub_pbed = self.slice(dir_id, val, tol=tol)
            r = np.array(sub_pbed.data.r)

        # Make the patch collection
        data = sub_pbed.data[sub_pbed.data.r != 0]
        x = np.array(data[xdir])
        y = np.array(data[ydir])

        if not isinstance(translation, type(None)):
            x += translation[0]
            y += translation[1]

        patches = []
        for i in range(len(data)):
            circle = Circle((x[i], y[i]), r[i])
            patches.append(circle)

        colors = np.array(data[field])

        if isinstance(clim, type(None)):
            clim = [data[field].min(), data[field].max()]
        if log_color:
            p = PatchCollection(patches, cmap=colormap, norm=LogNorm())
        else:
            p = PatchCollection(patches, cmap=colormap)
        p.set_array(colors)
        p.set_clim(clim)

        if new_fig:
            if not isinstance(fig_size, type(None)):
                plt.figure(figsize=fig_size, facecolor = "white")
            else:
                plt.figure()

        # Optional: superimpose Serpent geometry plot
        if superimpose_Serpent:
            self.show_Serpent_plot(Serpent_geom_path, new_fig=False, xlim=Serpent_xsize, ylim=Serpent_ysize)

        # Set up plot and colorbar layout, labels, and limits
        ax = plt.gca()
        ax.add_collection(p)
        if vertical_cbar:
            cbar = plt.colorbar(p, label=field_title, shrink=shrink_cbar, pad=pad_cbar)
        else:
            cbar = plt.colorbar(p, label=field_title, orientation="horizontal", shrink=shrink_cbar, pad=pad_cbar)
        cbar.formatter.set_powerlimits((-2, 4))
        if not isinstance(xlim, type(None)):
            ax.set_xlim(xlim)
        if not isinstance(ylim, type(None)):
            ax.set_ylim(ylim)
        xlab = xdir
        ylab = ydir
        if unit != '':
            xlab += f' [{unit}]'
            ylab += f' [{unit}]'
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        if isinstance(plot_title, type(None)):
            plt.title("{}={:.3f}".format([dir_id], val))
        else:
            plt.title(plot_title)
        ax.autoscale_view()
        if equal:
            ax.set_aspect("equal", adjustable="box")
        plt.tight_layout()
        if no_ax:
            plt.axis('off')

        # Save the figure if requested
        if save_fig:
            if isinstance(fig_name, type(None)):
                fig_path = f'{fig_folder}/2D_plot_{[dir_id]}{val:.2E}_{field}{fig_suffix}.png'
            else:
                fig_path = f'{fig_folder}/{fig_name}{fig_suffix}.png'
            plt.savefig(fig_path, dpi=fig_dpi, bbox_inches='tight')
        return ax 

    def plot3D(self, field='id', colormap='turbo', view=None, view_dist=None, xlim=None, ylim=None, zlim=None, transparent=False, translation=None, sample_fraction=None, fast=True, scatter_size=1, alpha=1, field_title=None, clim=None, shrink_cbar=1, pad_cbar=0.15, vertical_cbar=False, no_ax=False, plot_title=None, equal=True, fig_size=None, new_fig=True, save_fig=False, fig_folder='./', fig_name=None, fig_suffix='', fig_dpi=400):
        """Generates a 3D plot of the pebble bed."""
        # Prepare data for 3D plotting and configure plot settings
        data = self.data[self.data.r != 0]

        if not isinstance(sample_fraction, type(None)):
            data = data.sample(int(sample_fraction*len(data)))

        if isinstance(field_title, type(None)):
            field_title = field

        if isinstance(clim, type(None)):
            clim = [data[field].min(), data[field].max()]
        if new_fig:
            if not isinstance(fig_size, type(None)):
                plt.figure(figsize=fig_size, facecolor = "white")
            else:
                plt.figure()
            plt.gcf().add_subplot(111, projection="3d")
        ax = plt.gca()

        # Set plot appearance, apply transformations, and configure colorbar
        if not isinstance(translation, type(None)):
            data.x += translation[0]
            data.y += translation[1]
            data.z += translation[2]

        if not fast:
            values =  data[field]
            cmap = cm.get_cmap(colormap)
            norm = Normalize(vmin = clim[0], vmax = clim[1])
            normalized_values = norm(values)
            colors = cmap(normalized_values)

            i_row = 0
            for _, row in data.iterrows():
                # draw sphere
                u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
                x = row.r*np.cos(u)*np.sin(v)
                y = row.r*np.sin(u)*np.sin(v)
                z = row.r*np.cos(v)
                p = ax.plot_surface(x+row.x, y+row.y, z+row.z, color=colors[i_row], shade=False, alpha=alpha, zorder=1)
                i_row += 1
            if vertical_cbar:
                cb = plt.colorbar(p, label=field_title, shrink=shrink_cbar, pad=pad_cbar)
            else:
                cb = plt.colorbar(p, label=field_title, orientation="horizontal", shrink=shrink_cbar, pad=pad_cbar)
            cb.formatter.set_powerlimits((-2, 4))
        else:
            if isinstance(scatter_size, type(None)):
                raise Exception(f'Fast mode, needing manual parameter for scatter size')
            p = ax.scatter3D(
                data.x,
                data.y,
                data.z,
                s=scatter_size,
                c=data[field],
                alpha=alpha,
                zorder=1,
                vmin=clim[0],
                vmax=clim[1],
                cmap=colormap,
                edgecolor = None #'none'
            )
            if vertical_cbar:
                cb = plt.colorbar(p, label=field_title, shrink=shrink_cbar, pad=pad_cbar)
            else:
                cb = plt.colorbar(p, label=field_title, orientation="horizontal", shrink=shrink_cbar, pad=pad_cbar)

            cb.set_alpha(1)
            cb.formatter.set_powerlimits((-2, 4))
            plt.gcf().draw_without_rendering()

        # Set view angles, axis limits, and labels
        ax.set_xlabel("x [cm]")
        ax.set_ylabel("y [cm]")
        ax.set_zlabel("z [cm]")

        if not isinstance(view, type(None)):
            ax.view_init(view[0], view[1])
        if not isinstance(view_dist, type(None)):
            ax.dist = view_dist
        if isinstance(plot_title, type(None)):
            plt.title(f"{field}")
        else:
            plt.title(plot_title)

        # Optional adjustments for aspect ratio and axis visibility
        if equal:
            try:
                try:
                    ax.set_box_aspect((np.diff(ax.get_xlim())[0],np.diff(ax.get_ylim())[0],np.diff(ax.get_zlim())[0]))
                except:
                    ax.set_aspect('equal')
            except:
                pass
        ax.relim()      # make sure all the data fits
        ax.autoscale()  # auto-scale
        if not isinstance(xlim, type(None)):
            ax.set_xlim(xlim)
        if not isinstance(ylim, type(None)):
            ax.set_ylim(ylim)
        if not isinstance(zlim, type(None)):
            ax.set_zlim([zlim[0], zlim[1]])
        #ax.invert_zaxis()

        plt.tight_layout()
        if no_ax:
            plt.axis('off')

        # Save the figure if requested
        if save_fig:
            if isinstance(fig_name, type(None)):
                fig_path = f'{fig_folder}/3D_plot_{field}{fig_suffix}.png'
            else:
                fig_path = f'{fig_folder}/{fig_name}{fig_suffix}.png'
            plt.savefig(fig_path, dpi=fig_dpi, bbox_inches='tight', transparent=transparent)
        return ax

    def plot_summary(self, field='id', colormap='turbo', view=None, view_dist=10.5, xlim=None, ylim=None, zlim=None, translation=None, sample_fraction=None, fast=True, clipping_3D=True, shrink_cbar=1, pad_cbar=0.15, scatter_size=10, alpha=1, equal=True, field_title=None, clim=None, superimpose_Serpent=False, Serpent_xsize=None, Serpent_ysize=None, Serpent_zsize=None, Serpent_paths_dirxyz = [None, None, None], patches_clip_2D=[None, None, None], save_fig=False, fig_size=None, fig_folder='./', fig_name=None, fig_suffix='', fig_dpi=400):
        """Generates a summary plot consisting of 2D and 3D plots."""
        # Set up the overall figure for the summary plot
        if not isinstance(fig_size, type(None)):
            fig = plt.figure(figsize=fig_size)
        else:
            fig = plt.figure()

        if superimpose_Serpent:
            valx = np.mean(Serpent_xsize)
            valy = np.mean(Serpent_ysize)
            valz = np.mean(Serpent_zsize)
        else:
            valx = 'middle'
            valy = 'middle'
            valz = 'middle'
        if isinstance(clim, type(None)):
            clim = [np.min(self.data[field]), np.max(self.data[field])]

        # Generate individual 2D plots for different views
        ax1=fig.add_subplot(2,2,1)
        self.plot2D(field, dir_id='z', val=valz, plot_title='(X,Y)', translation=translation, superimpose_Serpent=superimpose_Serpent, Serpent_xsize=Serpent_xsize, Serpent_ysize=Serpent_ysize, Serpent_geom_path=Serpent_paths_dirxyz[2], new_fig=False, equal=equal, xlim=ylim, ylim=zlim, clim=clim, field_title=field_title, colormap=colormap, shrink_cbar=shrink_cbar, pad_cbar=pad_cbar, patch_clip=patches_clip_2D[2])
        ax2=fig.add_subplot(2,2,2)
        self.plot2D(field, dir_id='y', val=valy, plot_title='(X,Z)', translation=translation, superimpose_Serpent=superimpose_Serpent, Serpent_xsize=Serpent_xsize, Serpent_ysize=Serpent_zsize, Serpent_geom_path=Serpent_paths_dirxyz[1], new_fig=False, equal=equal, xlim=xlim, ylim=zlim, clim=clim, field_title=field_title, colormap=colormap, shrink_cbar=shrink_cbar, pad_cbar=pad_cbar, patch_clip=patches_clip_2D[1])
        ax3=fig.add_subplot(2,2,3)
        self.plot2D(field, dir_id='x', val=valx, plot_title='(Y,Z)', translation=translation, superimpose_Serpent=superimpose_Serpent, Serpent_xsize=Serpent_ysize, Serpent_ysize=Serpent_zsize, Serpent_geom_path=Serpent_paths_dirxyz[0], new_fig=False, equal=equal, xlim=xlim, ylim=ylim, clim=clim, field_title=field_title, colormap=colormap, shrink_cbar=shrink_cbar, pad_cbar=pad_cbar, patch_clip=patches_clip_2D[0])
        ax4=fig.add_subplot(2,2,4, projection="3d")

        # Generate a 3D plot or clipped 3D plot
        if not clipping_3D:
            self.plot3D(field, new_fig=False, xlim=xlim, ylim=ylim, zlim=zlim, translation=translation, clim=clim, field_title=field_title, colormap=colormap, view=view, view_dist=view_dist, sample_fraction=sample_fraction, fast=fast, equal=equal, scatter_size=scatter_size, alpha=alpha, shrink_cbar=shrink_cbar, pad_cbar=pad_cbar)
        else:
            clipped = self.clip(['x', 'y'], direction=[-1,+1], logical_and=False)
            clipped.plot3D(field, new_fig=False, xlim=xlim, ylim=ylim, zlim=zlim, translation=translation, clim=clim, field_title=field_title, colormap=colormap, view=view, sample_fraction=sample_fraction, fast=fast, equal=equal, scatter_size=scatter_size, alpha=alpha, shrink_cbar=shrink_cbar, pad_cbar=pad_cbar)

        # Adjust layout and save the figure if requested
        plt.tight_layout()
        if save_fig:
            if isinstance(fig_name, type(None)):
                fig_path = f'{fig_folder}/Summary_plot_{field}{fig_suffix}.png'
            else:
                fig_path = f'{fig_folder}/{fig_name}{fig_suffix}.png'
            plt.savefig(fig_path, dpi=fig_dpi, bbox_inches='tight')
        return (ax1, ax2, ax3, ax4)



#%% Zones utilities functions
def matrices(N_types, N_passes, N_axial, N_radial, uniform_axial_transfer, uniform_radial_transfer):
    """
    Generates a link matrix representing material transfers between different zones in a pebble bed reactor.
    The matrix captures axial and radial transfers for each pebble type and pass.
    """
    # Initialize the link matrix with extra axial zones for discarded pebbles
    link_matrix = np.zeros((N_types, N_passes, N_axial, N_radial, N_passes, N_axial, N_radial))

    # Iterate over pebble types, passes, axial and radial positions
    for pebble_type in range(N_types):
        for pass_num in range(N_passes):
            for axial_pos in range(N_axial):
                for radial_pos in range(N_radial):
                    matrix = link_matrix[pebble_type, pass_num, axial_pos, radial_pos]
                    rad_transfer = 0

                    # Handle axial and radial transfers
                    if axial_pos != 0:
                        matrix[pass_num, axial_pos - 1, radial_pos] = uniform_axial_transfer[pebble_type]

                        # Adjust for radial transfers
                        if radial_pos != 0:
                            matrix[pass_num, axial_pos - 1, radial_pos - 1] = uniform_radial_transfer[pebble_type]
                            rad_transfer += uniform_radial_transfer[pebble_type]
                        if radial_pos != N_radial - 1:
                            matrix[pass_num, axial_pos - 1, radial_pos + 1] = uniform_radial_transfer[pebble_type]
                            rad_transfer += uniform_radial_transfer[pebble_type]
                    else:
                        # Handle transfer to the next pass
                        if pass_num != 0:
                            matrix[pass_num - 1, N_axial - 1, :] = uniform_axial_transfer[pebble_type] / N_radial

                    # Update the current position transfer probability
                    matrix[pass_num, axial_pos, radial_pos] = 1 - uniform_axial_transfer[pebble_type] - rad_transfer

    return link_matrix


def plot_fluxes():
    df = pd.DataFrame(columns=['Axial', 'Radial', 'Value', 'Relative Uncertainty'])
    x = list(-np.sqrt(np.linspace(0, N_radial, N_radial+1)/(N_radial))*(N_radial))[-1:0:-1] + list(np.sqrt(np.linspace(0, N_radial, N_radial+1)/(N_radial))*(N_radial))
    y = range(N_axial+1)
    X, Y = np.meshgrid(x, y)
    arr = simulation.get_values(th_flux_zones_out).reshape(N_axial, N_radial)
    rev_arr = np.fliplr(arr)
    arr = np.concatenate((rev_arr, arr), axis=1)
    plt.pcolor(X, Y, arr, cmap='plasma')
    plt.xticks(x, np.abs(np.linspace(-N_radial, N_radial, 2*N_radial+1)).astype(int))        
    plt.title(f'Step #{time_idx}')
    cbar = plt.colorbar()
    cbar.formatter.set_powerlimits((0, 0))
    plt.savefig(f'./Plots/zone_flux_{time_idx}.png', dpi=400)
    #plt.show()
    plt.clf()
    plt.close('all')

    arr_unc = simulation.get_values(th_flux_zones_unc_out).reshape(N_axial, N_radial)
    rev_arr_unc = np.fliplr(arr_unc)
    arr_unc = np.concatenate((rev_arr_unc, arr_unc), axis=1)
    plt.pcolor(X, Y, arr_unc, cmap='plasma')
    plt.xticks(x, np.abs(np.linspace(-N_radial, N_radial, 2*N_radial+1)).astype(int))        
    plt.title(f'Step #{time_idx}')
    cbar = plt.colorbar()
    cbar.formatter.set_powerlimits((0, 0))
    plt.savefig(f'./Plots/zone_flux_unc_{time_idx}.png', dpi=400)
    #plt.show()
    plt.clf()
    plt.close('all')
    for axial_pos in range(N_axial):
        for radial_pos in range(N_radial):
            df.loc[len(df.index), ] = [axial_pos, radial_pos, arr[axial_pos, radial_pos], arr_unc[axial_pos, radial_pos]]
    df.to_csv(f'./Data/data_zone_fluxes_{time_idx}.csv')
    return df

def plot_powers():
    df = pd.DataFrame(columns=['Axial', 'Radial', 'Value', 'Relative Uncertainty'])
    x = list(-np.sqrt(np.linspace(0, N_radial, N_radial+1)/(N_radial))*(N_radial))[-1:0:-1] + list(np.sqrt(np.linspace(0, N_radial, N_radial+1)/(N_radial))*(N_radial))
    y = range(N_axial+1)
    X, Y = np.meshgrid(x, y)
    arr = simulation.get_values(power_zones_out).reshape(N_axial, N_radial)
    rev_arr = np.fliplr(arr)
    arr = np.concatenate((rev_arr, arr), axis=1)
    plt.pcolor(X, Y, arr, cmap='plasma')
    plt.xticks(x, np.abs(np.linspace(-N_radial, N_radial, 2*N_radial+1)).astype(int))        
    plt.title(f'Step #{time_idx}')
    cbar = plt.colorbar()
    cbar.formatter.set_powerlimits((0, 0))
    plt.savefig(f'./Plots/zone_power_{time_idx}.png', dpi=400)
    #plt.show()
    plt.clf()
    plt.close('all')

    arr_unc = simulation.get_values(power_zones_unc_out).reshape(N_axial, N_radial)
    rev_arr_unc = np.fliplr(arr_unc)
    arr_unc = np.concatenate((rev_arr_unc, arr_unc), axis=1)
    plt.pcolor(X, Y, arr_unc, cmap='plasma')
    plt.xticks(x, np.abs(np.linspace(-N_radial, N_radial, 2*N_radial+1)).astype(int))        
    plt.title(f'Step #{time_idx}')
    cbar = plt.colorbar()
    cbar.formatter.set_powerlimits((0, 0))
    plt.savefig(f'./Plots/zone_power_unc_{time_idx}.png', dpi=400)
    #plt.show()
    plt.clf()
    plt.close('all')
    for axial_pos in range(N_axial):
        for radial_pos in range(N_radial):
            df.loc[len(df.index), ] = [axial_pos, radial_pos, arr[axial_pos, radial_pos], arr_unc[axial_pos, radial_pos]]
    df.to_csv(f'./Data/data_zone_power_{time_idx}.csv')
    return df

def plot_zones(type_plot, suffix):
  df = pd.DataFrame(columns=['Zone', 'Type', 'Pass', 'Axial', 'Radial', 'Value'])
  x = list(-np.sqrt(np.linspace(0, N_radial, N_radial+1)/(N_radial))*(N_radial))[-1:0:-1] + list(np.sqrt(np.linspace(0, N_radial, N_radial+1)/(N_radial))*(N_radial))
  y = range(N_axial+1)
  X, Y = np.meshgrid(x, y)
  for pebble_type in range(N_types):
      if is_fuel[pebble_type]:
        if type_plot.lower()=='fima':
          matrix = simulation.get_values(fima_out[pebble_type]).reshape(N_radial, N_axial, N_passes).T*100
        elif type_plot.lower()=='u235':
          compositions = simulation.get_values(compositions_out[pebble_type])
          compositions = np.moveaxis(compositions.reshape(N_radial, N_axial, N_passes, -1), [0,1,2], [2,1,0])
          matrix = compositions[:,:,:,i_U235[pebble_type]]
        elif type_plot.lower()=='bu':
          matrix = simulation.get_values(bu_out[pebble_type]).reshape(N_radial, N_axial, N_passes).T
        for pass_num in range(N_passes):
          arr = matrix[pass_num]
          rev_arr = np.fliplr(arr)
          arr = np.concatenate((rev_arr, arr), axis=1)
          plt.pcolor(X, Y, arr, cmap='plasma')
          plt.xticks(x, np.abs(np.linspace(-N_radial, N_radial, 2*N_radial+1)).astype(int))
          plt.title(f'Step #{time_idx}.{substep_idx}: type {pebble_type+1}, pass {pass_num+1}')
          cbar = plt.colorbar()
          cbar.formatter.set_powerlimits((0, 0))
          plt.savefig(f'./Plots/{type_plot.lower()}_{time_idx}_{substep_idx}_{pebble_type+1}_{pass_num+1}_{suffix}.png', dpi=200)
          plt.clf()
          plt.close('all')
          for axial_pos in range(N_axial):
            for radial_pos in range(N_radial):
              df.loc[len(df.index)] = [zones_map[pebble_type, pass_num, axial_pos, radial_pos], pebble_type, pass_num, axial_pos, radial_pos, matrix[pass_num, axial_pos, radial_pos]]

        arr = np.mean(matrix, axis=0)
        rev_arr = np.fliplr(arr)
        arr = np.concatenate((rev_arr, arr), axis=1)
        plt.pcolor(X, Y, arr, cmap='plasma')
        plt.xticks(x, np.abs(np.linspace(-N_radial, N_radial, 2*N_radial+1)).astype(int))        
        plt.title(f'Step #{time_idx}.{substep_idx}: type {pebble_type+1}')
        cbar = plt.colorbar()
        cbar.formatter.set_powerlimits((0, 0))
        plt.savefig(f'./Plots/{type_plot.lower()}_{time_idx}_{substep_idx}_{pebble_type+1}_glob_{suffix}.png', dpi=200)
        #plt.show()
        plt.clf()
        plt.close('all')
  df.to_csv(f'./Data/data_{type_plot}_{time_idx}_{suffix}.csv')
  return df

def assign_random_array(source_list, possible_values, proportions='equal'):
	if isinstance(source_list, int):
		source_list = np.arange(source_list)

	if isinstance(proportions, str) and proportions=='equal':
		shuffled_list = np.array_split(np.random.permutation(source_list), len(possible_values))
	else:
		proportions = np.array(proportions) / np.sum(proportions)
		random_list = np.random.permutation(np.arange(len(source_list)))
		ind = np.add.accumulate(np.array(proportions) * len(random_list)).round(8).astype(int)
		shuffled_list = [x.tolist() for x in np.split(random_list, ind)][:len(proportions)]
	array = np.empty(len(source_list), dtype=np.array(possible_values).dtype)
	for i, val in enumerate(possible_values):
		for j in range(len(shuffled_list[i])):
			array[shuffled_list[i][j]] = val
	return array

#%% Input parameters
# Files
case_name = 'gFHR_test' # Name for folder to create
main_input_file = 'input_test' # Name of the main Serpent input file (must be in same folder as script)
pos_file = 'fpb_pos' # Name of pebbles position file (must be in same folder as script)

# Zones
N_passes = 8 # Number of passes to simulate (might want to add it to "division" later, or "material" so that the threshold can be pebble type-wise)
division = {'radial':{'Nzones':4, 'bounds':[0, 120]}, # Radial and axial division, number of zones and boundaries. Automatic division by Serpent in equal volume zones
            'axial': {'Nzones':8, 'bounds':[60, 369.47]}}
N_radial = division['radial']['Nzones'] # Number of radial zones
N_axial = division['axial']['Nzones'] # Number of axial zones

# Materials/Universes
materials = {'fuel':'u_fuel_pebble', # material names for each pebble type and corresponding universe (fresh material and universes must be defined in Serpent input)
             'fuel2':'u_fuel2_pebble',
             'graphite':'u_graph_pebble'}
fracs = [0.8, 0.1, 0.1] # fraction of pebbles in the pebble bed for each pebble type (might want to convert it to dictionary)
is_fuel = [True, True,False] # type of pebble (burnable or not) in the pebble bed for each pebble type (might want to convert it to dictionary)
vol_per_pebble = [0.362644, 0.362644,None] # volume of material in the pebble bed for each pebble type (might want to convert it to dictionary)
pbed_universe = 'u_pb' # name of the pebble bed universe
N_types = len(materials) # Number of pebble types
material_names = list(materials.keys()) # name of each pebble material

# Simulation
sss_exe = "/mnt/c/Users/yv3s9/HxF_tools/serpent2.2.0_HxF/sss2" # path to Serpent executable (HxF_tools version)
MPI_mode = False # if using MPI, turn it on
ncores = 20 # number of omp cores per node
ini_neutrons = 2500 # Initial neutrons per cycle, will get multiplied by multiplier after each transport
neutrons_multiplier = 1.5 # Multiplier for number of neutrons per cycle (typically, >=1 for more and more precise)
max_neutrons = 30000 # Limit not to exceed for number of neutrons per cycle (will not get multiplier over that number)

# Zones transfer rate
res_time = 522 # Time spent in core over all passes in days
t_zone = res_time/N_passes/N_axial # Time spent in a zone in days
uniform_axial_transfer_per_day=[1/t_zone] * N_types # For each pebble type, fraction of the previous axial zone being fed to the current zone each day (sets feeding and discarding as well)
uniform_radial_transfer_per_day=[0] * N_types # For each pebble type, fraction of the neighboring radial zones in the previous axial zone being fed to the current zone each day (to account for cross-mixing)
ini_motion_step = t_zone * 86400 # First motion/mixing/depletion step in seconds, will get multiplied by multiplier after each transport
step_multiplier = 0.9 # Multiplier for motion step (typically, <=1 for finer and finer mixing steps)

# Convergence
eps = 1e-5 # substeps (motion steps) convergence criterion to go out of inner loop
mini_substeps = N_axial*N_passes/2 # minimum number of steps before convergence
n_converged_substeps = 5 # number of steps satisfying criterion before going out for inner loop

eps_keff = 10e-5 # steps (transport steps) convergence criterion to go out of outer loop and end simulation

# Communication
verbosity_mode = 0 # Set if cerberus talks or not
cerberus.LOG.set_verbosity(verbosity_mode) 
plot_every = 30 # Plot zone-wise data every # of steps

#%% Initialize Serpent and create pebble bed object and transferables

# Start Serpent
simulation = Simulation()
simulation.start(case_name, main_input_file, pos_file, materials, fracs, is_fuel, pbed_universe, division, N_passes)

# Create pbed object
pbed = Pebble_bed()
pbed.read_pbed_file(pos_file)

# Get created detectors
th_flux_out = simulation.get_tra(f"DET_pebble_flux")
th_flux_unc_out = simulation.get_tra(f"DET_pebble_flux_rel_unc")
th_flux_zones_out = simulation.get_tra(f"DET_zone_flux")
th_flux_zones_unc_out = simulation.get_tra(f"DET_zone_flux_rel_unc")
power_zones_out = simulation.get_tra(f"DET_zone_power")
power_zones_unc_out = simulation.get_tra(f"DET_zone_power_rel_unc")

# Find which materials are fuel and get their fresh compositions
nuclides_lists = []
fresh_materials = []
for i in range(N_types): 
    if is_fuel[i]:
        nuclides_list = simulation.get_values(f'composition_{material_names[i]}z1_nuclides') # Extract nuclides list for each type
        nuclides_lists.append(nuclides_list)
        fresh_materials.append(np.array(simulation.get_values(f'composition_{material_names[i]}z1_adens'))) # Extract fresh composition for each type
    else:
        fresh_materials.append(None)
        nuclides_lists.append(None)

# Set materials volumes
# WARNING: for now, we assume uniform motion (simplified matrix) and same volume in each zone (partial pebble volumes even though they are not accounted in multiple zones, small assumption though)
for i in range(N_types):
    if is_fuel[i]:
        total_vol = vol_per_pebble[i] * len(pbed.data) * fracs[i] # Volume of a given pebble type in the core
        vol_per_zone = total_vol / (N_passes*N_axial*N_radial) # Volume of a given pebble type in a zone (assumed uniform, okay for gFHR)
        for j in range(N_passes*N_axial*N_radial):
            simulation.set_values(simulation.get_tra(f'material_{material_names[i]}z{j+1}_volume', input_parameter=True), vol_per_zone)

# Switch to change between transport/activation steps during simulation
switch_mode = simulation.get_tra('burn_step_type', input_parameter=True)

# Switch to write restart files at each converged step
restart_writer = simulation.get_tra('write_restart', input_parameter=True)

# Important transferrables for zones
compositions_out = []
# fima_out = []
bu_out=[]
compositions_in = np.empty((N_types, N_passes, N_axial, N_radial), dtype=object)
bu_in = np.empty((N_types, N_passes, N_axial, N_radial), dtype=object)
zones_map = np.empty((N_types, N_passes, N_axial, N_radial), dtype=int)
for pebble_type in range(N_types):
  if is_fuel[pebble_type]:
    compositions_out.append(simulation.get_tra(f'composition_div_{material_names[pebble_type]}_adens')) # To get composition vector for all zones for a given pebble type
    # fima_out.append(simulation.get_tra(f'material_div_{material_names[pebble_type]}_fima'))  # To get fima vector for all zones for a given pebble type
    bu_out.append(simulation.get_tra(f'material_div_{material_names[pebble_type]}_burnup'))  # To get burnup vector for all zones for a given pebble type
  else:
    compositions_out.append(None)
    # fima_out.append(None)
    bu_out.append(None)

  for pass_num in range(N_passes):
    for axial_pos in range(N_axial):
      for radial_pos in range(N_radial):
        i_zone = pass_num + axial_pos*N_passes + radial_pos*N_axial*N_passes + 1
        zones_map[pebble_type, pass_num, axial_pos, radial_pos] = i_zone # Map zones number to each material zone name (not used much, but can be very useful for advanced volumes/motion/geometries)
        if is_fuel[pebble_type]:
            compositions_in[pebble_type, pass_num, axial_pos, radial_pos] = simulation.get_tra(f'composition_{material_names[pebble_type]}z{i_zone}_adens', input_parameter=True) # To set composition vector for all zones for a given pebble type
            bu_in[pebble_type, pass_num, axial_pos, radial_pos] = simulation.get_tra(f'material_{material_names[pebble_type]}z{i_zone}_burnup', input_parameter=True) # To set burnup vector for all zones for a given pebble type
        else:
            compositions_in[pebble_type, pass_num, axial_pos, radial_pos] = None
            bu_in[pebble_type, pass_num, axial_pos, radial_pos] = None

# keff transferrable
keff = simulation.get_tra('ANA_KEFF') # To get multiplication factor
keff_rel_err = simulation.get_tra('ANA_KEFF_rel_unc')  # To get multiplication factor relative uncertainty

# Extract first concentrations
compositions = simulation.get_multiple_values(compositions_out)
for pebble_type in range(N_types):
  if is_fuel[pebble_type]:
    compositions[pebble_type] = np.moveaxis(compositions[pebble_type].reshape(N_radial, N_axial, N_passes, -1), [0,1,2], [2,1,0]) # This reshaping is used to get the compositions in the right zones order
  else:
    compositions[pebble_type] = None

# Initialize time and quantities trackers
time_idx=0
current_motion_step = float(ini_motion_step)
curr_time = 0
simulation.solver.set_current_time(curr_time) # Initialize simulation time

discarded_bu_table = [pd.DataFrame(columns=list(np.arange(1, N_radial+1, dtype=int))+['Avg']) for pebble_type in range(N_types)] # Will be used to track discarded burnups per radial zone
discarded_table = [pd.DataFrame(index=nuclides_lists[pebble_type]) for pebble_type in range(N_types)]  # Will be used to track discarded composition (mixed between radial zones)
keff_evol = [] # Will be used to track the multiplication factor evolution
keff_err_evol = []

# Set initial number of neutrons
neutrons_per_cycle_in = simulation.get_tra('neutrons_per_cycle', input_parameter=True)
current_neutrons = int(ini_neutrons)
simulation.set_values(neutrons_per_cycle_in, current_neutrons)

#%% Main Loop
while True:
  # Run transport and very quick depletion (~seconds) to obtain cross sections
  print(f'Step {time_idx}')
  print(f'\tNeutrons per cycle: {current_neutrons}')
  print(f'\tMotion step: {current_motion_step/86400:.2E} days')
  curr_time = simulation.deplete(curr_time, SHORT_STEP, TRANSPORT, switch_mode) # very short step, just to obtain XS distribution, which we assume to be constant at equilibrium (regardless of motion, fair assumption)

  # Get fluxes and plot 3D map of it (pebble-wise uncertainties do not matter much, the zones uncertainties are important)
  # Pebble-wise
  pbed.add_field('pebble_flux', simulation.get_values(th_flux_out))
  pbed.plot_summary('pebble_flux', fig_size=(10,10), save_fig=True, fig_suffix=f'_{time_idx}', fig_folder='./Plots/', pad_cbar=0.15, scatter_size=1)
  pbed.add_field('pebble_flux_unc', simulation.get_values(th_flux_unc_out))
  pbed.plot_summary('pebble_flux_unc', fig_size=(10,10), save_fig=True, fig_suffix=f'_{time_idx}', fig_folder='./Plots/', pad_cbar=0.15, scatter_size=1)
  # Zone-wise
  plot_fluxes()
  plot_powers()

  # Extract current keff and plot
  keff_evol.append(simulation.get_values(keff)[0])
  keff_err_evol.append(keff_evol[-1]*simulation.get_values(keff_rel_err)[0])
  print(f'keff = {keff_evol[-1]:.5f} +/- {keff_err_evol[-1]*1e5:.0f} pcm')
  np.savetxt('./Data/keff_evol.csv', keff_evol, delimiter=',')
  np.savetxt('./Data/keff_err_evol.csv', keff_err_evol, delimiter=',')
  plt.errorbar(np.arange(len(keff_evol)), keff_evol, yerr=keff_err_evol)
  plt.xlabel('Outer iteration #')
  plt.ylabel('Multiplication factor')
  plt.savefig(f'./Plots/keff.png')

  # Outer convergence test, if keff converges, equilibrium is found and calculation ends
  if time_idx != 0:
    rel_dif = (keff_evol[-1] - keff_evol[-2])/keff_evol[-2]
    print(f'Relative keff difference = {rel_dif*1e5:.0f} pcm')
    if np.abs(rel_dif) <= eps_keff:
      break # Test if calculation ends
  
  # Do activation/motion steps until inner convergence
  substep_idx = 0
  
  # Make updated motion matrices, based on the current motion step
  uniform_axial_transfer = [i*current_motion_step/86400 for i in uniform_axial_transfer_per_day] # update for each pebble type
  uniform_radial_transfer = [i*current_motion_step/86400 for i in uniform_radial_transfer_per_day] # update for each pebble type
  link_matrix = matrices(N_types, N_passes, N_axial, N_radial, uniform_axial_transfer, uniform_radial_transfer) # make motion matrix
  rel_dif_norm_evol = [] # track the relative difference on the convergence metric
  while True:
    # Extract compositions and burnup for each zone
    compositions = simulation.get_multiple_values(compositions_out)
    bu = simulation.get_multiple_values(bu_out)
    for pebble_type in range(N_types):
      if is_fuel[pebble_type]:
        compositions[pebble_type] = np.moveaxis(compositions[pebble_type].reshape(N_radial, N_axial, N_passes, -1), [0,1,2], [2,1,0])
        bu[pebble_type] = np.moveaxis(bu[pebble_type].reshape(N_radial, N_axial, N_passes), [0,1,2], [2,1,0])
      else:
        compositions[pebble_type] = None
        bu[pebble_type] = None

    # Motion step
    new_compositions = []
    new_bu = []
    for pebble_type in range(N_types):
        if is_fuel[pebble_type]:
            new_compositions.append(np.ones_like(compositions[pebble_type]))
            new_bu.append(np.ones_like(bu[pebble_type]))
            for pass_num in range(N_passes):
                for axial_pos in range(N_axial):
                    for radial_pos in range(N_radial):
                        matrix = link_matrix[pebble_type, pass_num, axial_pos, radial_pos] # fraction of current zone coming from other zones: part of the motion matrix corresponding to the current type, pass number, axial and radial zone
                        feeding = np.einsum('ijk,ijkl->l',matrix,compositions[pebble_type]) # mix of the other zones which feed current zone
                        fueling = (1-np.sum(matrix))*fresh_materials[pebble_type] # fraction of current zone coming from fresh fuel
                        new_compositions[pebble_type][pass_num, axial_pos, radial_pos] = feeding + fueling # mix of fresh fuel and the mix from other zones
                        new_bu[pebble_type][pass_num, axial_pos, radial_pos] = np.einsum('ijk,ijk',matrix,bu[pebble_type]) # same with burnup, mix of other zones (fresh fuel counts as 0)
                        simulation.set_values(compositions_in[pebble_type, pass_num, axial_pos, radial_pos], new_compositions[pebble_type][pass_num, axial_pos, radial_pos]) # communicate to Serpent
                        simulation.set_values(bu_in[pebble_type, pass_num, axial_pos, radial_pos], new_bu[pebble_type][pass_num, axial_pos, radial_pos]) # communicate to Serpent
        else:
            new_compositions.append(None)
            new_bu.append(None)

    if substep_idx%plot_every==0:
      burnup = plot_zones('bu','before') # Plot when necessary the zones burnups per pass and average (can be changed, isotopes concentrations can be added for instance)

    # Activation step to deplete with current flux distribution
    curr_time = simulation.deplete(curr_time, current_motion_step, ACTIVATION, switch_mode) # Once compositions were moved, burn for a time corresponding to motion step (no corrector)

    # Extract data and track
    discarded_bu = []
    discarded = []
    compositions = simulation.get_multiple_values(compositions_out) # To make sure things work properly, do not use the new_compositions, but rather extract it from Serpent (if removed, faster, but no way to check)
    bu = simulation.get_multiple_values(bu_out) # Same with burnup
    for pebble_type in range(N_types):
      if is_fuel[pebble_type]:
        compositions[pebble_type] = np.moveaxis(compositions[pebble_type].reshape(N_radial, N_axial, N_passes, -1), [0,1,2], [2,1,0])
        bu[pebble_type] = np.moveaxis(bu[pebble_type].reshape(N_radial, N_axial, N_passes), [0,1,2], [2,1,0])
        
        # Update radial zone-wise discarded BU and discarded compositions mix, and save/plot
        discarded.append(np.zeros(compositions[pebble_type].shape[-1]))
        discarded_bu.append(np.zeros(N_radial))
        for radial_pos in range(N_radial):
          discarded_bu[pebble_type][radial_pos] = bu[pebble_type][-1, -1, radial_pos]
          discarded[pebble_type] += compositions[pebble_type][-1, -1, radial_pos]/N_radial
        # Save
        discarded_bu_table[pebble_type].loc[f'{time_idx}_{substep_idx}', range(1, N_radial+1)] = discarded_bu[pebble_type]
        discarded_bu_table[pebble_type].loc[f'{time_idx}_{substep_idx}', 'Avg'] = discarded_bu_table[pebble_type].loc[f'{time_idx}_{substep_idx}', range(1, N_radial+1)].mean()
        discarded_bu_table[pebble_type].to_csv(f'./Data/discarded_bu_{pebble_type+1}.csv')
        # Plot
        plt.plot(discarded_bu_table[pebble_type].loc[:, range(1, N_radial+1)].values, label=range(1, N_radial+1))
        plt.plot(discarded_bu_table[pebble_type].loc[:, 'Avg'].values, linestyle='--', color='k', label='Avg')
        plt.grid()
        plt.xlabel('Iteration #')
        plt.ylabel('Burnup [MWd/kg]')
        plt.title(f'Type {pebble_type+1} discard burnup')
        plt.legend(title='Radial zone')
        plt.savefig(f'./Plots/discarded_bu_evol_{pebble_type+1}.png')

        if substep_idx%plot_every==0: # Update discarded table if necessary and plot top 20 composition (can be changed)
            discarded_table[pebble_type][f'{time_idx}_{substep_idx}'] = discarded[pebble_type]
            discarded_table[pebble_type].to_csv(f'./Data/discarded_{pebble_type+1}.csv')
            discarded_table[pebble_type].iloc[:,-1].sort_values().iloc[-20:].plot.bar(log=True, grid=True, ylabel='N [at/b.cm]')
            plt.tight_layout()
            plt.savefig(f'./Plots/discarded_{time_idx}_{pebble_type+1}.png', dpi=200, bbox_inches='tight')
      else:
        bu[pebble_type] = None
        discarded.append(None)
        discarded_bu.append(None)
    
    # Check for inner loop (equilibrium) convergence, did we reach an equilibrium with current flux distribution?
    if substep_idx != 0:
        # Metric based on norm of the relative variation of each last pass radial zone's discarded burnup
        rel_dif_list = [np.linalg.norm((discarded_bu_table[i].iloc[-1,:-1] - discarded_bu_table[i].iloc[-2,:-1]) / discarded_bu_table[i].iloc[-2,:-1]) for i in range(N_types) if is_fuel[i]]
        rel_dif_norm = np.linalg.norm(rel_dif_list)
        rel_dif_norm_evol.append(rel_dif_norm)
        print(f'\t[{substep_idx}] Variation = {rel_dif_norm:.2E}')
        # Check for convergence
        if substep_idx > mini_substeps and substep_idx > n_converged_substeps and np.all(np.abs(rel_dif_norm_evol)[-n_converged_substeps:] <= eps):
            print('\tConverged!')
            # Plot and update/save latest current step data
            burnup = plot_zones('bu','after')
            for pebble_type in range(N_types):
                if is_fuel[pebble_type]:
                    discarded_table[pebble_type][f'{time_idx}_{substep_idx}'] = discarded[pebble_type]
                    discarded_table[pebble_type].to_csv(f'./Data/discarded_{pebble_type+1}.csv')
                    discarded_table[pebble_type].iloc[:,-1].sort_values().iloc[-20:].plot.bar(log=True, grid=True, ylabel='N [at/b.cm]')
                    plt.tight_layout()
                    plt.savefig(f'./Plots/discarded_{time_idx}_{pebble_type+1}.png', dpi=200, bbox_inches='tight')
            simulation.set_values(restart_writer, time_idx)
            break # Inner convergence
            
    if substep_idx%plot_every==0:
      burnup = plot_zones('bu','after')

    substep_idx += 1
    plt.clf()
    plt.close('all')
  time_idx += 1
  
  # Adjust variables for next step
  current_neutrons = min(int(current_neutrons*neutrons_multiplier), max_neutrons)
  current_motion_step *= step_multiplier
  simulation.set_values(neutrons_per_cycle_in, current_neutrons)
