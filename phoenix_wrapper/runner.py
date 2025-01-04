import os
import sys
SELF_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SELF_PATH)

import numpy as np
import json
import subprocess
import matplotlib.pyplot as plt

from builder import build as build_phoenix_exec
from builder import update as update_phoenix_exec

# class that runs, plots, loads and saves matrices, etc.
# should also work in jupyter notebooks

class PhoenixSolver:

    def __init__(self, *, working_directory: str | None = None, platform: str = "windows"):
        self.m_path = SELF_PATH
        self.m_platform = platform
        self.m_executables = list()
        self.m_settings = dict()
        self.m_cwd = os.getcwd() if working_directory is None else working_directory

    def build(self, targets: list[str] = ["cpu", "gpu"]):
        build_phoenix_exec("https://github.com/Schumacher-Group-UPB/PHOENIX", "master", targets, 4)

    def update_executables(self, exec_paths: list[str] | str = list()):
        if isinstance(exec_paths, str):
            exec_paths = [exec_paths]
        self.m_executables = update_phoenix_exec(["phoenix_repository", "phoenix_fallback"] + exec_paths, "exe" if self.m_platform == "windows" else "out")

        if not len(self.m_executables):
            print("No executables found. Please build the project first using the .build() method or provide a valid executable.")
    
    def get_executable(self, target: str = "cpu", precision: str = "fp32") -> str:
        if not len(self.m_executables):
            self.update_executables()
        
        target = target.lower()
        precision = precision.lower()

        # find the first executable that matches the target and precision
        for exec_path in [path.lower() for path in self.m_executables]:
            if target in exec_path and precision in exec_path:
                return exec_path
        
        print(f"Could not find executable for target: {target} and precision: {precision}")
        return None

    def build_runstring(self) -> str:
        """
        Settings["parameters"] : dict includes key values --> --key *values
        Settings["flags"] : list includes boolean flags --> -flag
        """
        runstring = ""
        for key, value in self.m_settings["parameters"].items():
            if isinstance(value, (list,tuple)):
                if isinstance(value[0], (list,tuple)):
                    for v in value:
                        runstring += f" --{key} "
                        for vv in v:
                            if isinstance(vv, str) and " " in vv:
                                runstring += f"'{vv}' "
                            else:
                                runstring += f"{vv} "
                else:
                    runstring += f" --{key} "
                    for v in value:
                        if isinstance(v, str) and " " in v:
                            runstring += f"'{v}' "
                        else:
                            runstring += f"{v} "
            else:
                if isinstance(value, str) and " " in value:
                    runstring += f" --{key} '{value}' "
                else:
                    runstring += f" --{key} {value} "
        for flag in self.m_settings["flags"]:
            runstring += f" -{flag}"
        return runstring.replace("  ", " ")
    
    def export_settings(self, path: str | None = None) -> dict:
        """
        Export the solver settings either to file or return them
        """

        if path is None:
            return self.m_settings
        with open(path, "w") as f:
            json.dump(self.m_settings, f)
        return self.m_settings

    def import_settings(self, *, settings: dict | None = None, parameters: dict | None = None, flags: list[str] | None = None, path: str | None = None) -> dict:
        """
        Only valid way to update the solver settings
        Does validation of settings general structure. Does not validate the values of the settings.
        """

        if settings is not None:
            self.m_settings.update(settings)

        # Make sure settings contain parameters and flags
        self.validate_settings()

        if parameters is not None:
            self.m_settings["parameters"].update(parameters)
        if flags is not None:
            self.m_settings["flags"].extend(flags)
        
        if path is not None:
            with open(path, "r") as f:
                loaded_settings = json.load(f)
                self.m_settings.update(loaded_settings)

        # Do a final validation
        return self.validate_settings()

    def reset_settings(self):
        """
        Resets settings to empty dict
        """
        self.m_settings = dict()
        return self.validate_settings()

    def validate_settings(self):
        if not isinstance(self.m_settings, dict):
            self.reset_settings()
        if "parameters" not in self.m_settings:
            self.m_settings["parameters"] = dict()
        if "flags" not in self.m_settings:
            self.m_settings["flags"] = list()

        # Relative path from the cwd to the cwd of the solver
        rel_path = os.path.relpath(self.m_cwd, os.getcwd())
        self.m_settings["parameters"]["path"] = rel_path
        return self.m_settings
    
    def set_working_directory(self, path: str):
        self.m_cwd = path
        return self.validate_settings()
    
    def hints(self):
        # if some commong variables are not set, provide hints
        pass

    def quick_run(self, *, target: str = "gpu", precision: str = "fp32", executable = None):
        """
        Provide target and or precision, or provide an executable
        """
        # set animatable = true/false
        # set plotable = true/false
        # suppress output

        # Find valid executable
        if executable is None:
            executable = self.get_executable(target, precision)
        if executable is None:
            raise Exception("No executable provided/found. Please build the project first using the .build() method or provide a valid executable.")
        
        # Build runstring
        runstring = self.build_runstring()
        runstring = (f"./{executable} " + runstring).replace("  ", " ")

        # Run the executable
        process = subprocess.check_call(runstring, shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return process
    
    def is_ipython(self):
        try:
            __IPYTHON__
            return True
        except NameError:
            return False

    def run(self, *, target: str = "gpu", precision: str = "fp32", executable = None, output_progress: bool = True, output_verbose: bool = False):
        
        local_ipython = False
        if output_progress and self.is_ipython():
            from tqdm import tqdm
            from tqdm.notebook import tqdm as notebook_tqdm
            local_ipython = True

        # Find valid executable
        if executable is None:
            executable = self.get_executable(target, precision)
        if executable is None:
            raise Exception("No executable provided/found. Please build the project first using the .build() method or provide a valid executable.")
        
        # Build runstring
        runstring = self.build_runstring()
        runstring = (f"./{executable} " + runstring).replace("  ", " ")
        print(runstring)
        process = subprocess.Popen(
            runstring, 
            shell=False, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True  # Ensures text mode instead of binary mode
        )

        if local_ipython:
            progressbar = notebook_tqdm(total=100, desc="Progress", unit="%", leave=True)

        try:
            summary = None
            # Read the output line by line as it is produced
            for line in process.stdout:
                if output_verbose:
                    print(line.strip())  # Print the output line-by-line in real-time
                if "Runtime" in line and "remaining" in line:
                    runtime = int(line.split(",")[0].split(":")[1].strip()[:-1])
                    remaining = int(line.split(",")[1].split(":")[1].strip()[:-1])
                    progress_percent = int(100 * runtime / (runtime + remaining))

                    if local_ipython:
                        progressbar.n = progress_percent
                        progressbar.refresh()
                    else:
                        print(f"Progress: {progress_percent}%", end="\r")
                elif "Time per ps" in line:
                    summary = line.strip()
            
            # Wait for the process to complete and get the exit code
            process.wait()
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # Close the streams
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, runstring)
        
        if local_ipython:
            progressbar.n = 100
            progressbar.refresh()
            progressbar.close()

        return summary

    def plot_matrix(self, *, matrix: np.ndarray | None = None, name: str = "", sub_path: str = "", 
                    # Figure Parameters
                    figure = None, axes = None, figsize: tuple = (8,6), 
                    # Labeling
                    colorbar: bool = True, xlabel: str = "", ylabel: str = "", title: str = "", cblabel: str = "",
                    # Format
                    use_log: bool = False, use_abs: bool = False, use_phase: bool = False, pixel_coords: bool = False, use_fft: bool = False, fft_shift: bool = False, k_space: bool = False,
                    # Plotting Parameters
                    **kwargs) -> tuple[plt.figure, plt.axes]:
        # plot a matrix using matplotlib
        if matrix is None:
            try:
                matrix = self.load_matrix(name, sub_path=sub_path)
            except Exception as e:
                print(f"Error loading matrix: {e}")
                return None
        
        # Generate a valid figure and ax
        if figure is None and axes is None:
            figure, axes = plt.subplots(figsize=figsize)
        if figure is None and axes is not None:
            figure = axes.figure
        if axes is None:
            axes = figure.add_subplot(111)
        
        axes.set_aspect('equal', 'box')

        # Transform data
        data = matrix.copy()
        if use_fft:
            data = np.fft.fft2(data)
        if fft_shift:
            data = np.fft.fftshift(data)
        if use_phase:
            data = np.angle(data)
        if use_abs:
            data = np.abs(data)
        if use_log:
            data = np.log10(data)
        
        # Set pixel coordinates
        if not pixel_coords:
            Lx,Ly = self.m_settings["parameters"].get("L", (1.0,1.0))
            Nx,Ny = self.m_settings["parameters"].get("N", (matrix.shape[1], matrix.shape[0]))
            X = np.linspace(-Lx/2, Lx/2, Nx)
            Y = np.linspace(-Ly/2, Ly/2, Ny)
        elif k_space:
            X = np.fft.fftfreq(matrix.shape[1])
            Y = np.fft.fftfreq(matrix.shape[0])
        else:
            X = np.arange(matrix.shape[1])
            Y = np.arange(matrix.shape[0])

        # if the matrix is complex and no phase nor abs is used, raise an error
        if not use_phase and not use_abs and np.iscomplexobj(data):
            raise ValueError("Complex matrices must be plotted using either the phase or the absolute value. Set use_phase or use_abs to True.")

        # Plot the matrix
        im = axes.pcolor(X, Y, data, **kwargs)
        if colorbar:
            cb = figure.colorbar(im, ax=axes)
            cb.set_label(cblabel)

        # Set labels
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_title(title)

        return figure, axes

    def plot_scalar(self, figure = None, axes = None, figsize: tuple = (12,5), **kwargs):
        try:
            times = np.loadtxt( os.path.join(self.m_settings["parameters"]["path"], "times.txt"), unpack=True, skiprows=1 )
            times_header = open( os.path.join(self.m_settings["parameters"]["path"], "times.txt"), "r").readline().split()
            scalar = np.loadtxt( os.path.join(self.m_settings["parameters"]["path"], "scalar.txt"), unpack=True, skiprows=1 )
            scalar_header = open( os.path.join(self.m_settings["parameters"]["path"], "scalar.txt"), "r").readline().split()
        except Exception as e:
            print(f"Error loading scalar data: {e}")
            return None
        
        # Generate a valid figure and ax
        if figure is None and axes is None:
            figure, axes = plt.subplots(figsize=figsize, nrows=1, ncols=2)
        if figure is None and axes is not None:
            figure = axes.figure
        if axes is None or len(axes) != 2:
            axes = figure.subplots(nrows=1, ncols=2)
        
        # Time in ps
        time_index = scalar_header.index("t")
        t = scalar[time_index]
        
        # Plot time statistics
        avg_time = np.mean(times[1])
        standard_deviation = np.std(times[1])
        ck_it = int(len(times[1])/20)
        convolve_kernel = np.ones( ck_it ) / ck_it
        moving_avg = np.convolve(times[1], convolve_kernel, 'valid')
        axes[0].scatter(t, times[1], s=1, label="Iteration Time", alpha = 0.4)
        axes[0].plot(t[ck_it//2:len(moving_avg)+ck_it//2], moving_avg, label="Moving Average")
        axes[0].axhline(avg_time, color="red", linestyle="--", label=f"Average Time: {(avg_time*1e6):.2f} $\\mu$s")
        axes[0].axhline(avg_time + standard_deviation, color="blue", linestyle="--", label=f"Standard Deviation: {(standard_deviation*1e6):.2f} $\\mu$s")
        axes[0].axhline(avg_time - standard_deviation, color="blue", linestyle="--")
        axes[0].set_ylim(avg_time - 3*standard_deviation, avg_time + 3*standard_deviation)

        # Plot scalar statistics
        for i in range(1, len(scalar_header)):
            if scalar_header[i] == "t":
                continue
            axes[1].plot(t, scalar[i], label=scalar_header[i])

        # Set labels
        axes[0].set_xlabel("Time [ps]")
        axes[0].set_ylabel("Walltime [s]")
        axes[0].set_title("Iteration Time")
        axes[0].legend()

        axes[1].set_xlabel("Time [ps]")
        axes[1].set_ylabel("Scalar Value")
        axes[1].set_title("Scalar Values")
        axes[1].legend()

        return figure, axes

    def plot_system(self):
        # find files, plot them all in a neat way
        # update layout depending on what is available
        # return matplotlib figure and axes
        pass

    def animate(self):
        # if animatable = true ...
        # return animation instance where user can customize the matplotlib animation
        pass

    def load_matrix(self, name: str, *, sub_path: str = "") -> np.ndarray:
        raw = np.loadtxt( os.path.join(self.m_settings["parameters"]["path"], sub_path, name), skiprows=1 )
        # load first row to get the size of the matrix
        with open( os.path.join(self.m_settings["parameters"]["path"], sub_path, name), "r") as f:
            header = f.readline().split()
        
        # if the header does not start with "# SIZE", raise an error
        if header[0] != "#" or header[1] != "SIZE":
            raise ValueError("Invalid matrix file. Header must start with '# SIZE'.")

        Nx,Ny,*_ = header[2:]
        Nx,Ny = int(Nx), int(Ny)
        
        if np.prod(raw.shape) == 2 * Nx * Ny:
            matrix = raw[:Ny,:] + 1j*raw[Ny:,:]
        else:
            matrix = raw
        matrix = matrix.reshape(Ny,Nx)

        # if the matrix size does not match the settings, raise an error
        # if this is desired, the user can always catch the exception and ignore it
        if matrix.shape[0] != Ny or matrix.shape[1] != Nx:
            raise ValueError("Matrix dimensions do not match the solver settings. {Nx}x{Ny} expected, {matrix.shape[0]}x{matrix.shape[1]} provided.")
        
        return matrix

    def save_matrix(self, matrix: np.ndarray, name: str, *, sub_path: str = "", decimals = 5):
        # First, check if the matrix is compatible with the current settings.
        if not isinstance(matrix, np.ndarray):
            raise ValueError("Matrix must be a numpy array.")
        if matrix.ndim != 2:
            raise ValueError("Matrix must be a 2D numpy array.")
        if self.m_settings["parameters"].get("N", None) is None:
            raise ValueError("Matrix size not set in the solver settings.")
        Nx,Ny = self.m_settings["parameters"]["N"]
        if matrix.shape[0] != Nx or matrix.shape[1] != Ny:
            raise ValueError("Matrix dimensions do not match the solver settings. {Nx}x{Ny} expected, {matrix.shape[0]}x{matrix.shape[1]} provided.")
        
        x_max, y_max = self.m_settings["parameters"].get("L", (1.0,1.0))
        dx = x_max / Nx
        dy = y_max / Ny
        header = f"SIZE {Nx} {Ny} {x_max} {x_max} {dx} {dx} PYTHON-GENERATED"
        
        if np.isrealobj(matrix):
            np.savetxt( os.path.join(self.m_settings["parameters"]["path"], sub_path, name), matrix, header=header, fmt=f'%.{decimals}f' )
        else:
            np.savetxt( os.path.join(self.m_settings["parameters"]["path"], sub_path, name), np.vstack((matrix.real, matrix.imag)), header=header, fmt=f'%.{decimals}f' )

    # TODO: envelope builder / visualizer
    # TODO: parameter adding functions
    # -> .add_parameter("key", value)
    # -> .setTime(t1,dt)
    # -> .setSpace(L,N)
    # -> .addEnvelope(what,...)
    # --> .addPotential, .addPump, .addPulse etc.

    # Parameter adding functions
    def add_parameter(self, key: str, values, *, append: bool = False):
        if values is None:
            return self.validate_settings()
        if key not in self.m_settings["parameters"]:
            self.m_settings["parameters"][key] = list()
        if not isinstance(values, (list,tuple)):
            values = [values]
        if append:
            self.m_settings["parameters"][key].extend(values)
        else:
            self.m_settings["parameters"][key] = values
        return self.validate_settings()
    
    def add_flag(self, flag: str):
        if flag not in self.m_settings["flags"]:
            self.m_settings["flags"].append(flag)
        return self.validate_settings()

    def remove_parameter(self, key: str):
        if key in self.m_settings["parameters"]:
            del self.m_settings["parameters"][key]
        return self.validate_settings()
    
    def remove_flag(self, flag: str):
        if flag in self.m_settings["flags"]:
            self.m_settings["flags"].remove(flag)
        return self.validate_settings()
    
    def clear_parameters(self):
        self.m_settings["parameters"] = dict()
        return self.validate_settings()
    
    def clear_flags(self):
        self.m_settings["flags"] = list()
        return self.validate_settings()
    
    # Specialized parameter adding functions
    def p_set_time(self, *, t1: float | None = None, dt: float | None = None):
        self.add_parameter("tend", t1)
        self.add_parameter("dt", dt)
        return self.validate_settings()
    
    def p_set_space(self, *, L: float, N: int):
        self.add_parameter("L", L)
        self.add_parameter("N", N)
        return self.validate_settings()
    
    # TODO: assisted evenlope builder using ipywidgets
    def p_add_envelope(self, key: str, amp: float, w_x: float, w_y: float, p_x: float, p_y: float, *, 
                       # Optional Parameters
                       power: float = 1.0, behaviour: str = "add", polarization: str = "both", topological_charge: int = 1, type: str = "gauss+noDivide",
                       # Time Parameters
                       t_type: str = "const", t0: float = 0.0, t_width: float = 1e5, t_frequency: float = 0.0
                       ):
        # Add envelope to the solver settings
        if t_type == "const" or t_type is None:
            envelope = (amp, behaviour, w_x, w_y, p_x, p_y, polarization, power, topological_charge, type)
        else:
            envelope = (amp, behaviour, w_x, w_y, p_x, p_y, polarization, power, topological_charge, type, "time", t_type, t0, t_width, t_frequency)
        
        self.add_parameter(key, envelope, append=True)
        return self.validate_settings()

    # TODO: include type
    def precalculate_envelope(self, amp: float, w_x: float, w_y: float, p_x: float, p_y: float, *, 
                       # Optional Parameters
                       power: float = 1.0, topological_charge: int = 1, type: str = "gauss+noDivide"):
        Lx,Ly = self.m_settings["parameters"].get("L", None)
        Nx,Ny = self.m_settings["parameters"].get("N", None)
        if Lx is None or Ly is None or Nx is None or Ny is None:
            raise ValueError("System size not set in the solver settings. Please set the system size before precalculating the envelope.")
        
        X,Y = np.meshgrid(np.linspace(-Lx/2, Lx/2, Nx), np.linspace(-Ly/2, Ly/2, Ny))
        envelope = amp * np.exp(-power * ((X-p_x)**2 / w_x**2 + (Y-p_y)**2 / w_y**2))
        envelope = envelope * np.exp(1j * topological_charge * np.angle(envelope))

        return envelope


    def p_add_envelope_assist(self):
        if not self.is_ipython():
            raise Exception("This function requires an IPython environment.")
        import ipywidgets as widgets
        from IPython.display import display, clear_output
        
        key_selector = widgets.Dropdown( options=("potential", "pump", "pulse", "initialState", "initialReservoir", "fftMask"), description="What to add:" ) 
        amp_numinput = widgets.FloatText(value=1.0, description="Amplitude:")
        wx_numinput = widgets.FloatText(value=1.0, description="Width X:")
        wy_numinput = widgets.FloatText(value=1.0, description="Width Y:")
        px_numinput = widgets.FloatText(value=0.0, description="Position X:")
        py_numinput = widgets.FloatText(value=0.0, description="Position Y:")
        
        confirm_button = widgets.Button(description="Confirm")
        output = widgets.Output()
        controls = widgets.VBox([key_selector, amp_numinput, wx_numinput, wy_numinput, px_numinput, py_numinput, confirm_button])

        plot_output = widgets.Output()

        def update_plot():
            with plot_output:
                clear_output(wait=True)
                # Get current values from the widgets
                key = key_selector.value
                amp = amp_numinput.value
                w_x = wx_numinput.value
                w_y = wy_numinput.value
                p_x = px_numinput.value
                p_y = py_numinput.value

                # Create the envelope
                envelope = self.precalculate_envelope(amp, w_x, w_y, p_x, p_y)

                # Plot the envelope
                fig, ax = self.plot_matrix(matrix=envelope.real, title=f"{key.capitalize()} Envelope")
                plt.show()

        def on_change(change):
            update_plot()

        def on_confirm_click(button):
            with output:
                clear_output(wait=True)
                # Get current values from the widgets
                key = key_selector.value
                amp = amp_numinput.value
                w_x = wx_numinput.value
                w_y = wy_numinput.value
                p_x = px_numinput.value
                p_y = py_numinput.value

                # Add the envelope to the settings
                self.p_add_envelope(key, amp, w_x, w_y, p_x, p_y)
                print(f"Envelope added to {key.capitalize()}.")

        key_selector.observe(on_change, names="value")
        amp_numinput.observe(on_change, names="value")
        wx_numinput.observe(on_change, names="value")
        wy_numinput.observe(on_change, names="value")
        px_numinput.observe(on_change, names="value")
        py_numinput.observe(on_change, names="value")
        confirm_button.on_click(on_confirm_click)

        # Display Widget
        display( widgets.HBox([controls, plot_output]), output )


    # TODO: parameter sweeper / scanner