
import os
import subprocess

package_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../"))

# Phoenix will try to use these binaries first, then fall back to the fallback binaries
phoenix_binary_name = "NATIVE_PHOENIX"

def build(repo_path: str, repo_url: str, branch: str, targets: list[str], threads: int = 4):
    # Clone the repository
    repo_path = os.path.abspath(os.path.join(package_path, repo_path))
    print(f"Cloning repository {repo_url} (branch: {branch}) into {repo_path}...")
    if os.path.exists(repo_path):
        print(f"-> Removing existing repository at {repo_path}")
        subprocess.check_call(['rm', '-rf', repo_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.check_call(['git', 'clone', '--branch', branch, repo_url, repo_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Change to the cloned directory and install the package
    print("Installing PHOENIX...")
    subprocess.check_call(['make', 'clean', '-C', repo_path])

    if 'cpu' in targets:
        print("Building PHOENIX for CPU...")
        try:
            subprocess.check_call(['make', '-C', repo_path, f'-j{threads}', f'COMPILER={args.compiler_cpu}', 'OPTIMIZATION=-O3', 'CPU=TRUE', 'FP32=TRUE', 'SFML=FALSE', f'TARGET={f"{phoenix_binary_name}_CPU_FP32.{extension}"}'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Check if the compilations were successful, meaning that the binaries were created
        except subprocess.CalledProcessError as e:
            print(f"Error during compilation of PHOENIX for CPU (FP32): {e}")
        
        try:
            subprocess.check_call(['make', '-C', repo_path, f'-j{threads}', f'COMPILER={args.compiler_cpu}', 'OPTIMIZATION=-O3', 'CPU=TRUE', 'FP32=FALSE', 'SFML=FALSE', f'TARGET={f"{phoenix_binary_name}_CPU_FP64.{extension}"}'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"Error during compilation of PHOENIX for CPU (FP64): {e}")

    if 'gpu' in targets:
        print("Building PHOENIX for GPU...")
        try:
            subprocess.check_call(['make', '-C', repo_path, f'-j{threads}', f'COMPILER={args.compiler_gpu}', 'OPTIMIZATION=-O0', 'CPU=FALSE', 'FP32=TRUE', 'SFML=FALSE', f'TARGET={f"{phoenix_binary_name}_GPU_FP32.{extension}"}'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Check if the compilations were successful, meaning that the binaries were created
        except subprocess.CalledProcessError as e:
            print(f"Error during compilation of PHOENIX for GPU (FP32): {e}")

        try:
            subprocess.check_call(['make', '-C', repo_path, f'-j{threads}', f'COMPILER={args.compiler_gpu}', 'OPTIMIZATION=-O0', 'CPU=FALSE', 'FP32=FALSE', 'SFML=FALSE', f'TARGET={f"{phoenix_binary_name}_GPU_FP64.{extension}"}'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"Error during compilation of PHOENIX for GPU (FP32): {e}")

def update(exec_paths: list[str], extension: str = '.exe'):
    print("Look for binaries. Target directories: ", exec_paths)
    executables = []
    for current_path in exec_paths:
        files = os.listdir(os.path.abspath(current_path))
        for file in files:
            if not file.endswith(extension):
                continue
            if os.path.isfile(os.path.join(current_path, file)):
                try:
                    subprocess.check_call([os.path.join(current_path, file), '-h'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print(f"Found working fallback binary: {file}")
                    executables.append( os.path.join(current_path, file) )
                except subprocess.CalledProcessError as e:
                    print(f"Binary {file} is not working.")
    return executables

if __name__ == '__main__':
    import subprocess
    import argparse

    """Custom command to clone the repository and prepare the package."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--repo-url', type=str, default='https://github.com/Schumacher-Group-UPB/PHOENIX', help='URL of the Git repository')
    parser.add_argument('--repo-path', type=str, default='phoenix_repository/', help='Target directory for the repository clone, relative to the package directory')
    parser.add_argument('--branch', type=str, default='master', help='Branch to clone')
    parser.add_argument('--target', type=str, default='cpu,gpu', help='Target Device to build PHOENIX for gpu')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads to use for the compilation')

    args = parser.parse_args()

    # if windows, use .exe extension, otherwise use .out
    extension = 'exe' if os.name == 'nt' else 'out'
    
    build(args.repo_path, args.repo_url, args.branch, args.target.split(','), args.threads)

    update([os.path.join(package_path, folder) for folder in (args.repo_path, "phoenix_fallback")], extension)