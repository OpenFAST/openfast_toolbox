import os
import ctypes
import platform

from openfast_toolbox.tools.strings import INFO, FAIL, OK, WARN, print_bold

def check_discon_library(libpath):
    """
    Try to load a DISCON-style library and optionally call its DISCON entry point.
    
    Parameters
    ----------
    libpath : str
        Path to the shared library (.dll, .so, .dylib)

    Returns
    -------
    success : bool
        True if the library is suitable for the current OS and the DISCON function is callable.
    msg : str
        Description of the result.
    """
    # Normalize path
    libpath = os.path.abspath(libpath)

    # Check extension against OS
    system = platform.system()
    expected_ext = { "Windows": ".dll", "Linux": ".so", "Darwin": ".dylib" }
    if system not in expected_ext:
        WARN(f"Unsupported OS: {system}")
    if not libpath.endswith(expected_ext[system]):
        WARN(f"Library extension mismatch: expected {expected_ext[system]} for {system}")

    if not os.path.isfile(libpath):
        FAIL(f"File not found: {libpath}")
        return False

    try:
        lib = ctypes.CDLL(libpath)
        OK(f"Successully loaded library {libpath}")
    except OSError as e:
        FAIL(f"Failed to load library {libpath}\nError: {e}")
        return False

    # Check if DISCON function exists
    try:
        discon_func = lib.DISCON
        # Optionally set argument/return types (DISCON has a big Fortran-style interface)
        # but for a simple check, just ensure we can obtain the symbol
    except AttributeError:
        FAIL(f"Library loaded but no DISCON symbol found in {libpath}")
        return False
    # Try a "dry call" — this will fail unless proper arguments are passed.
    # We just test that the function pointer exists and is callable.
    try:
        _ = callable(discon_func)
        OK(f'DISCON function is present in library.')
    except Exception as e:
        FAIL[f"DISCON symbol found but not callable:\nLibrary:{libpath}\nError: {e}"]
        return False

    #OK(f'Successfully loaded {libpath} and DISCON function is present.')
    return True
