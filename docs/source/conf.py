# Configuration file for the Sphinx documentation builder.

import os
import sys
import shutil
import pathlib

# -- Path setup --------------------------------------------------------------

# Add the project root directory to sys.path, so Sphinx can find your modules.
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

# -- Project information -----------------------------------------------------

project = 'AIMotionLab-Virtual'
copyright = '2024, Botond Gaal'
author = 'Botond Gaal'
release = '1.0.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',   # Automatically generate documentation from docstrings
    'sphinx.ext.napoleon',  # Support for NumPy and Google style docstrings
    'sphinx.ext.viewcode',  # Add links to highlighted source code
    'sphinx.ext.todo',      # Support for TODOs
    'sphinx.ext.autosummary',
]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'private-members': False,
    'no-value': True,
}

autodoc_member_order = 'groupwise'
todo_include_todos = True
templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

pygments_style = "sphinx"
pygments_dark_style = "monokai"
highlight_language = "python"
html_theme = 'furo'
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

# -- Custom RST Generation ---------------------------------------------------

def generate_package_rst(package_path, output_path, package_prefix=''):
    """
    Recursively generate .rst files for a package, mimicking the package structure.

    :param package_path: Path to the package directory.
    :param output_path: Path where the .rst files will be generated.
    :param package_prefix: The importable package prefix (e.g., 'aiml_virtual.').
    """
    package_name = package_path.name
    package_import_path = f"{package_prefix}{package_name}"
    rst_output_dir = output_path if output_path.parts[-1] == package_name else output_path / package_name

    # Create the output directory if it doesn't exist
    rst_output_dir.mkdir(parents=True, exist_ok=True)

    subpackages = []
    modules = []

    # Iterate over items in the package directory
    for item in package_path.iterdir():  # iterdir() returns all paths, both .py and directories
        if item.is_dir() and (item / '__init__.py').exists():  # if this is a package
            subpackages.append(item.name)
        elif item.suffix == '.py' and item.name != '__init__.py':  # if this is a file (that's not __init__.py)
            modules.append(item.stem)

    # Generate the .rst file for the package (as opposed to a .rst file for a module, since .rst files corresponding
    # to a package contain a toctree, whereas .rst files corresponding to a module contain an automodule directive)
    rst_file_path = rst_output_dir / f"{package_name}_pkg.rst"  # rst file for the package will have _pkg suffix
    with rst_file_path.open('w', encoding='utf-8') as rst_file:
        # Write the package title
        rst_file.write(f"{package_name} package\n")
        rst_file.write(f"{'=' * (len(package_name) + 8)}\n\n")

        # Write Subpackages section if there are subpackages
        if subpackages:
            rst_file.write("Subpackages\n")
            rst_file.write("-----------\n\n")
            rst_file.write(".. toctree::\n")
            rst_file.write("   :maxdepth: 1\n\n")
            for subpkg in sorted(subpackages):
                rst_file.write(f"   {subpkg}/{subpkg}_pkg\n")  # rst files for packages will have _pkg suffix
            rst_file.write("\n")

        # Write Modules section if there are modules
        if modules:
            rst_file.write("Modules\n")
            rst_file.write("-------\n\n")
            rst_file.write(".. toctree::\n")
            rst_file.write("   :maxdepth: 1\n\n")
            for module in sorted(modules):
                rst_file.write(f"   {module}\n")
            rst_file.write("\n")

    # Generate rst files for modules
    for module in modules:
        module_rst_path = rst_output_dir / f"{module}.rst"
        with module_rst_path.open('w', encoding='utf-8') as module_rst_file:
            module_rst_file.write(f"{module}.py module\n")
            module_rst_file.write(f"{'=' * (len(module) + 10)}\n\n")
            module_rst_file.write(f".. automodule:: {package_import_path}.{module}\n")
            module_rst_file.write("\n")

    # Recursively generate .rst files for subpackages
    for subpkg in subpackages:
        subpkg_path = package_path / subpkg
        generate_package_rst(subpkg_path, rst_output_dir, package_prefix=f"{package_import_path}.")

def generate_rst_files():
    docs_dir = pathlib.Path(__file__).parent.resolve()  # normally docs/source
    project_root = docs_dir.parent.parent.resolve()  # AIMotionLab-Virtual
    module_name = 'aiml_virtual'
    module_dir = project_root / module_name  # where the module to be documented lies
    output_dir = docs_dir / module_name  # where to generate rst files

    # Remove the output directory if it exists
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Generate RST files
    generate_package_rst(module_dir, output_dir)

def setup(app):
    app.connect('builder-inited', lambda _: generate_rst_files())

if __name__ == "__main__":
    generate_rst_files()
