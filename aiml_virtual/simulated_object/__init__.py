from aiml_virtual.utils import utils_general
# this makes sure that all the necessary subclasses get initialized with __init_subclass__, even if they aren't
# explicitly imported
utils_general.import_submodules(__name__)