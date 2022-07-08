# Modular-Programming with Python

## Key Concepts

* Modular programming provides the ability to group codes into files and subfolders for organizational purpose.
* Technically, module is a Python source file that contains Python code including functions and classes.
  The name of the module is simply the name of the souce file without the .py suffix.
* Conceptually, modules are namespaced so identifiers in one module will not conflict with a function or variable sharing the same name in another file or module.
* Package is a folder that contains a collection of modules identified by a package initialization file
  __init__.py
  A Package can contain other packages (nested packages).
* Examples:
  The Python Standard Library at https://docs.python.org/3/library/
  The Machine learning Library for Python at https://github.com/scikit-learn/scikit-learn
* Project contains modules and packages plus a main program.
* Scoping
  - the global namespace holds all the names defined at the gloable level: print(globals())
    we can access it by name from anywhere in our program
  - the local namespace

## Creating Modules and Packages

* Step 1 Design the Logical Structure of your Project
  Example:
  Main Program: src/main.py
  - data package: src/data
  - models package: src/models
  - utilities package: src/utilities

* Step 2 Design the Logical Structure of each package
  Example: the data package
  - data_ingestion: src/data/data_ingestion
  - data_validation: src/data/data_validation
  - data_transformation: src/data/data_transformation

* Step 3 Implement Modules

* Step 4 Write the main programm main.py that is going to start up and run the project.
         import statement

         def main():
             ......

         if __main__ == "__main__":
            main()

## Using Modules and Packages

* Two versions of the import statement
  - import <module_name> in the current directory
  - from <package_name> import <module_name>

* Two special cases of the import statement
  - a wildcard import: import everything that doesn't start with an underscore character (pravite)
    from  <module_name> or <package_name> import *
  - relative import; import relative to the current module's position
    from . import <module_name>

* Controlling what gets imported
  - By default, import everything from the given module or package except the wildcard import
  - Use the special variable named __all__ to control what gets imported
    Examples:
    in a module: __all__ = ["variable_1", "func_1", "class_1"]
    in a __init__.py: __all__ = ["variable_1", "module_1", "sub_package_1"]

* How does the import statement actually do?
  - we are adding entries to the gloabl namespace
  - implicit: execute any top-level code within that module
  - explicit:
    by defining a top-level function called init() for initializing a module
    Example:
    def init():
        global var_name
        var_name = 0;
    by placing the code inside the __init__.py file
    Example:
    import <module_name>
    ....


* Running modules from the command line (excutable modules)
  - Use the trick at the bottom of the source file
    if __name__ == "__main__":
       statements
  - python <module_name.py> or python -m <module_name.py> if the module uses relative imports


