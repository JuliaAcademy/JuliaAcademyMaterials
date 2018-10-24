[![Build Status](https://dev.azure.com/JuliaComputing/Julia%20Academy/_apis/build/status/JuliaComputing.JuliaAcademyMaterials)](https://dev.azure.com/JuliaComputing/Julia%20Academy/_build/latest?definitionId=1)

# Source files for Julia Academy Notebooks

- Running `julia --project build.jl` will
    1. Create a `Notebooks/` directory
    2. Course content in `Courses/` will get converted to notebook (if `.jl` file) or get directly copied.
    3. A specific course can be built by giving it as an argument, e.g. `julia build.jl SomeCourse`.
