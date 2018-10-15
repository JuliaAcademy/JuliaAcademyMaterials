using Literate 

# Add courses here
courses = [
    (name = "Course Template",                  dir = "Template"),
    (name = "Big Data Analysis with JuliaDB",   dir = "JuliaDB")
]

#-----------------------------------------------------------------------#
for c in courses 
    files = filter(x -> endswith(x, ".jl"), readdir(c.dir))
    for f in files 
        path = joinpath(c.dir, f)
        try
            include(path)  # "test" that the code runs
            Literate.notebook(path, "_generated_notebooks/$(c.name)/"; credit=false)
        catch 
            @warn "File $path contains an error"
        end
    end
end