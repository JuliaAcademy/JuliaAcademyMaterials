using Literate 

srcpath = joinpath(@__DIR__(), "Courses")
buildpath = joinpath(@__DIR__(), "Notebooks")

rm(buildpath, recursive=true, force=true)
mkdir(buildpath)

for dir in readdir(srcpath)
    subdir = mkdir(joinpath(buildpath, dir))
    for file in readdir(joinpath(srcpath, dir))
        if endswith(file, ".jl")
            try
                Literate.notebook(joinpath(srcpath, dir, file), joinpath(buildpath, subdir); credit=false)
            catch
                @info "Notebook failed to build:" joinpath(path, dir, file)
            end
        else
            cp(joinpath(srcpath, dir, file), joinpath(buildpath, subdir, file))
        end
    end
end