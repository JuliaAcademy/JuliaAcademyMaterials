import Pkg

srcpath = joinpath(@__DIR__(), "Courses")
buildpath = joinpath(@__DIR__(), "Notebooks")

if isempty(ARGS)
    courses = filter(x->isdir(joinpath(srcpath), x) && x !== "Template", readdir(srcpath))
else
    courses = ARGS
end

rm(buildpath, recursive=true, force=true)
mkdir(buildpath)

notebooks_failed = String[]

Base.julia_cmd()
for dir in courses
    @info "building $dir..."
    src_course = joinpath(srcpath, dir)
    build_course = mkdir(joinpath(buildpath, dir))
    cp(src_course, build_course; force=true)

    Pkg.activate(build_course)
    Pkg.add(Pkg.PackageSpec(url="https://github.com/JuliaAcademy/JuliaAcademyData.jl.git"))
    Pkg.resolve()
    test = false
    Pkg.instantiate()
    for file in readdir(build_course)
        if endswith(file, ".jl")
            course = joinpath(build_course, file)
            academy_environment = @__DIR__
            # Create a stacked environment with the JuliaAcademy environment in
            # it so we can load Literate for each notebook
            script = """
            pushfirst!(LOAD_PATH, $(repr(academy_environment)));
            import Literate;
            Literate.notebook($(repr(course)), $(repr(build_course)); credit=false, execute=false)
            """
            try
                run(`$(Base.julia_cmd()) -e $script`)
            catch
                err_src_course = relpath(joinpath(src_course, file), @__DIR__)
                @error "Script failed to build: $err_src_course)"
                push!(notebooks_failed, err_src_course)
            end
            rm(course)
        end
    end
end

if !isempty(notebooks_failed)
    error("The following notebooks failed: \n",
          join(notebooks_failed, '\n'))
end
