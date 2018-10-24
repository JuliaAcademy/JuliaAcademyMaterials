if isempty(ARGS)
    courses = readdir(srcpath)
else
    courses = ARGS
end

srcpath = joinpath(@__DIR__(), "Courses")
buildpath = joinpath(@__DIR__(), "Notebooks")

rm(buildpath, recursive=true, force=true)
mkdir(buildpath)

notebooks_failed = String[]

Base.julia_cmd()
for dir in courses
    src_course = joinpath(srcpath, dir)
    build_course = mkdir(joinpath(buildpath, dir))
    cp(src_course, build_course; force=true)

    for file in readdir(build_course)
        if endswith(file, ".jl")
            course = joinpath(build_course, file)
            academy_environment = @__DIR__
            # Create a stacked environment with the JuliaAcademy environment in
            # it so we can load Literate for each notebook
            script = """
            pushfirst!(LOAD_PATH, $(repr(academy_environment)));
            import Literate;
            Literate.notebook($(repr(course)), $(repr(build_course)); credit=false)
            """
            try
                run(`$(Base.julia_cmd()) --project=$(build_course) -e $script`)
            catch
                src_course = relpath(joinpath(src_course, file), @__DIR__)
                @error "Script failed to build: $relpath)"
                push!(notebooks_failed, src_course)
            end
            rm(course)
        end
    end
end

if !isempty(notebooks_failed)
    error("The following notebooks failed: \n",
          join(notebooks_failed, '\n'))
end
