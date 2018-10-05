using Literate 

courses = [
    (name = "Course Template", dir = "Course_Template")
]

for c in courses 
    files = readdir(c.dir)
    for f in files 
        path = joinpath(c.dir, f)
        include(path)
        Literate.notebook(path, "build/"; credit=false)
    end
end

include("Course_Template/testfile.jl")