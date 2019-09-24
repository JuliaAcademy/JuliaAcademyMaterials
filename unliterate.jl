import JSON, Glob

function unliterate(path, output)
    if isdir(path)
        !isdir(output) && mkdir(output)
        for file in Glob.glob(joinpath(path, "*.ipynb"))
            unliterate(file, joinpath(output, splitext(splitdir(file)[2])[1] * ".jl"))
        end
        return
    end
    
    endswith(path, ".ipynb") || throw(ArgumentError("only Jupyter notebooks are supported"))
    nb = open(JSON.parse, path, "r")
    out = open(output, "w")
    prev_type = ""
    for cell in nb["cells"]
        type = cell["cell_type"]
        type == prev_type && print(out, "#-\n\n")
        prev_type = type
        
        if type == "markdown"
            for line in cell["source"]
                print(out, "#")
                if !isempty(strip(line))
                    print(out, " ")
                    print(out, line)
                else
                    print(out, "\n")
                end
            end
            print(out, "\n\n")
        elseif type == "code"
            for line in cell["source"]
                startswith(line, "# ") && print(out, "#")
                print(out, line)
            end
            print(out, "\n\n")
        else
            error("unknown cell type $type")
        end
    end
    close(out)
    return
end

function main()
    if length(ARGS) != 2
        println("USAGE: unliterate source.ipynb output.jl")
        exit(1)
    end
    unliterate(ARGS[1], ARGS[2])
    return
end

main()
