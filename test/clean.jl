"""
    clean([distclean=false])

Clean up build/doc/testing artifacts. Restore to clean checkout state
(distclean)
"""
function clean(; distclean=false, _exit=true)

    _glob(folder, ending) =
        [name for name in readdir(folder; join=true) if (name |> endswith(ending))]
    _exists(name) = isfile(name) || isdir(name)
    _push!(lst, name) = _exists(name) && push!(lst, name)

    ROOT = dirname(@__DIR__)

    ###########################################################################
    CLEAN = String[]
    for folder in [ROOT, joinpath(ROOT, "src"), joinpath(ROOT, "test")]
        append!(CLEAN, _glob(folder, ".cov"))
    end
    _push!(CLEAN, joinpath(ROOT, "coverage"))
    _push!(CLEAN, joinpath(ROOT, "docs", "build"))
    _push!(CLEAN, joinpath(ROOT, "lcov.info"))
    ###########################################################################

    ###########################################################################
    DISTCLEAN = String[]
    for folder in [ROOT, joinpath(ROOT, "docs"), joinpath(ROOT, "test")]
        push!(DISTCLEAN, joinpath(folder, "Manifest.toml"))
    end
    _push!(DISTCLEAN, joinpath(ROOT, ".JuliaFormatter.toml"))
    ###########################################################################

    for name in CLEAN
        @info "rm $name"
        rm(name, force=true, recursive=true)
    end
    if distclean
        for name in DISTCLEAN
            @info "rm $name"
            rm(name, force=true, recursive=true)
        end
        if _exit
            @info "Exiting"
            exit(0)
        end
    end

end

distclean() = clean(distclean=true)
