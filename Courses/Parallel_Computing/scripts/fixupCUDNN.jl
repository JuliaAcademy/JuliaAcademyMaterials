let
import CuArrays
hasmethod(args...; kwargs...) = Base.hasmethod(args...; kwargs...)
if VERSION < v"1.2"
	# Backport keyword checking support from https://github.com/JuliaLang/julia/pull/30712
   function hasmethod(@nospecialize(f), @nospecialize(t), kwnames::Tuple{Vararg{Symbol}}; world=typemax(UInt))
      Base.hasmethod(f, t, world=world) || return false
      isempty(kwnames) && return true
      m = which(f, t)
      Base.max_world(m) <= world || return false
      kws = Base.kwarg_decl(m, Core.kwftype(typeof(f)))
      for kw in kws
          endswith(String(kw), "...") && return true
      end
      issubset(kwnames, kws)
   end
end

# Backport https://github.com/JuliaGPU/CuArrays.jl/pull/181 if needed
if !hasmethod(CuArrays.CUDNN.conv!, Tuple{CuArrays.CuArray{Float32}, CuArrays.CuArray{Float32}, CuArrays.CuArray{Float32}}, (:flipkernel,))
@eval CuArrays.CUDNN begin
 function conv!(y::CuArray{T}, x::CuArray{T}, w::CuArray{T};
               pad=0, stride=1, flipkernel=0, alpha=1, dilation=1,
               workspace::Union{CuVector, Nothing}=nothing, algo=0) where T<:CUDNNFloat
  if CUDNN_VERSION < 6000
    all(x -> x == 1, dilation) || error("Only dilation = 1 is supported in cuDNN version < 6")
  end
  if workspace === nothing
    workspace_size =
      cudnnGetConvolutionForwardWorkspaceSize(y, x, w, padding=pad, stride=stride, dilation=dilation,
                                              algo=algo, mode=flipkernel)
    workspace = workspace_size != 0 ? conv_workspace(workspace_size) : workspace
  else
    workspace_size = length(workspace[])
  end
  cudnnConvolutionForward(y, x, w, padding=pad, stride=stride, dilation=dilation, mode=flipkernel,
			  alpha=alpha, algo=algo, workspace=workspace, workspace_size=workspace_size)
end

function ∇conv_filter!(dw::CuArray{T}, dy::CuArray{T}, x::CuArray{T}, w::CuArray{T};
                       pad=0, stride=1, flipkernel=0, alpha=1, dilation=1,
                       workspace::Union{CuVector, Nothing}=nothing, algo=0) where T<:CUDNNFloat
  if CUDNN_VERSION < 6000
    all(x -> x == 1, dilation) || error("Only dilation = 1 is supported in cuDNN version < 6")
  end
  if workspace === nothing
    workspace_size =
      cudnnGetConvolutionBackwardFilterWorkspaceSize(dw, x, w, dy, padding=pad, stride=stride,
					             dilation=dilation, algo=algo, mode=flipkernel)
    workspace = workspace_size != 0 ? conv_workspace(workspace_size) : workspace
  else
    workspace_size = length(workspace[])
  end
  cudnnConvolutionBackwardFilter(dw, x, w, dy, padding=pad, stride=stride, dilation=dilation,
				 mode=flipkernel, alpha=alpha, algo=algo, workspace=workspace,
                                 workspace_size=workspace_size)
end

function ∇conv_data!(dx::CuArray{T}, dy::CuArray{T}, x::CuArray{T}, w::CuArray{T};
                     pad=0, stride=1, flipkernel=0, alpha=1, dilation=1,
                     workspace::Union{CuVector, Nothing}=nothing, algo=0) where T<:CUDNNFloat
  if CUDNN_VERSION < 6000
    all(x -> x == 1, dilation) || error("Only dilation = 1 is supported in cuDNN version < 6")
  end
  if workspace === nothing
    workspace_size =
      cudnnGetConvolutionBackwardDataWorkspaceSize(dx, x, w, dy, padding=pad, stride=stride,
                                                   dilation=dilation, algo=algo, mode=flipkernel)
    workspace = workspace_size != 0 ? conv_workspace(workspace_size) : workspace
  else
    workspace_size = length(workspace[])
  end
  cudnnConvolutionBackwardData(dx, x, w, dy, padding=pad, stride=stride, dilation=dilation,
			       mode=flipkernel, alpha=alpha, algo=algo, workspace=workspace,
                               workspace_size=workspace_size)
end
end
end
end
