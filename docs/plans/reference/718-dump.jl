# Dump an asymmetric CTMRG fixed point + all §V.3 intermediates, so the
# Python port can be checked against a root the reference itself validates.

using Random
using LinearAlgebra
using PEPSKit
using TensorKit
using ImplicitDifferentiationPEPS

using MPSKit
using PEPSKit: _prev_coordinate, _next_coordinate, EnlargedCorner,
    half_infinite_environment, eachcoordinate, ctmrg_iteration,
    compute_gauge_fix_gauge, ScramblingEnvGauge, fix_relative_phases, sdiag_pow
using ImplicitDifferentiationPEPS: remove_inverse_roots,
    check_asymmetric_environment, generate_asymmetric_characteristic_equation

Random.seed!(42039482048)

D = 2
χ = 4
d = 2

Pspace = ComplexSpace(d)
Vspace = ComplexSpace(D)
Espace = ComplexSpace(χ)

state = InfinitePEPS(randn, ComplexF64, Pspace, Vspace, Vspace)
alg = SimultaneousCTMRG(; tol = 1.0e-12, verbosity = 2, maxiter = 500)
env0 = CTMRGEnv(randn, ComplexF64, state, Espace)
env, info = leading_boundary(env0, state, alg)

println("=== converged ===")

# SVD tensors at the fixed point: one extra iteration, then gauge-fix,
# exactly as the reference's own _rrule does.
env_conv, svdinfo = ctmrg_iteration(InfiniteSquareNetwork(state), env, alg)
signs, = compute_gauge_fix_gauge(env_conv, env, ScramblingEnvGauge())
S = normalize.(svdinfo.S)
U, V = fix_relative_phases(svdinfo.U, svdinfo.V, signs)

sz(t) = size(convert(Array, t))
println("shapes: C1 ", sz(env.corners[1,1,1]), " T1 ", sz(env.edges[1,1,1]))
println("        U ", sz(U[1,1,1]), " S ", sz(S[1,1,1]), " V ", sz(V[1,1,1]))
println("        peps ", sz(state[1,1]))

check_asymmetric_environment(state, env, U, S, V; tol = 1.0e-8)

# modified corners / edges (Eq. 82)
Sfull = TensorMap.(S)
C̃, Ẽ = remove_inverse_roots(env.corners, env.edges, S)

# evaluate the reference characteristic equations at the root
UL = left_null.(U)
VR = right_null.(V)
u = map(zip(U, UL)) do (Uc, ULc)
    zeros(scalartype(Uc), MPSKit._lastspace(ULc)' ← MPSKit._lastspace(Uc)')
end
v = map(zip(V, VR)) do (Vc, VRc)
    zeros(scalartype(Vc), MPSKit._firstspace(Vc) ← MPSKit._firstspace(VRc))
end
is = sdiag_pow.(S, -1)
F = generate_asymmetric_characteristic_equation(is, U, V, UL, VR, Val(:implicit))
FS = F(state, C̃, Ẽ, u, Sfull, v)
println("reference |F| at root: ", norm.(FS))

# ---- dump raw arrays ----
function dumparr(io, name, t)
    a = convert(Array, t)
    println(io, "@ ", name, " ", join(size(a), ","))
    for x in vec(a)              # column-major order
        println(io, real(x), " ", imag(x))
    end
end

open("dump.txt", "w") do io
    dumparr(io, "peps", state[1, 1])
    for k in 1:4
        dumparr(io, "C$k", env.corners[k, 1, 1])
        dumparr(io, "T$k", env.edges[k, 1, 1])
        dumparr(io, "Ct$k", C̃[k, 1, 1])
        dumparr(io, "Et$k", Ẽ[k, 1, 1])
        dumparr(io, "S$k", S[k, 1, 1])
        dumparr(io, "U$k", U[k, 1, 1])
        dumparr(io, "V$k", V[k, 1, 1])
    end
end
println("wrote dump.txt")

# ---- recompute the §V.3 intermediates explicitly, mirroring the closure in
# generate_asymmetric_characteristic_equation, so they can be dumped ----
using ImplicitDifferentiationPEPS: absorb_left, absorb_right, _proj_sinv_indices,
    _leftvec_invfroot_indices, _rightvec_invfroot_indices, fourthroot,
    _rotate_north_localsandwich, _contract_PR_PL, _contract_PR_M
nr, nc = 1, 1
coords = eachcoordinate(Sfull)
isv  = map(inv, Sfull)
isqsR = map(fourthroot, adjoint.(isv) .* isv)
isqsL = map(fourthroot, isv .* adjoint.(isv))
Uv = map(coords) do co; U[co...] + UL[co...] * u[co...] end
Vv = map(coords) do co; V[co...] + v[co...] * VR[co...] end
Ud2 = map(coords) do co
    absorb_right(Uv[co...]', isqsR[_leftvec_invfroot_indices(co, nr, nc)...])
end
Vd2 = map(coords) do co
    absorb_left(Vv[co...]', isqsL[_rightvec_invfroot_indices(co, nr, nc)...])
end
iCi2 = map(coords) do co
    isv[_prev_coordinate(co, nr, nc)...] * C̃[co...] * isv[co...]
end
EC2 = map(coords) do co
    TensorMap(EnlargedCorner(InfiniteSquareNetwork(state), CTMRGEnv(iCi2, Ẽ), co))
end
PR2 = map(coords) do co
    absorb_right(Ud2[co...] * EC2[co...], isv[_proj_sinv_indices(co, nr, nc)...])
end
PLpart2 = map(coords) do co
    EC2[_next_coordinate(co, nr, nc)...] * Vd2[co...]
end
PL2 = map(coords) do co
    absorb_left(PLpart2[co...], isv[_proj_sinv_indices(co, nr, nc)...])
end
println("intermediate shapes: iCi ", sz(iCi2[1,1,1]), " EC ", sz(EC2[1,1,1]),
        " Ud ", sz(Ud2[1,1,1]), " Vd ", sz(Vd2[1,1,1]),
        " PR ", sz(PR2[1,1,1]), " PL ", sz(PL2[1,1,1]))

open("dump2.txt", "w") do io
    for k in 1:4
        dumparr(io, "is$k",  isv[k,1,1])
        dumparr(io, "rootL$k", isqsR[k,1,1])
        dumparr(io, "rootR$k", isqsL[k,1,1])
        dumparr(io, "iCi$k", iCi2[k,1,1])
        dumparr(io, "EC$k",  EC2[k,1,1])
        dumparr(io, "Ud$k",  Ud2[k,1,1])
        dumparr(io, "Vd$k",  Vd2[k,1,1])
        dumparr(io, "PR$k",  PR2[k,1,1])
        dumparr(io, "PL$k",  PL2[k,1,1])
        dumparr(io, "UL$k",  UL[k,1,1])
        dumparr(io, "VR$k",  VR[k,1,1])
    end
end
println("wrote dump2.txt")
