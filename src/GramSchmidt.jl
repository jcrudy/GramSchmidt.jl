module GramSchmidt
using Base.BLAS, Base.LinAlg, Debug
import Base.BLAS: gemv!
export gramschmidt, gramschmidt!

function gramschmidt(Q, v, eta)
    # Apply Gram-Schmidt with reorthogonalization
    r = 0.0
    prevnorm = inf(Float64)
    newnorm = norm(v)
    while newnorm <= eta * prevnorm
        s = transpose(Q) * v
        r = r + s
        u = Q * s
        v = v - u
        prevnorm = newnorm
        newnorm = norm(v)
    end
    rho = newnorm
    q = v ./ rho
    q, r, rho
end

function gramschmidt!{T<:FloatingPoint}(Q::Array{T,2}, v::Array{T,1}, eta::T, r::Array{T,1})
    m, n = size(Q)
    u::Array{T, 1} = zeros(T, m)
    fill!(r, zero(T))
    # r::Array{T, 1} = zeros(T, n)
    s::Array{T, 1} = zeros(T, n)
    gramschmidt!(Q, v, eta, r, u, s)
end

function gramschmidt!{T<:FloatingPoint}(Q::Array{T,2}, v::Array{T,1}, eta::T,
    r::Array{T,1}, u::Array{T,1}, s::Array{T,1})
    # Apply Gram-Schmidt with reorthogonalization in place using BLAS
    m, n = size(Q)
    prevnorm::T = inf(T)
    newnorm::T = nrm2(m, v, stride(v, 1))
    while newnorm <= eta * prevnorm
        # s = transpose(Q) * v
        gemv!('T', one(T), Q, v, zero(T), s)
        # r = r + s
        axpy!(n, one(T), s, stride(s, 1), r, stride(r, 1))
        # u = Q * s
        gemv!('N', one(T), Q, s, zero(T), u)
        # v = v - u
        axpy!(m, -one(T), u, stride(u, 1), v, stride(v, 1))
        prevnorm = newnorm
        # newnorm = norm(v)
        newnorm = nrm2(m, v, stride(v, 1))
    end
    scal!(m, 1.0 / newnorm, v, stride(v,1))
    newnorm
end

end # module
