module TestGramSchmidt

using FactCheck
using Distributions
using GramSchmidt
m = 10
n = 5
Q, R = qr(rand(Normal(), (m, n)))
v = rand(Normal(), m)
facts("gramschmidt") do
    # m = 1000
    # n = 50
    # Q, R = qr(rand(Normal(), (m, n)))
    # v = rand(Normal(), m)
    q, r, rho = gramschmidt(Q, v, 1/sqrt(2))
    @fact transpose(Q) * q => roughly(zeros(n), atol=1e-12)
    @fact Q * r + q .* rho => roughly(v)
    @fact transpose(Q) * v => roughly(r)
end

facts("gramschmidt!") do

    q = copy(v)
    r = zeros(n)
    rho = gramschmidt!(Q, q, 1/sqrt(2), r)
    @fact transpose(Q) * q => roughly(zeros(n), atol=1e-12)
    @fact Q * r + q .* rho => roughly(v)
    @fact transpose(Q) * v => roughly(r)
end

end
