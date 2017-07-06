# Solving -Δ^s u = f on an interval
using FastGaussQuadrature
using SymEngine
using Plots
using Dierckx

struct ∫
    lower::Number
    upper::Number
    degree::Int
    xq::Vector{Real}
    wq::Vector{Real}
end

function ∫(lower::Number, upper::Number, degree::Int)
    @assert lower <= upper
    lower == upper && return 0.
    # Map points and weights to domain
    xq, wq = gausslegendre(5*degree)  # on (-1, 1)
    xq = lower*(1-xq)/2 + upper*(1+xq)/2
    wq *= abs(upper-lower)/2

    ∫(lower, upper, degree, xq, wq)
end

(I::∫)(f::Function) = dot(I.wq, f.(I.xq))

typealias Callable Union{Function, Spline1D}

# -----------------------------------------------------------------------------

import Base: +, -, *, /

for op in (:+, :-, :*, :/)
    @eval ($op)(f::Callable, g::Callable) = x -> ($op)(f(x), g(x))
    @eval ($op)(f::Callable, a::Number) = x -> ($op)(f(x), a)
    @eval ($op)(a::Number, f::Callable) = x -> ($op)(a, f(x))
end

# -----------------------------------------------------------------------------

function solve(s::Real, deg::Int, f::Callable, bdries::Tuple{Number, Number}, bvalues::Tuple{Number, Number})
    @assert -1 <= s <= 1
    a, b = bdries

    # Basis function of the Galerkin space
    ϕ = ((x -> sqrt(2/(b-a))*sin(k*pi*(x-a)/(b-a))) for k in 1:deg)
    # Transformation of f to coefficient space
    integral = ∫(a, b, 5*deg)
    F = [integral(f*ϕ_i) for ϕ_i in ϕ]

    # Coefs of the homog part of the solution
    F = F./[(k*pi/(b-a))^(2s) for k in 1:deg]

    u0(x) = sum(Fk*sqrt(2/(b-a))*sin(k*pi*(x-a)/(b-a)) for (k, Fk) in enumerate(F))

    # Finally the bit that handles bcs
    A, B = bvalues
    u(x) = u0(x) + B*(x-a)/(b-a) + A*(b-x)/(b-a)
end

# -----------------------------------------------------------------------------

gr()

bdries = (-1, 1)
###
fraction_s = 0.5

#if fraction_s == 1.
#    x = symbols("x")
#    uexact = x*(1-x)+sin(4x) + x^5
#    force = lambdify(-diff(uexact, x, 2))
#    uexact = lambdify(uexact)
#end

if fraction_s != 1.
    uexact(x) = 1 - x^2
    x = collect(linspace(-1, 1, 10000))
    y = sum(32*sin(k*pi/2)/((pi*k)^3)*(k*pi/2)^(2*fraction_s)*cos.(k*pi/2*x)
                    for k in 1:3000)
    force = Spline1D(x, y)
end

bvalues = (uexact(first(bdries)), uexact(last(bdries)))

a, b = bdries
u = nothing
for deg in (4:3:45)
    u = solve(1, deg, force, bdries, bvalues)

    integral = ∫(a, b, 5*deg)
    error = sqrt(integral((u-uexact)*(u-uexact)))

    println("$(deg) gives $(error)")
end

x = linspace(a, b, 200)
plot(x, u.(x))
plot!(x, uexact.(x))

## FIXME: domain decomposition, 2 domain, additive
#mid = 0.0
#left = (first(bdries), mid+0.1)
#right = (mid-0.1, last(bdries))
#
#deg = 12
#schwarz = :multiplicative
#chi(x) = (x <= mid) ? 1. : 0.
#
#errors = []
#x = linspace(a, b, 200)
#
#u0 = x -> 0.0
#for iters in 1:100
## Left solve
#    vleft = (uexact(first(left)), u0(last(left)))
#    u_left = solve(1, deg, force, left, vleft)
#
#    ubr = (schwarz == :additive) ? u0(first(right)) : u_left(first(right))
#
#    vright = (ubr, uexact(last(right)))
#    u_right = solve(1, deg, force, right, vright)
#
#    u0 = chi*u_left + (1-chi)*u_right
#
#    error = norm((u0-uexact).(x), Inf)
#    push!(errors, error)
#end
#
#plot(x, uexact.(x))
#plot!(x, u0.(x))
#
#k = collect(1:length(errors))
#plot(k, errors, marker=:*)
#yaxis!("L2 error", :log10)
