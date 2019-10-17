#!/usr/bin/env python3
# -*- coding: utf-8 -*-
using DifferentialEquations
# 1.  define a problem
#   equation, initial condition, timespan

#sol = solve(prob,DynamicSS(Tsit5()),abstol=1e-5)
#sol = solve(prob,DynamicSS(CVODE_BDF()),dt=1)
function sode(f, u0, dt, abstol, reltol)
    prob = SteadyStateProblem(f, u0, mass_matrix=I)
    sol = solve(prob, DynamicSS(CVODE_BDF()),
                dt=dt, abstol=abstol, reltol=reltol)
    return sol.t, sol.u
end

function sode_ssrootfind(f, u0, abstol, reltol)
    prob = SteadyStateProblem(f, u0, mass_matrix=I)
    sol = solve(prob, SSRootfind())
    return sol
