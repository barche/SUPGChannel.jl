module SUPGChannel

using Gridap
using LineSearches: MoreThuente

export channelflow

function boundaryconditions(u0, periodic)
   top = "tag_5"
   bottom = "tag_6"
   inlet = "tag_7"
   outlet = "tag_8"
   outlet_top = "tag_2" # top right corner, not adding corners results in a "jump" at the outlet
   outlet_bottom = "tag_4" # bottom right corner
   inlet_top = "tag_1"
   inlet_bottom = "tag_3"
   u_diri_tags = [top, bottom]
   u_walls(x, t::Real) = VectorValue(0.0, 0.0)
   u_in_v(x, t::Real) = VectorValue(u0, 0.0)
   u_walls(t::Real) = x -> u_walls(x, t)
   u_in_v(t::Real) = x -> u_in_v(x, t)
   p_diri_tags = String[]
   p_diri_values = Float64[]

   if periodic
       u_diri_values = [u_walls, u_walls]
   else
       append!(u_diri_tags, [inlet, inlet_top, inlet_bottom, outlet_top, outlet_bottom])
       append!(p_diri_tags, [outlet, outlet_top, outlet_bottom])
       u_diri_values = [u_walls, u_walls, u_in_v, u_in_v, u_in_v, u_walls, u_walls]
       append!(p_diri_values, [0, 0, 0])
   end

   return u_diri_tags,p_diri_tags,u_diri_values,p_diri_values
end

function channelflow(;u0=1.0, N=32::Integer, h=1.0, writemodel=false, periodic=true, order = 1)
   ν = 1.0 # Kinematic vicosity
   body_force = periodic ? 2*ν*u0/h^2 : 0.0

   model = makemesh(; N, writemodel, periodic)

   u_diri_tags,p_diri_tags,u_diri_values,p_diri_values = boundaryconditions(u0,  periodic)

   reffeᵤ = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
   V = TestFESpace(model, reffeᵤ, conformity=:H1, dirichlet_tags=u_diri_tags)
   reffeₚ = ReferenceFE(lagrangian, Float64, order)

   if periodic
      Q = TestFESpace(model, reffeₚ, conformity=:H1, constraint=:zeromean)
   else
      Q = TestFESpace(model, reffeₚ, conformity=:H1, dirichlet_tags=p_diri_tags)
   end

   U = TransientTrialFESpace(V, u_diri_values)
   if periodic
      P = TrialFESpace(Q)
   else
      P = TrialFESpace(Q, p_diri_values)
   end

   Y = MultiFieldFESpace([V, Q])
   X = TransientMultiFieldFESpace([U, P])

   degree = order*2
   Ω = Triangulation(model)
   dΩ = Measure(Ω, degree)

   h = lazy_map(h -> h^(1 / 2), get_cell_measure(Ω))

   function τ(u, h)
      τ₂ = h^2 / (4 * ν)
      val(x) = x
      val(x::Gridap.Fields.ForwardDiff.Dual) = x.value
  
      u = val(norm(u))
      if iszero(u)
          return τ₂
      end
      τ₁ = h / (2 * u)
      τ₃ = dt / 2
      return 1 / (1 / τ₁ + 1 / τ₂ + 1 / τ₃)
   end

   τb(u, h) = (u ⋅ u) * τ(u, h)


   hf(x, t::Real) = VectorValue(body_force, 0)
   hf(t::Real) = x -> hf(x, t)

   res(t, (u, p), (v, q)) = ∫(
      q ⊙ (∇⋅u) + (τ∘(u,h)) * ∇(q) ⊙ (u⋅∇(u)) +
      (τ∘(u,h)) * (∇(q) ⊙ ∇(p)) +
      ν * ∇(v) ⊙ ∇(u) + (v + (τ∘(u,h)) * u⋅∇(u)) ⊙ (u⋅∇(u)) +
      (v + (τ∘(u,h)) * u⋅∇(u)) ⊙ ∇(p) +
      (τb∘(u,h)) * (∇⋅v) ⊙ (∇⋅u) +
      (τ∘(u,h)) * ∇(q) ⊙ ∂t(u) +
      (v + (τ∘(u,h)) * u⋅∇(u)) ⊙ ∂t(u) +
      (v + (τ∘(u,h)) * u⋅∇(u)) ⊙ hf(t)
   )dΩ

   op = TransientFEOperator(res, X, Y)

   nls = NLSolver(show_trace=true, method=:newton, linesearch=MoreThuente(), iterations=30)

   U0 = U(0.0)
   P0 = P(0.0)
   X0 = X(0.0)

   u_in_v(x, t::Real) = VectorValue(u0, 0.0)
   u_in_v(t::Real) = x -> u_in_v(x, t)
   uh0 = interpolate_everywhere(u_in_v(0), U0)
   ph0 = interpolate_everywhere(0.0, P0)
   xh0 = interpolate_everywhere([uh0, ph0], X0)

   t0 = 0.0
   dt = 1.0 # timestep
   ntimesteps = 10
   tend = ntimesteps*dt

   θ = 1.0

   ode_solver = ThetaMethod(nls, dt, θ)


   sol_t = solve(ode_solver, op, xh0, t0, tend)

   for (i,(xh_tn, _)) in enumerate(sol_t)
      writevtk(Ω, "out_channel_$i.vtu"; cellfields=["Velocity" => xh_tn[1], "Pressure" => xh_tn[2]])
   end
end

function benchmark_assembly()
   du = get_trial_fe_basis(U)
   dv = get_fe_basis(V)
   uhd = zero(U)
   data = collect_cell_matrix_and_vector(U,V,a,l,uhd)
   Tm = SparseMatrixCSC{Float64,Int32}
   Tv = Vector{Float64}
   assem = SparseMatrixAssembler(Tm,Tv,U,V)
   A, b = assemble_matrix_and_vector(assem,data) # This is the assembly loop + allocation and compression of the matrix
   assemble_matrix_and_vector!(A,b,assem,data) # This is the in-place assembly loop on a previously allocated matrix/vector.
end

# Simplified version of Carlo Brunell's mesh_channel (2D and serial only)
function makemesh(;h=1,N=32::Integer, writemodel=false, periodic=true)

  Lx = 8*h
  Ly = 2*h
  nx = N
  ny = N


  #N = 32 # Partition (i.e., number of cells per space dimension)
  function stretching(x::Point)
     m = zeros(length(x))
     m[1] = x[1]

     gamma1 = 2.5
     m[2] = -tanh(gamma1 * (x[2])) / tanh(gamma1)
     if length(x) > 2
        m[3] = x[3]
     end
     Point(m)
  end

  pmin = Point(0, -Ly / 2)
  pmax = Point(Lx, Ly / 2)
  partition = (nx, ny)
  periodic_tuple = (periodic, false)
  model_name = "channel"

  model = CartesianDiscreteModel(pmin, pmax, partition, map=stretching, isperiodic=periodic_tuple)

  if writemodel
     writevtk(model, model_name)
  end
  return model

end



end
