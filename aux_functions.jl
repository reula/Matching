function D2x_SBP_ts(v,par_Dx_st) #threads safe version
    N, dx = par_Dx_st
    dv = Vector{Float64}(undef,N)
    dv[1] = (v[2] - v[1])/dx
    for j in 2:(N-1)
        dv[j] = (v[j+1] - v[j-1])/dx/2
    end
    dv[N] = (v[N] - v[N-1])/dx
    return dv[:]
end


#### fourth order derivative coefficients ###############################

const Qd = [-24/17 59/34 -4/17 -3/34    0  0 
            -1/2   0    1/2    0      0  0
            8/86  -59/86  0   59/86  -8/86 0
            3/98   0   -59/98  0 32/49  -4/49]
const h_00_Qd = 17/48
"""
Fourth order finite difference satisfying summation by parts and thread safe 
"""
function D4x_SBP_ts(v,par_Dx,Qd)
    N, dx = par_Dx
    dv = Vector{Float64}(undef,N)
    dv[1] = (v[1]*Qd[1,1] + v[2]*Qd[1,2] + v[3]*Qd[1,3] + v[4]*Qd[1,4])/dx
    dv[2] = (v[1]*Qd[2,1] + v[3]*Qd[2,3])/dx
    dv[3] = (v[1]*Qd[3,1] + v[2]*Qd[3,2] + v[4]*Qd[3,4] + v[5]*Qd[3,5])/dx
    dv[4] = (v[1]*Qd[4,1] + v[3]*Qd[4,3] + v[5]*Qd[4,5] + v[6]*Qd[4,6])/dx
    for i in 5:N-4
        dv[i] = (-v[i+2] + 8.0*v[i+1] - 8.0 * v[i-1] + v[i-2] )/dx/12
    end
    dv[N]   = -(v[N]*Qd[1,1] + v[N-1]*Qd[1,2] + v[N-2]*Qd[1,3] + v[N-3]*Qd[1,4])/dx
    dv[N-1] = -(v[N]*Qd[2,1] + v[N-2]*Qd[2,3])/dx
    dv[N-2] = -(v[N]*Qd[3,1] + v[N-1]*Qd[3,2] + v[N-3]*Qd[3,4] + v[N-4]*Qd[3,5])/dx
    dv[N-3] = -(v[N]*Qd[4,1] + v[N-2]*Qd[4,3] + v[N-4]*Qd[4,5] + v[N-5]*Qd[4,6])/dx
    return dv[:]
end

############################################ TIME INTEGRATION #########################################

function RK4(f,y0,t0,h,p)
    k1 = h*f(y0,t0,p)
    k2 = h*f(y0+0.5*k1, t0+0.5*h,p)
    k3 = h*f(y0+0.5*k2, t0+0.5*h,p)
    k4 = h*f(y0+k3, t0+h,p)
    return y0 + (k1 + 2k2 + 2k3 + k4)/6
end

"""
Runge Kutta step
This is the one used here, for some reason we need the .=
""" 
function RK4_Step!(f,y0,t0,h,p_f,par_RK)
    k1, k2, k3, k4 = par_RK
    k1 = h*f(k1,y0,t0,p_f)
    k2 = h*f(k2,y0+0.5*k1, t0+0.5*h,p_f)
    k3 = h*f(k3,y0+0.5*k2, t0+0.5*h,p_f)
    k4 = h*f(k4,y0+k3, t0+h,p_f)
    y0 .= y0 + (k1 + 2k2 + 2k3 + k4)/6
    return y0[:]
end

function ODEproblem(Method, f, y0, intervalo, M,p)
    ti,tf = intervalo
    h = (tf-ti)/(M-1)
    N = length(y0)
    y = zeros(M,N)
    t = zeros(M)
    y[1,:] = y0
    t[1] = ti
    for i in 2:M
        t[i] = t[i-1] + h
        y[i,:] = Method(f,y[i-1,:],t[i-1],h,p)
    end
    return (t ,y)
end

"""
Function to integrate W along u=constant, 
the initial value is vm at the last point on the left
"""
function get_W!(W,ϕ_R,ρ_R,vm0,p_get)
    Nl, dl, Nr, dr = p_get
    W[1] = vm0
    for i in 1:(Nr-1)
        W[i+1] = W[i] + ρ_R[i]*dr/2 
    end
    return W[:]
end

"""
The evolution ecuation
"""
function F!(du,u,t,p_F)
    Nl, dl, Nr, dr, ρ_L, ρ_R = p_F
    p_get = Nl, dl, Nr, dr
    par_Dx_r = (Nr, dr)
    par_Dx_l = (Nl, dl)
    ϕ_L = view(u,1:Nl)
    vp = view(u,Nl+1:2Nl)
    vm = view(u,2Nl+1:3Nl)
    ϕ_R = view(u,3Nl+1:3Nl+Nr)
    S = view(u,3Nl+Nr+1:3Nl+2Nr)
    W = view(u,3Nl+2Nr+1:3Nl+3Nr)
    dϕ_L = view(du,1:Nl)
    dvp = view(du,Nl+1:2Nl)
    dvm = view(du,2Nl+1:3Nl)
    dϕ_R = view(du,3Nl+1:3Nl+Nr)
    dS = view(du,3Nl+Nr+1:3Nl+2Nr)
    dW = view(du,3Nl+2Nr+1:3Nl+3Nr)
    dϕ_L = (vp+vm)/2
    dvp = D4x_SBP_ts(vp,par_Dx_l,Qd) + ρ_L
    dvm = -D4x_SBP_ts(vm,par_Dx_l,Qd) + ρ_L
    dϕ_R = (S+W)/2
    dS = D4x_SBP_ts(S,par_Dx_r,Qd)/2 + ρ_R/2
    get_W!(W,ϕ_R,ρ_R,vm[end],p_get) # this updates as an integral
    # Penalties
    h_00 = 17/48
    σ = 1/2/h_00/dl
    dvp[end] += σ*(S[1]-vp[end]) # vp takes the outgoing left mode from right side
    dvm[1] += σ*(vp[1]-vm[1]) # make ϕ_x = 0 at origin
    dS[end] += -σ*S[end] # nothing enters from the far rigth
    return [dϕ_L;dvp;dvm;dϕ_R;dS;dW][:]
    #return du[:]
end

###################### algunas funciones para hacer los datos iniciales ###############

function bump(x,x0,x1,p,A)
    if x > x0 && x < x1
    return A*(x-x0)^p*(x-x1)^p*((x1-x0)/2)^(-2p)
    else
        return 0
    end
end

function bump_x(x,x0,x1,p,A) #x derivative of b
    if x > x0 && x < x1
    return p*A*((x1-x0)/2)^(-2p)*(x-x0)^(p-1)*(x-x1)^(p-1)*(2x-x0-x1)
    else
        return 0
    end
end
#plot(x->bump(x,1,2,4,1)) #checked ok