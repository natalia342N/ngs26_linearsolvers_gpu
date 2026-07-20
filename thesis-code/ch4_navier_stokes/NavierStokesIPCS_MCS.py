from ngsolve import *
from ngsolve.webgui import Draw
import numpy as np
from ngsolve.krylovspace import QMRSolver
try:
    import ngsolve.ngscuda as ngscuda
    _have_cuda = True
except ImportError:
    _have_cuda = False

realcompile=True



__all__ = ["NavierStokes"]

# method from:
# https://epubs.siam.org/doi/pdf/10.1137/19M1248960

# Incremental Pressure-Correction Scheme


ngsglobals.symbolic_integrator_uses_diff = True

class NavierStokes:

    def __init__(self, mesh, nu, inflow, outflow, wall, uin, timestep, order=2, verbose=0, use_gpu=False):

        self.mesh = mesh
        self.use_gpu = use_gpu
        self.nu = nu
        self.timestep = timestep
        self.uin = CF(uin)
        self.inflow = inflow
        self.outflow = outflow
        self.wall = wall
        self.verbose = verbose

        if mesh.dim==2:
            self.boundaryvalues=mesh.BoundaryCF( { self.inflow:self.uin , }, default=CF((0,0)))
        else:
            self.boundaryvalues=mesh.BoundaryCF( { self.inflow:self.uin , }, default=CF((0,0,0)))


        self.useRT = False
        V = HDiv(mesh, order=order, dirichlet=inflow+"|"+wall+"|", RT=self.useRT, highest_order_dc=True)
        Vhat = TangentialFacetFESpace(mesh, order=order-1, dirichlet=inflow+"|"+wall+"|"+"|"+outflow)
        Sigma = Discontinuous(HCurlDiv(mesh, order = order-1, orderinner=order))
        S = MatrixValued(L2(mesh, order=order-1), skewsymmetric=True)
        self.Q = L2(mesh, order=order-1+(1 if self.useRT else 0))


        self.V = V
        Sigma = Compress(PrivateSpace(Sigma))
        S = Compress(PrivateSpace(S))

        self.X = V*Vhat*Sigma*S
        if self.verbose >= 1:
            print ("ndof X =", self.X.ndof, " = ", V.ndof, "+", Vhat.ndof) # , Sigma.ndof, S.ndof)

        for i in range(self.X.ndof):
            if self.X.CouplingType(i) == COUPLING_TYPE.WIREBASKET_DOF:
                self.X.SetCouplingType(i, COUPLING_TYPE.INTERFACE_DOF)

        u, uhat, sigma, W  = self.X.TrialFunction()
        v, vhat, tau, R  = self.X.TestFunction()

        dS = dx(element_boundary=True, bonus_intorder=2)
        n = specialcf.normal(mesh.dim)
        def tang(u): return u-(u*n)*n

        stokesA = -0.5/nu * InnerProduct(sigma,tau) * dx + \
           (div(sigma)*v+div(tau)*u) * dx + \
           (InnerProduct(W,tau) + InnerProduct(R,sigma)) * dx + \
           -(((sigma*n)*n) * (v*n) + ((tau*n)*n )* (u*n)) * dS + \
           (-(sigma*n)*tang(vhat) - (tau*n)*tang(uhat)) * dS

        self.astokes = BilinearForm (self.X, eliminate_hidden = True, condense=True, store_inner=True)
        self.astokes += stokesA
        self.astokes += 1e8*nu*div(u)*div(v) * dx

        self.a = BilinearForm (self.X, eliminate_hidden = True)
        self.a += stokesA

        self.div = BilinearForm(div(u)*self.Q.TestFunction()*dx).Assemble()



        self.gfu = GridFunction(self.X)
        self.f = LinearForm(self.X)

        self.mstar = BilinearForm(self.X, eliminate_hidden = True, condense=True)
        self.mstar += u*v * dx + timestep * stokesA

        if use_gpu:
            # Non-condensed form for GPU: regular sparse matrix supports CreateDeviceMatrix
            mstar_nc = BilinearForm(self.X, eliminate_hidden=True)
            mstar_nc += u*v * dx + timestep * stokesA
            mstar_nc.Assemble()
            adev_mstar = mstar_nc.mat.CreateDeviceMatrix()
            proj_mstar = Projector(self.X.FreeDofs(), True)
            jdev_mstar = proj_mstar.CreateDeviceMatrix()
            self._gpu_mstar = ngscuda.DevCGSolver(
                mat=adev_mstar, pre=jdev_mstar,
                adev_raw=adev_mstar, cdev_raw=jdev_mstar,
                precision=1e-7, printrates=False, maxsteps=2000)
            # Still need condensed mstar assembled for inner_solve / harmonic extension
            self.premstar = preconditioners.BDDC(self.mstar)
            self.mstar.Assemble()
        else:
            self.premstar = preconditioners.BDDC(self.mstar)
            self.mstar.Assemble()
            self.invmstar1 = solvers.CGSolver(self.mstar.mat, pre=self.premstar, atol=1e-7, printrates=False, maxiter=1000)
            ext  = IdentityMatrix() + self.mstar.harmonic_extension
            extT = IdentityMatrix() + self.mstar.harmonic_extension_trans
            self.invmstar = ext @ self.invmstar1 @ extT + self.mstar.inner_solve

        # the convective term


        VL2 = VectorL2(mesh, order=order, piola=True)
        ul2,vl2 = VL2.TnT()
        utot = ul2
        self.conv_l2 = BilinearForm(VL2, nonassemble=True)
        self.conv_l2 += InnerProduct(Grad(vl2)*utot, utot).Compile(realcompile=realcompile, wait=True) * dx(bonus_intorder=2)
        self.conv_l2 += (-IfPos(utot * n, utot*n*utot*vl2, \
                            utot*n*(ul2.Other(bnd=mesh.BoundaryCF({inflow:self.uin }, default=0.5*ul2)))*vl2)) \
            .Compile(realcompile=realcompile, wait=True) * dS

        self.convertl2 = V.ConvertL2Operator(VL2) @ self.X.Restriction(0)
        self.conv_operator = self.convertl2.T @ self.conv_l2.mat @ self.convertl2


        # use implicit step for convection:
        class MMstarInvClass(BaseMatrix):
            def __init__ (self, X, timestep, conv_l2):
                super(MMstarInvClass, self).__init__()
                self.X = X
                self.timestep = timestep
                self.conv_l2 = conv_l2

                self.rest = X.Restriction(0)

                self.VL2 = VectorL2(mesh, order=order, piola=True, dgjumps=True)
                self.ul2, self.vl2 = self.VL2.TnT()
                self.wind = GridFunction(self.VL2)

                self.MstarB = BilinearForm(self.VL2)
                self.MstarB += ul2*vl2*dx
                self.MstarB += -timestep * InnerProduct(Grad(self.vl2)*self.wind, self.ul2)*dx(bonus_intorder=2)
                self.MstarB += -timestep * (-self.wind*n) * IfPos(self.wind * n, self.ul2, self.ul2.Other()) * self.vl2 *dS


                # have to change to facet space
                self.MstarB_matfree = BilinearForm(self.VL2, matrix_free_bdb=True)
                self.MstarB_matfree += ul2*vl2*dx
                self.MstarB_matfree += -timestep * InnerProduct(Grad(self.vl2)*self.wind, self.ul2)*dx(bonus_intorder=2)
                # self.MstarB_matfree += -timestep * (-self.wind*n) * IfPos(self.wind * n, self.ul2, self.ul2.Other()) * self.vl2 *dS

                self.M = BilinearForm(self.ul2*self.vl2*dx).Assemble()


            def Mult (self, x, y):
                self.wind.vec.data = x
                self.MstarB.Assemble()

                mat = self.MstarB.mat.DeleteZeroElements(1e-12)

                blocks = self.MstarB.space.CreateSmoothingBlocks(blocktype="element")
                pre = mat.CreateBlockSmoother(blocks)

                inv = QMRSolver(mat=mat, matT=mat.CreateTranspose(sorted=False), pre=pre, printrates=False)
                y.data = self.M.mat@inv@self.conv_l2.mat * x


                def Shape (self):
                    return self.M.shape


        # setup problem for pressure projection (hybrid mixed)
        self.V2 = Discontinuous(self.V)
        self.gfp = GridFunction(self.Q)
        self.Qhat = FacetFESpace(mesh, order=order, dirichlet=outflow)
        self.Xproj = self.V2*self.Q*self.Qhat
        (u,p,phat),(v,q,qhat) = self.Xproj.TnT()
        aproj = BilinearForm(self.Xproj, condense=True)
        aproj += (-u*v+ div(u)*q + div(v)*p) * dx + (u*n*qhat+v*n*phat) * dS


        # Auxiliary space preconditioner (used for both CPU and GPU paths)
        fesh1 = H1(mesh, order=1, dirichlet="outlet")
        uh1,vh1 = fesh1.TnT()
        conv = self.Xproj.embeddings[2]@ConvertOperator(fesh1, self.Qhat, geom_free=True)
        ah1 = BilinearForm(grad(uh1)*grad(vh1)*dx)
        preh1 = preconditioners.H1AMG(ah1, maxcoarse=1000, verbose=self.verbose)
        ah1.Assemble()
        aproj.Assemble()
        prefacet = preconditioners.Local(aproj, GS=False, blocktype=["facet"]).mat
        cproj = conv @ preh1 @ conv.T + prefacet
        self.invproj1 = solvers.CGSolver(aproj.mat, pre=cproj, printrates=False, tol=1e-8, maxiter=1000)
        ext = IdentityMatrix() + aproj.harmonic_extension
        self.invproj = ext @ self.invproj1 @ ext.T + aproj.inner_solve

        normal = specialcf.normal(mesh.Materials(".*"))
        self.bproj = BilinearForm(div(self.V.TrialFunction())*q*dx + self.V.TrialFunction()*normal*qhat*dx(element_boundary=True), geom_free=True).Assemble()

        self.rhsproj = LinearForm (self.boundaryvalues*normal*qhat.Trace()*ds).Assemble()

        # mapping of discontinuous to continuous H(div)
        self.mapV = ConvertOperator(self.V2, self.V)@self.Xproj.restrictions[0]
        self.a.Assemble()
        self.f.Assemble()


    @property
    def velocity(self):
        return self.gfu.components[0]
    @property
    def pressure(self):
        return self.gfp

    def Save(self, filename):
        self.gfu.vec.FV().NumPy().tofile(filename)

    def Load(self, filename):
        import numpy as np
        self.gfu.vec.FV().NumPy()[:] = np.fromfile(filename)
        self.gfp.Set(-1e8*self.nu*div(self.gfu.components[0]))

    def SolveInitial(self, direct=False):

        self.gfu.components[0].Set(self.boundaryvalues, BND, definedon=self.mesh.Boundaries(self.inflow+"|"+self.wall))
        self.gfu.components[1].Set(self.boundaryvalues, BND, definedon=self.mesh.Boundaries(self.inflow+"|"+self.wall+"|"+self.outflow))

        self.Project(self.gfu.components[0].vec, True)
        self.astokes.Assemble()
        if direct:
            inv = self.astokes.mat.Inverse(self.X.FreeDofs(), inverse="sparsecholesky")
        else:
            prestokes = preconditioners.Local(self.astokes, block=True, GS=True, blocktype=["facet", "vertexpatch:hdivlo"])
            inv = solvers.CGSolver(self.astokes.mat, pre=prestokes, printrates=self.verbose>=2, maxiter=1000, tol=1e-6)

        if (self.astokes.condense):
            ext = IdentityMatrix()+self.astokes.harmonic_extension
            extT = IdentityMatrix()+self.astokes.harmonic_extension_trans
            fullinv = ext @ inv @ extT + self.astokes.inner_solve

            extm = IdentityMatrix()-self.astokes.harmonic_extension
            extmT = IdentityMatrix()-self.astokes.harmonic_extension_trans
            fullastokes = extmT @ (self.astokes.mat+self.astokes.inner_matrix) @ extm
        else:
            fullinv = inv
            fullastokes = self.astokes.mat

        rhs = (fullastokes * self.gfu.vec + self.f.vec).Evaluate()
        self.gfu.vec.data -= fullinv * rhs
        self.gfp.Set(-1e8*self.nu*div(self.gfu.components[0]))

    def AddForce(self, force):
        force = CF(force)
        v, vhat, tau, R  = self.X.TestFunction()
        self.f += -force*v*dx

    def DoTimeStep(self):

        self.temp = self.a.mat.CreateColVector()
        self.temp2 = self.a.mat.CreateRowVector()
        self.f.Assemble()

        self.temp.data = self.conv_operator * self.gfu.vec
        self.temp.data += self.f.vec
        self.temp.data += -self.a.mat * self.gfu.vec

        self.temp.data += self.div.mat.T * self.gfp.vec

        if self.use_gpu:
            fdev = self.temp.CreateDeviceVector(copy=True)
            self._gpu_mstar.Mult(fdev, self.temp2)
        else:
            self.temp2.data = self.invmstar * self.temp

        self.Project(self.temp2, False)
        self.gfu.vec.data += self.timestep * self.temp2.data

    def Project(self, vel, usebndvals):
        emb = self.X.Embedding(0)
        rest = self.X.Restriction(0)
        if usebndvals:
            projsol = (self.invproj * (self.bproj.mat @ rest * vel - self.rhsproj.vec)).Evaluate()
        else:
            projsol = (self.invproj @ self.bproj.mat @ rest * vel).Evaluate()
        vel.data -= emb @ self.mapV * projsol
        self.gfp.vec.data -= self.Xproj.Restriction(1)*projsol

    def ComputePressure(self, rhsstep1):
        emb = self.X.Embedding(0)
        rest = self.X.Restriction(0)
        projsol = (self.invproj @ self.mapV.T @ emb.T) * rhsstep1
        self.gfp.vec.data = -self.Xproj.Restriction(1)*projsol
