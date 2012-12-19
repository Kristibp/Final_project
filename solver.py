from dolfin import *
import numpy, sys

eps = 1E-14
def u0_boundary(x,on_boundary):
    return on_boundary and abs(x[-1]) < eps;


class Problem:
    mesh = None;
    V = None;
    f = None;
    I = None;
    a = None;

    def generate_first_verification(self,h):
        self.nx = int(round(1/sqrt(h))); self.ny = self.nx;
        print 'nx = ny =',self.nx
        self.mesh = UnitSquare(self.nx, self.ny);
        self.V = FunctionSpace(self.mesh, 'Lagrange', 1);
        self.f = Constant(0.0);
        self.a = Constant(1.0);
        self.I = Expression("cos(pi*x[0])")
        self.rho = 1;

        eps = 1E-14;
        self.b = u0_boundary

        self.T = 0.1
        self.nt = int(round(self.T/h))
        self.dt = h
        print 'nt = ', self.nt
        self.exact = Expression("exp(-pi*pi*t)*cos(pi*x[0])",t=self.T)

    def generate_second_verification(self,T):
        self.nx = 10;
        self.mesh = UnitInterval(self.nx);
        self.V = FunctionSpace(self.mesh, 'Lagrange', 1);
        self.rho = 2;

        """For some reason I could not use exponentials in the Expression. The solution seems to work, but is a bit clunky to read"""
        self.f = Expression('-rho*x[0]*x[0]*x[0]/3.0 + rho*x[0]*x[0]/2.0 + 8.0*t*t*t*x[0]*x[0]*x[0]*x[0]*x[0]*x[0]*x[0]/9 - 28.0*t*t*t*x[0]*x[0]*x[0]*x[0]*x[0]*x[0]/9 + 7.0*t*t*t*x[0]*x[0]*x[0]*x[0]*x[0]/2 - 5.0*t*t*t*x[0]*x[0]*x[0]*x[0]/4 + 2*t*x[0] - t',t=0, rho = self.rho);

        self.a = Expression('1 + u*u',u=0);
        self.I = Constant(0.0)

        eps = 1E-14;
        self.b = u0_boundary

        self.T = T
        self.nt = 10
        self.dt = self.T/self.nt;
        self.exact = Expression("t*x[0]*x[0]*(1/2 - x[0]/3.0)",t=self.T)

    def generate_gaussian(self,beta):
        self.nx = 10; self.ny = 10;
        self.mesh = UnitSquare(self.nx, self.ny);
        self.V = FunctionSpace(self.mesh, 'Lagrange', 1);
        self.f = Constant(0.0);
        self.a = Expression('1 + B*u*u', B = beta, u = 0);
        theta = 2.0
        self.I = Expression("exp(- 1/(2.0*theta*theta)*(x[0]*x[0] + x[1]*x[1]))",theta = theta)
        self.rho = 1;

        eps = 1E-14;
        self.b = u0_boundary

        self.T = 0.1
        self.nt = 10
        self.dt = self.T/self.nt
        self.exact = Expression("exp(-pi*pi*t)*cos(pi*x[0])",t=self.T)

class Solver:
    V = None;
    mesh = None;
    f = None;
    u_p = None;
    alph = None;
    bcs = None;

    u_test = None;

    def __init__(self, problem):
            self.V = problem.V;
            self.mesh = problem.mesh;
            self.f = problem.f;
            self.u_p = interpolate(problem.I,self.V);
            self.alph = problem.a
            self.k = problem.dt/problem.rho;
            self.b = problem.b;
            self.nt = problem.nt;
            self.dt = problem.dt;
            self.t = 0;

    def Solve(self):
        for i in range(1,self.nt+1):
            self.t = self.t + self.dt
            self.Picard()
        self.u = self.u_p

    def Picard(self):
        V = self.V;
        mesh = self.mesh;
        f = self.f;
        u_p = self.u_p;
        u = TrialFunction(V);
        v = TestFunction(V);

        bcs = DirichletBC(self.V,u_p,self.b)
        self.alph.u_p = u_p
        f.t = self.t
        a = (self.alph*inner(grad(u), grad(v)) + inner(u,v))*dx;
        """self.k*self.alph(self.u_p)*"""
        L = inner(u_p + self.k*f,v)*dx;
        """a = (self.dt*inner(grad(u), grad(v)) + inner(u,v))*dx"""
        u = Function(V);
        """L = -inner(u_p,v)*dx"""
        solve(a == L, u, bcs);
        self.u_p.assign(u)
        if self.u_test == None: self.u_test = u;
class Tester:
    def convergance_test_1(self):
        for h in [0.1, 0.01, 0.001]:
            ph = Problem();
            ph.generate_first_verification(h)
            sh = Solver(ph);
            sh.Solve()
            u = sh.u
            u_e = project(ph.exact,ph.V)
            e = u.vector().array() - u_e.vector().array()
            E = sqrt(numpy.sum(e**2))/u.vector().array().size
            print 'E/h = ', E/h
    
    def manufactured_test(self):
        for T in [3, 1, 0.2]:
            p = Problem()
            p.generate_second_verification(T)
            s = Solver(p);
            s.Solve()
            u = s.u
            u_e = project(p.exact,p.V)
            e = u.vector().array() - u_e.vector().array()
            E = sqrt(numpy.sum(e**2))/u.vector().array().size
            print 'T = ', T, ' E = ', E
            
    def gaussian_test(self):
        for B in [3, 1, 0.2]:
            p = Problem()
            p.generate_gaussian(B)
            s = Solver(p);
            s.Solve()
            print 'Beta = ', B
            print s.u.vector().array()

if __name__ == '__main__':
    tester1 = Tester()
    """tester1.convergance_test_1()"""
    """tester1.manufactured_test()"""
    """tester1.gaussian_test()"""
