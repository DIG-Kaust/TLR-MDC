
__all__ = ['lsqr']
import numpy as np
from math import sqrt
from scipy.sparse.linalg.interface import aslinearoperator

eps = np.finfo(np.float64).eps
import os
from mpi4py import MPI
from os.path import join, exists
import time

def _sym_ortho(a, b):
    if b == 0:
        return np.sign(a), 0, abs(a)
    elif a == 0:
        return 0, np.sign(b), abs(b)
    elif abs(b) > abs(a):
        tau = a / b
        s = np.sign(b) / sqrt(1 + tau * tau)
        c = s * tau
        r = b / s
    else:
        tau = b / a
        c = np.sign(a) / sqrt(1+tau*tau)
        s = c * tau
        r = a / c
    return c, s, r

import time
def lsqr(A, b, damp=0.0, atol=1e-8, btol=1e-8, conlim=1e8,
    iter_lim=None, show=False, calc_var=False, x0=None, FIG_PATH=None):
    comm = MPI.COMM_WORLD
    comm.Barrier()
    mpirank = comm.rank
    mpisize = comm.size
    A = aslinearoperator(A)
    b = np.atleast_1d(b)
    if b.ndim > 1:
        b = b.squeeze()

    m, n = A.shape
    if iter_lim is None:
        iter_lim = 2 * n
    var = np.zeros(n)

    msg = ('The exact solution is  x = 0                              ',
           'Ax - b is small enough, given atol, btol                  ',
           'The least-squares solution is good enough, given atol     ',
           'The estimate of cond(Abar) has exceeded conlim            ',
           'Ax - b is small enough for this machine                   ',
           'The least-squares solution is good enough for this machine',
           'Cond(Abar) seems to be too large for this machine         ',
           'The iteration limit has been reached                      ')

    if show:
        print(' ')
        print('LSQR            Least-squares solution of  Ax = b')
        str1 = f'The matrix A has {m} rows and {n} columns'
        str2 = 'damp = %20.14e   calc_var = %8g' % (damp, calc_var)
        str3 = 'atol = %8.2e                 conlim = %8.2e' % (atol, conlim)
        str4 = 'btol = %8.2e               iter_lim = %8g' % (btol, iter_lim)
        print(str1)
        print(str2)
        print(str3)
        print(str4)

    itn = 0
    istop = 0
    ctol = 0
    if conlim > 0:
        ctol = 1/conlim
    anorm = 0
    acond = 0
    dampsq = damp**2
    ddnorm = 0
    res2 = 0
    xnorm = 0
    xxnorm = 0
    z = 0
    cs2 = -1
    sn2 = 0

    # Set up the first vectors u and v for the bidiagonalization.
    # These satisfy  beta*u = b - A*x,  alfa*v = A'*u.
    u = b
    bnorm = np.linalg.norm(b)
    if x0 is None: # this won't work with cupy
        x = np.zeros(n)
        beta = bnorm.copy()
    else:
        x = x0.copy()
        u = u - A.matvec(x)
        beta = np.linalg.norm(u)

    if beta > 0:
        u = (1/beta) * u
        v = A.rmatvec(u)
        alfa = np.linalg.norm(v)
    else:
        v = x.copy()
        alfa = 0

    if alfa > 0:
        v = (1/alfa) * v
    w = v.copy()

    rhobar = alfa
    phibar = beta
    rnorm = beta
    r1norm = rnorm
    r2norm = rnorm

    # Reverse the order here from the original matlab code because
    # there was an error on return when arnorm==0
    arnorm = alfa * beta
    if arnorm == 0:
        if show:
            print(msg[0])
        return x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var

    head1 = '   Itn      x[0]       r1norm     r2norm '
    head2 = ' Compatible    LS      Norm A   Cond A'

    if show:
        print(' ')
        print(head1, head2)
        test1 = 1
        test2 = alfa / beta
        str1 = '%6g %12.5e' % (itn, x[0])
        str2 = ' %10.3e %10.3e' % (r1norm, r2norm)
        str3 = '  %8.1e %8.1e' % (test1, test2)
        print(str1, str2, str3)

    # Main iteration loop.


    while itn < iter_lim:
        comm.Barrier()
        tstart = time.time()
        itn = itn + 1
        # Perform the next step of the bidiagonalization to obtain the
        # next  beta, u, alfa, v. These satisfy the relations
        #     beta*u  =  a*v   -  alfa*u,
        #     alfa*v  =  A'*u  -  beta*v.
        t0matvec = time.time()
        u = A.matvec(v) - alfa * u
        t1matvec = time.time()
        if mpirank == 0:
            print(f"MDC matvec time : {t1matvec-t0matvec} s.")
            
        beta = np.linalg.norm(u)
        enterbeta = False
        if beta > 0:
            enterbeta = True
            u = (1/beta) * u
            anorm = sqrt(anorm**2 + alfa**2 + beta**2 + dampsq)
            t0rmatvec = time.time()
            v = A.rmatvec(u) - beta * v
            t1rmatvec = time.time()
            if mpirank == 0:
                print(f"MDC rmatvec time : {t1rmatvec-t0rmatvec} s.")
            alfa = np.linalg.norm(v)
            if alfa > 0:
                v = (1 / alfa) * v
        # Use a plane rotation to eliminate the damping parameter.
        # This alters the diagonal (rhobar) of the lower-bidiagonal matrix.
        if damp > 0:
            rhobar1 = sqrt(rhobar**2 + dampsq)
            cs1 = rhobar / rhobar1
            sn1 = damp / rhobar1
            psi = sn1 * phibar
            phibar = cs1 * phibar
        else:
            # cs1 = 1 and sn1 = 0
            rhobar1 = rhobar
            psi = 0.

        # Use a plane rotation to eliminate the subdiagonal element (beta)
        # of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.
        cs, sn, rho = _sym_ortho(rhobar1, beta)

        theta = sn * alfa
        rhobar = -cs * alfa
        phi = cs * phibar
        phibar = sn * phibar
        tau = sn * phi

        # Update x and w.
        t1 = phi / rho
        t2 = -theta / rho
        dk = (1 / rho) * w

        x = x + t1 * w
        w = v + t2 * w
        ddnorm = ddnorm + np.linalg.norm(dk)**2

        if calc_var:
            var = var + dk**2

        # Use a plane rotation on the right to eliminate the
        # super-diagonal element (theta) of the upper-bidiagonal matrix.
        # Then use the result to estimate norm(x).
        delta = sn2 * rho
        gambar = -cs2 * rho
        rhs = phi - delta * z
        zbar = rhs / gambar
        xnorm = sqrt(xxnorm + zbar**2)
        gamma = sqrt(gambar**2 + theta**2)
        cs2 = gambar / gamma
        sn2 = theta / gamma
        z = rhs / gamma
        xxnorm = xxnorm + z**2

        # Test for convergence.
        # First, estimate the condition of the matrix  Abar,
        # and the norms of  rbar  and  Abar'rbar.
        acond = anorm * sqrt(ddnorm)
        res1 = phibar**2
        res2 = res2 + psi**2
        rnorm = sqrt(res1 + res2)
        arnorm = alfa * abs(tau)

        # Distinguish between
        #    r1norm = ||b - Ax|| and
        #    r2norm = rnorm in current code
        #           = sqrt(r1norm^2 + damp^2*||x||^2).
        #    Estimate r1norm from
        #    r1norm = sqrt(r2norm^2 - damp^2*||x||^2).
        # Although there is cancellation, it might be accurate enough.
        if damp > 0:
            r1sq = rnorm**2 - dampsq * xxnorm
            r1norm = sqrt(abs(r1sq))
            if r1sq < 0:
                r1norm = -r1norm
        else:
            r1norm = rnorm
        r2norm = rnorm

        # Now use these norms to estimate certain other quantities,
        # some of which will be small near a solution.
        test1 = rnorm / bnorm
        test2 = arnorm / (anorm * rnorm + eps)
        test3 = 1 / (acond + eps)
        t1 = test1 / (1 + anorm * xnorm / bnorm)
        rtol = btol + atol * anorm * xnorm / bnorm

        # The following tests guard against extremely small values of
        # atol, btol  or  ctol.  (The user may have set any or all of
        # the parameters  atol, btol, conlim  to 0.)
        # The effect is equivalent to the normal tests using
        # atol = eps,  btol = eps,  conlim = 1/eps.
        if itn >= iter_lim:
            istop = 7
        if 1 + test3 <= 1:
            istop = 6
        if 1 + test2 <= 1:
            istop = 5
        if 1 + t1 <= 1:
            istop = 4

        # Allow for tolerances set by the user.
        if test3 <= ctol:
            istop = 3
        if test2 <= atol:
            istop = 2
        if test1 <= rtol:
            istop = 1
        tend = time.time()
        if mpirank == 0:
            print("Iteration time {:.6f} s.".format(tend-tstart))

        if show:
            # See if it is time to print something.
            prnt = False
            if n <= 40:
                prnt = True
            if itn <= 10:
                prnt = True
            if itn >= iter_lim-10:
                prnt = True
            # if itn%10 == 0: prnt = True
            if test3 <= 2*ctol:
                prnt = True
            if test2 <= 10*atol:
                prnt = True
            if test1 <= 10*rtol:
                prnt = True
            if istop != 0:
                prnt = True

            if prnt:
                str1 = '%6g %12.5e' % (itn, x[0])
                str2 = ' %10.3e %10.3e' % (r1norm, r2norm)
                str3 = '  %8.1e %8.1e' % (test1, test2)
                str4 = ' %8.1e %8.1e' % (anorm, acond)
                print(str1, str2, str3, str4)

        if istop != 0:
            break

    # End of iteration loop.
    # Print the stopping condition.
    if show:
        print(' ')
        print('LSQR finished')
        print(msg[istop])
        print(' ')
        str1 = 'istop =%8g   r1norm =%8.1e' % (istop, r1norm)
        str2 = 'anorm =%8.1e   arnorm =%8.1e' % (anorm, arnorm)
        str3 = 'itn   =%8g   r2norm =%8.1e' % (itn, r2norm)
        str4 = 'acond =%8.1e   xnorm  =%8.1e' % (acond, xnorm)
        print(str1 + '   ' + str2)
        print(str3 + '   ' + str4)
        print(' ')

    return x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var
