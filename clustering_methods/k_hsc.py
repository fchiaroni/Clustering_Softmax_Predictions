# Implementation based on HSC authors code

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

class Multinomial( object ):

    ##################################################################################
    # static methods
    ##################################################################################

    @staticmethod
    def hilbert_distance( a, b ):
        return a.hilbert_distance( b )

    ################################################################################
    # compute means and centers
    ################################################################################

    @staticmethod
    def hilbert_cut( a, b, r ):
        return a.hilbert_cut( b, r )

    ################################################################################
    # compute means and centers
    ################################################################################

    @staticmethod
    def hilbert_center( distributions, max_itrs=200, tol=1e-4, verbose=False ):
        '''
        compute the Hilbert center of distributions
        '''
        return Multinomial.__center( distributions, Multinomial.hilbert_distance, Multinomial.hilbert_cut, max_itrs, verbose )

    @staticmethod
    def __center( distributions, compute_distance, compute_cut, max_itrs, verbose ):
        '''
        compute minimax center
        '''
        # initialize the center
        C = Multinomial( np.random.choice( distributions ).p )

        # walk to the farest point
        for i in range( 1, max_itrs+1 ):
            far = np.argmax( [ compute_distance( _d, C ) for _d in distributions ] )
            C = compute_cut( C, distributions[far], 1/(i+1) )

        if verbose:
            _distance = [ compute_distance( _d, C ) for _d in distributions ]
            top5 = sorted( _distance, reverse=True )[:5]
            print( 'after {0} iterations:'.format( i ) )
            print( ' '.join( [ '%.3f' % _ for _ in top5 ] ) )

        return C

    ##########################################################################################
    # clustering
    ##########################################################################################

    @staticmethod
    def hilbert_kcenters( distributions, k, max_itrs=100, max_center_itrs=100, 
                          seed=None, init_strategy="kmeans_plusplus_init", 
                          verbose=False ):
        return Multinomial.__kmeans( distributions, k, 
                                     Multinomial.hilbert_distance,
                                     Multinomial.hilbert_center,
                                     max_itrs, max_center_itrs, seed, 
                                     init_strategy, verbose )

    @staticmethod
    def __kmeansplusplus( distributions, k, compute_distance ):
        '''
        choosing k distributions based on kmeans++
        this is for initalizing the kmeans algorithm
        '''
        print("kmeans plusplus init")
        centers   = [ np.random.choice(distributions) ]
        _distance = np.array( [ compute_distance( _d, centers[0] ) for _d in distributions ] )
        # tackle infinity distance
        infidx = np.isinf( _distance )
        idx = np.logical_not( infidx )
        _distance[infidx] = _distance[idx].max()

        while len(centers) < k:
            p = _distance**2
            p /= p.sum() + distributions[0].eps
            centers.append( np.random.choice( distributions, p=p ) )

            _distance = np.minimum( _distance, [ compute_distance( _d, centers[-1] )
                                                 for _d in distributions ] )
        return centers

    @staticmethod
    def __vertices_init( distributions, k, compute_distance ):
        '''
        near vertices init
        '''
        print("vertices init")
        centers = None
        if k==len(distributions[0].p):
            center_max_val = (1.-(1/k))
            centers = np.identity(k)*center_max_val
            centers[centers==0] = centers[centers==0] + (1-center_max_val)/(k-1) # because Hilbert metric cannot deal with centers near to simplex boundaries
        else:
            print("vertices_init is not possible because n_dim != n_clusters.")
        return [Multinomial(p) for p in centers]

    @staticmethod
    def __kmeans( distributions, k, compute_distance, compute_center, max_itrs, max_center_itrs, seed, init_strategy, verbose ):
        '''
        general kmeans clustering

        distributions    -- a list of distributions
        k                -- number of clusters
        compute_distance -- callback
        compute_center   -- callback
        max_itrs         -- maximum number of iterations
        max_center_itrs  -- maximum number of iterations for center computation
        seed             -- random seed
        init_strategy    -- whether to use the kmeans++ seeding
        verbose          -- verbose or not

        return the clustering scheme, e.g.
        [ 0, 0, 1, 2, 1, 0, 1, ... ]
        '''
        
        if seed is not None: np.random.seed( seed )
        if init_strategy == "kmeans_plusplus_init":
            centers = Multinomial.__kmeansplusplus( distributions, k, compute_distance )
        elif init_strategy == "vertices_init":
            centers = Multinomial.__vertices_init( distributions, k, compute_distance )
        else:
            centers = [ distributions[i] for i in np.random.randint( len(distributions), size=k ) ]
        prev_assign = [ 0 for _ in distributions ]

        for itr in range( 1, max_itrs+1 ):
            print("itr: ", itr)
            # re-assign all distributions to the centers
            assign      = []
            ##
            saved_all_distances = []
            ##
            for  _dist in distributions:
                distance = []
                for c in centers:
                    if c is None:
                        distance.append( np.inf )
                    else:
                        distance.append( compute_distance( _dist, c ) )
                assign.append( np.argmin( distance ) )
                ##
                saved_all_distances.append(np.asarray(distance))
                ##

            # check convergence
            if np.allclose( assign, prev_assign ):
                if verbose: print( 'clustering converged in %d iterations' % (itr+1) )
                break
            prev_assign = assign

            # re-compute the cluster centers
            centers = []
            for center_idx in range( k ):
                cluster = [ distributions[i] for i,idx in enumerate(assign) if np.isclose(idx, center_idx) ]
                if cluster:
                    centers.append( compute_center( cluster, max_itrs=max_center_itrs ) )
                else:
                    centers.append( None )
            
        return assign, saved_all_distances, centers

    # static methods
    #######################################################################

    def __init__( self, p=None, theta=None, eta=None ):
        '''
        the user should provide one of p/theta/eta to initialize

        p is a (possibly unnormalized) probability vector
        it can be a list or numpy 1D array

        theta is natural parameters

        eta is moment parameters
        '''
        #self.unchanged = p
        self.dtype = np.float
        self.eps = np.finfo( self.dtype ).eps

        if p is not None:
            p = np.array( p, dtype=self.dtype ).flatten()

        elif theta is not None:
            theta = np.array( theta, dtype=self.dtype ).flatten()
            p = np.hstack( [1, np.exp( theta )] )

        elif eta is not None:
            eta = np.array( eta, dtype=self.dtype ).flatten()
            eta0 = 1 - eta.sum()
            p = np.hstack( [eta0, eta] )

        else:
            raise RuntimeError( 'no way to initialize' )

        assert( np.all( p >= 0 ) )
        self.p = p / ( p.sum() + self.eps )
        
    def eta( self ):
        '''
        dual parameters
        '''
        return self.p[1:]

    def theta( self ):
        '''
        natural parameters
        '''
        return np.log( self.p[1:] / ( self.p[0] + self.eps ) + self.eps )

    def __str__( self ):
        return ' '.join( [ '%.4f' % f for f in self.p ] )

    def hilbert_cut( self, other, r ):
        '''
        On the line connecting self and other,
        find the point A so that 
        HD( self, A ) = r HD( self, other )

        hilbert_cut( self, other, 0 ) == self
        hilbert_cut( self, other, 1 ) == other
        '''
        assert( r >= 0 )
        assert( r <= 1 )

        if np.isclose( r, 0 ): return Multinomial( self.p )
        if np.isclose( r, 1 ): return Multinomial( other.p )
        if np.allclose( self.p, other.p ): return Multinomial( self.p )

        idx = np.logical_not( np.isclose( self.p, other.p ) )
        lamb = self.p[idx] / (self.p[idx] - other.p[idx])
        t0 = lamb[ lamb <= 0 ].max()
        t1 = lamb[ lamb >= 1 ].min()
        if np.isclose( t0, 0 ):
            # the cut is undefined
            return Multinomial( other.p )
        elif np.isclose( t1, 1 ):
            # the cut is undefined
            return Multinomial( self.p )
        else:
            _dist = np.abs( np.log( 1-1/t0 ) - np.log( 1-1/t1 ) ) * r
            _ed   = np.exp( _dist )
            x = t0 * t1 * (1-_ed) / ( t1 - t0*_ed )
            return Multinomial( (1-x) * self.p + x * other.p )

    def hilbert_distance( self, other ):
        '''
        Hilbert distance
        '''
        if np.allclose( self.p, other.p ): return 0

        idx = np.logical_not( np.isclose( self.p, other.p ) )
        if ( idx.sum() == 1 ): return 0

        lamb = self.p[idx] / (self.p[idx] - other.p[idx])
        t0 = lamb[ lamb <= 0 ].max()
        t1 = lamb[ lamb >= 1 ].min()
        if np.isclose( t0, 0 ) or np.isclose( t1, 1 ): return np.inf

        return np.abs( np.log( 1-1/t0 ) - np.log( 1-1/t1 ) )

