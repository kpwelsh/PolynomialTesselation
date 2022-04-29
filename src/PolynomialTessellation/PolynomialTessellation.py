import numpy as np
import itertools
from collections import defaultdict
from regex import P
from scipy.spatial import Voronoi
import scipy


class OneDTessellation:
    def __init__(self, mu, ratio = None, c = None):
        if (ratio is None) == (c is None):
            raise ValueError(f'Provide only one of \'ratio\' or \'c\'. ratio: {ratio}, c: {c}')
        if c is not None:
            self.Ratio = c / (1 + c)
        elif ratio is not None:
            self.Ratio = ratio
        self.C = self.Ratio / (1 - self.Ratio)
        self.Points = mu.reshape((-1,))
        
    def __call__(self, x):
        i = np.argmin(np.abs(self.Points - x))
        p = self.Points[i]
        if x - p >= 0:
            l = p
            if i < len(self.Points) - 1:
                r = self.Points[i+1]
            else:
                r = None
        else:
            r = p
            if i > 0:
                l = self.Points[i-1]
            else:
                l = None
        return self.eval_region(x, l, r)
    
    def eval_region(self, x, l, r):
        if r is None:
            return (x - l)**2, x - l, 2
        elif l is None:
            return (x - r)**2, x - r, 2
        
        m = (l + r) / 2
        d = r - l
        if abs(x - m) <= d * (1 - self.Ratio) / 2:
            return (d / 2)**2 * (self.Ratio**2 * (1 + self.C) + self.C * (1 - 2 * self.Ratio))  - self.C * (x - m)**2, -self.C * (x - m), self.C
        
        if abs(x - l) <= abs(x - r):
            return (x - l)**2, x - l, 2
        else:
            return (x - r)**2, x - r, 2
        
        

class PolynomialTessellation:
    def __init__(self, mu, ratio = None, c = None, max_dist = 20):
        if (ratio is None) == (c is None):
            raise ValueError(f'Provide only one of \'ratio\' or \'c\'. ratio: {ratio}, c: {c}')
        if c is not None:
            self.Ratio = c / (1 + c)
        elif ratio is not None:
            self.Ratio = ratio
        self.InitialCount = len(mu)
        boundaries = []
        for combo in itertools.product(*(((-1.,1.),) * len(mu[0]))):
            boundaries.append(np.array(combo) * max_dist)
        
        mu = np.concatenate((mu, boundaries))
        self.Vor = Voronoi(mu)
        self.Point_to_point = defaultdict(set)
        self.Vertex_to_point = defaultdict(set)
        self.Point_to_vertex = defaultdict(set)
        self.Ridge_to_vertex = []
        self.Point_to_ridge = defaultdict(set)
        self.Point_point_to_ridge = {}
        for r, (rps, rvs) in enumerate(zip(self.Vor.ridge_points, self.Vor.ridge_vertices)):
            self.Ridge_to_vertex.append(set(v for v in rvs if v != -1))
            for i in rps:
                if r not in self.Point_to_ridge[i]:
                    self.Point_to_ridge[i].add(r)
                self.Point_point_to_ridge[frozenset(rps)] = r
                self.Point_to_point[i] |= set(j for j in rps if j != i)
                for j in rvs:
                    if j > -1 and i not in self.Vertex_to_point[j]:
                        self.Vertex_to_point[j].add(i)
                    if j > -1 and j not in self.Point_to_vertex[i]:
                        self.Point_to_vertex[i].add(j)

        self.PointBoundaries = []
        for i, p in enumerate(self.Vor.points):
            A = []
            b = []
            adj = []
            for a in self.Point_to_point[i]:
                n = self.Vor.points[a] - p
                r = np.linalg.norm(n) / 2
                n /= 2*r
                A.append(n)
                b.append(r)
                adj.append(a)
            A = np.array(A)
            b = np.array(b)
            self.PointBoundaries.append((i,A,b,adj))

    def neighbor_regions(self, point_combo, dimension):
        if len(point_combo) > 1:
            for combo in itertools.combinations(point_combo, len(point_combo) - 1):
                yield combo
        if len(point_combo) <= dimension:
            possible_additions = set.intersection(*(self.Point_to_point[p] for p in point_combo))
            for a in possible_additions:
                yield (*point_combo, a)
        return None
    
    def new_direction(self, region, new_p):
        points = np.array([self.Vor.points[p] for p in region]).reshape((len(region), self.Vor.points.shape[1]))
        next = self.Vor.points[new_p]

        next = next - points[0,:]
        points = points - points[0,:]
        directions = []
        for new_d in points[1:,:]:
            for d in directions:
                new_d -= new_d @ d * d
            new_d /= np.linalg.norm(new_d)
            directions.append(new_d)
        
        for d in directions:
            next -= next @ d * d
        next /= np.linalg.norm(next)
        return next

    def get_verts(self, region, f0 = False):
        pvs = itertools.product(region, set.intersection(*(self.Point_to_vertex[p] for p in region)))
        ps, vs = list(zip(*pvs))
        ps = list(ps)
        vs = list(vs)

        xs = self.Vor.points[ps] + self.Ratio * (self.Vor.vertices[vs] - self.Vor.points[ps])
        if not f0:
            return xs
        
        return xs, -(xs[0]-self.Vor.points[ps[0]]) @ (xs[0]-self.Vor.points[ps[0]])
    
    def should_move(self, region, neighbor, x, region_verts):
        region = set(region)
        neighbor = set(neighbor)
        new_point = list(region ^ neighbor)
        if len(new_point) != 1:
            raise ValueError('The new region must only have one point intersection with the current region')
        n = self.new_direction(region & neighbor, new_point[0])

        vals = region_verts @ n
        xn = n.dot(x)
        l, r = np.min(vals), np.max(vals)

        nn = 0
        for v in set.intersection(*(self.Point_to_vertex[p] for p in neighbor)):
            for p in neighbor:
                nn = n @ (self.Vor.points[p] + self.Ratio * (self.Vor.vertices[v] - self.Vor.points[p]))
                if nn > r:
                    return xn > r
                elif nn < l:
                    return xn < l
        
    def contains(self, region, x):
        if len(region) == 1:
            p = region[0]
            _, A, b, _ = self.PointBoundaries[p]
            return np.all(A @ (x - self.Vor.points[p]) <= b*self.Ratio)
        return False

    def __call__(self, x):
        p_i = np.argmin(np.linalg.norm(self.Vor.points - x, axis = 1))
        best_region = (p_i,)
        while True:
            if self.contains(best_region, x):
                return self.eval_region(best_region, x)

            region_verts = self.get_verts(best_region)

            for neighbor in self.neighbor_regions(best_region, len(x)):
                if self.should_move(best_region, neighbor, x, region_verts):
                    best_region = neighbor
                    break
            else:
                return self.eval_region(best_region, x)

    def eval_region(self, region, x):
        if len(region) == 1:
            p = region[0]
            point = self.Vor.points[p]
            return -(x - point) @ (x - point), - 2 * (x - point), 2 * np.eye(len(x))
        
        xs, f0 = self.get_verts(region, True)
        
        points = np.array([self.Vor.points[p] for p in region])
        A, b, c, p = fast_q_fit(points, self.Ratio, np.mean(xs, axis = 0), xs[0], f0)

        return (x - p) @ A @ (x - p) + b @ (x - p) + c, 2 * A @ (x - p) + b, 2 * A

def fast_q_fit(points, ratio, mean_v, rx, rf):
    p = np.mean(points, axis = 0)
    points = points[1:] - points[0]
    D = points.T
    
    orth = scipy.linalg.orth(D, rcond = None)
    null = scipy.linalg.null_space(D.T, rcond = None)

    s = np.diag([ratio / (1-ratio)] * len(orth.T) + [-1] * len(null.T))
    V = np.column_stack((orth, null))
    A = V @ s @ V.T

    b = -2*(mean_v - p) - 2 * A @ (mean_v - p)

    c = rf - ((rx - p) @ A @ (rx - p) + b @ (rx - p))

    return A, b, c, p

def show():
    import matplotlib.pyplot as plt
    mu = np.random.random((5, 2))
    mu = np.column_stack((mu, np.zeros(mu.shape[0])))
    tess = PolynomialTessellation(mu, 0.5)


    h = 0.025

    x = np.arange(0,1,h)
    y = np.arange(0,1,h)
    X, Y = np.meshgrid(x,y)
    XY = np.column_stack((X.flatten(), Y.flatten(), np.zeros_like(X.flatten())))

    f = []
    gds = []
    for xy in XY:
        v, gd, _ = tess(xy)
        f.append(v)
        gds.append(gd/2)
    f = np.array(f).reshape(X.shape)
    gds = np.array(gds)


    fig = plt.figure(figsize= (14, 4))
    a1 = fig.add_subplot(1, 3, 1, projection = '3d')
    a2 = fig.add_subplot(1, 3, 2, projection = '3d')
    a3 = fig.add_subplot(1, 3, 3, projection = '3d')
    a1.plot_wireframe(X, Y, f)
    a1.set_title('f')

    dfdx = (f[2:,1:-1] - f[:-2,1:-1]) / (2*h)
    dfdy = (f.T[2:,1:-1] - f.T[:-2,1:-1]) / (2*h)
    
    a2.plot_wireframe(X[1:-1,1:-1], Y[1:-1,1:-1], dfdx)
    a2.set_title(r'$\nabla_x f$')

    a3.plot_wireframe(X[1:-1,1:-1], Y[1:-1,1:-1], dfdy)
    a3.set_title(r'$\nabla_y f$')

    plt.show()

def perf_test():
    from cProfile import Profile
    from pstats import Stats
    d = 6
    mu = np.random.random((100, d))
    tess = PolynomialTessellation(mu, 0.5)

    p = Profile()
    p.enable()
    for _ in range(100):
        tess(np.random.random(d) * 2 - 1)
    p.disable()
    Stats(p).sort_stats('cumtime').print_stats()

if __name__ == '__main__':

    np.random.seed(1)
    #perf_test()
    show()

