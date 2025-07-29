import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
import pandas as pd

def setupGeometry(csvPath, gridResolution=200, padding=0.1):
    """
    Setup computational domain from boundary points
    
    Args:
        csvPath: Path to CSV file containing boundary points
        gridResolution: Number of grid points in each dimension
        padding: Extra space around geometry
    
    Returns:
        X, Y: Meshgrid coordinates
        interiorMask: Boolean mask for interior points
        boundaryMask: Boolean mask for boundary points
        T: Temperature array with boundary values
        boundaryPoints: Original boundary points
        boundaryTemps: Boundary temperatures
        hx, hy: Grid spacing in x and y directions
    """
    # Read boundary data from CSV
    data = pd.read_csv(csvPath)
    boundaryPoints = data[['x', 'y']].values
    boundaryTemps = data['T'].values
    
    # Create kd-tree for efficient nearest neighbor search
    boundaryTree = cKDTree(boundaryPoints)
    
    # Calculate domain bounds with padding
    xMin, xMax = boundaryPoints[:, 0].min(), boundaryPoints[:, 0].max()
    yMin, yMax = boundaryPoints[:, 1].min(), boundaryPoints[:, 1].max()
    xMin -= padding; xMax += padding
    yMin -= padding; yMax += padding
    
    # Create computational grid
    x = np.linspace(xMin, xMax, gridResolution)
    y = np.linspace(yMin, yMax, gridResolution)
    X, Y = np.meshgrid(x, y)
    hx = x[1] - x[0]
    hy = y[1] - y[0]
    
    # Identify points inside the boundary contour
    contourPath = mpath.Path(boundaryPoints)
    gridPoints = np.vstack((X.ravel(), Y.ravel())).T
    inside = contourPath.contains_points(gridPoints).reshape(X.shape)
    
    # Identify boundary points (inside points with exterior neighbors)
    boundaryMask = np.zeros_like(inside, dtype=bool)
    neighbors = np.array([(-1,-1), (-1,0), (-1,1), (0,-1), 
                         (0,1), (1,-1), (1,0), (1,1)])
    
    for i in range(1, gridResolution-1):
        for j in range(1, gridResolution-1):
            if inside[i, j]:
                # Check all 8 neighbors
                for di, dj in neighbors:
                    ni, nj = i + di, j + dj
                    if not inside[ni, nj]:
                        boundaryMask[i, j] = True
                        break
    
    interiorMask = inside & ~boundaryMask
    
    # Initialize temperature array
    T = np.full_like(X, np.nan)
    T[~inside] = np.nan
    
    # Assign boundary temperatures using kd-tree
    boundaryGridPoints = np.column_stack((X[boundaryMask], Y[boundaryMask]))
    _, indices = boundaryTree.query(boundaryGridPoints)
    T[boundaryMask] = boundaryTemps[indices]
    
    return X, Y, interiorMask, boundaryMask, T, boundaryPoints, boundaryTemps, hx, hy

def solveHeatEquation(X, Y, interiorMask, boundaryMask, T, kxx, kxy, kyy, hx, hy):
    """
    Solve anisotropic heat equation using finite difference method
    
    Args:
        X, Y: Meshgrid coordinates
        interiorMask: Boolean mask for interior points
        boundaryMask: Boolean mask for boundary points
        T: Temperature array with boundary values
        kxx, kxy, kyy: Thermal conductivity coefficients
        hx, hy: Grid spacing in x and y directions
    
    Returns:
        T: Solved temperature field
    """
    ny, nx = X.shape
    
    # Discretization coefficients
    cxx = kxx / hx**2
    cyy = kyy / hy**2
    cxy = kxy / (4 * hx * hy)

    # Precompute neighbor offsets
    mainOffsets = [(-1, 0, cxx), (1, 0, cxx), 
                   (0, -1, cyy), (0, 1, cyy)]
    crossOffsets = [(1, 1, cxy), (1, -1, -cxy), 
                   (-1, 1, -cxy), (-1, -1, cxy)]
    
    # Create mapping for interior points
    nInterior = np.sum(interiorMask)
    idxMap = np.full((ny, nx), -1)
    idxMap[interiorMask] = np.arange(nInterior)
    
    # Initialize sparse system
    A = lil_matrix((nInterior, nInterior))
    b = np.zeros(nInterior)
    
    # Build linear system
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            if interiorMask[i, j]:
                idx = idxMap[i, j]
                
                # Main diagonal coefficient
                A[idx, idx] = -2 * (cxx + cyy)
                
                # Standard derivatives (x and y directions)
                for di, dj, coef in mainOffsets:
                    ni, nj = i + di, j + dj
                    if interiorMask[ni, nj]:
                        col = idxMap[ni, nj]
                        A[idx, col] = coef
                    else:
                        b[idx] -= coef * T[ni, nj]
                
                # Mixed derivative term
                for di, dj, coef in crossOffsets:
                    ni, nj = i + di, j + dj
                    if interiorMask[ni, nj]:
                        col = idxMap[ni, nj]
                        A[idx, col] = coef
                    else:
                        b[idx] -= coef * T[ni, nj]
    
    # Solve linear system
    A_csr = A.tocsr()
    solution = spsolve(A_csr, b)
    T[interiorMask] = solution
    
    return T

def visualizeResults(X, Y, T, boundaryPoints, boundaryTemps):
    """
    Visualize temperature distribution with true 1:1 scale
    
    Args:
        X, Y: Meshgrid coordinates
        T: Temperature field
        boundaryPoints: Boundary point coordinates
        boundaryTemps: Boundary temperatures
    """
    # Create figure with proper aspect ratio
    fig = plt.figure(figsize=(10, 8), dpi=100)
    ax = fig.add_subplot(111)
    
    # Calculate proper aspect ratio based on data range
    x_range = X.max() - X.min()
    y_range = Y.max() - Y.min()
    aspect_ratio = y_range / x_range
    
    # Set figure size proportional to data dimensions
    fig.set_size_inches(10, 10 * aspect_ratio)
    
    # Create filled contour plot
    contour = ax.contourf(X, Y, T, levels=50, cmap='jet')
    cbar = fig.colorbar(contour, ax=ax, label='Temperature (°C)')
    cbar.formatter.set_powerlimits((0, 0))
    
    # Plot boundary contour and points
    ax.plot(boundaryPoints[:,0], boundaryPoints[:,1], 
            'k-', linewidth=1.5, label='Boundary')
    ax.scatter(boundaryPoints[:,0], boundaryPoints[:,1], 
               c=boundaryTemps, cmap='jet', s=30, 
               edgecolors='k', label='Boundary Points',
               zorder=3)  # Ensure points are on top
    
    # Set true 1:1 scale
    ax.set_aspect('equal', adjustable='datalim')
    
    # Format plot for scientific presentation
    ax.set_title('Dir:0 - Res:500', fontsize=14)
    ax.set_xlabel('X(m)', fontsize=12)
    ax.set_ylabel('Y(m)', fontsize=12)
    ax.grid(alpha=0.2, linestyle='--')
    ax.legend(fontsize=10, loc='upper right')
    
    # Add scale bar for reference
    scale_length = 0.1 * x_range
    ax.plot([X.min() + 0.05*x_range, X.min() + 0.05*x_range + scale_length],
            [Y.min() + 0.05*y_range, Y.min() + 0.05*y_range],
            'w-', linewidth=3, zorder=4)
    ax.text(X.min() + 0.05*x_range + 0.5*scale_length, 
            Y.min() + 0.06*y_range,
            f'{scale_length:.2f} m', 
            color='white', ha='center', fontsize=10,
            bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))
    
    plt.tight_layout()
    
    # Save high-quality figure
    plt.savefig('true_scale_temperature.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main execution routine"""
    # Physical parameters
    KXX, KXY, KYY = 6, 0.0, 0.58  # Anisotropic conductivity
    GRID_RESOLUTION = 500
    
    # Setup computational domain
    geoResults = setupGeometry('coordinate.csv', GRID_RESOLUTION)
    X, Y, interiorMask, boundaryMask, T, bPoints, bTemps, hx, hy = geoResults
    
    # Solve heat equation
    T = solveHeatEquation(X, Y, interiorMask, boundaryMask, T, 
                         KXX, KXY, KYY, hx, hy)
    
    # Visualize results
    visualizeResults(X, Y, T, bPoints, bTemps)

if __name__ == "__main__":
    main()
