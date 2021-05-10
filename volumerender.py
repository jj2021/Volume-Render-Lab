import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import concurrent.futures
import math 
from scipy.interpolate import interpn

"""
Create Your Own Volume Rendering (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate the Schrodinger-Poisson system with the Spectral method
"""

def rayTransfer(voxels):
    r,g,b,a = [0,0,0,0]
    for x in voxels:
        rn = 1.0*math.exp( -(x - 9.0)**2/1.0 ) +  0.1*math.exp( -(x - 3.0)**2/0.1 ) +  0.1*math.exp( -(x - -3.0)**2/0.5 )
        gn = 1.0*math.exp( -(x - 9.0)**2/1.0 ) +  1.0*math.exp( -(x - 3.0)**2/0.1 ) +  0.1*math.exp( -(x - -3.0)**2/0.5 )
        bn = 0.1*math.exp( -(x - 9.0)**2/1.0 ) +  0.1*math.exp( -(x - 3.0)**2/0.1 ) +  1.0*math.exp( -(x - -3.0)**2/0.5 )
        a  = 0.6*math.exp( -(x - 9.0)**2/1.0 ) +  0.1*math.exp( -(x - 3.0)**2/0.1 ) + 0.01*math.exp( -(x - -3.0)**2/0.5 )

        #Alpha compositing
        r = a*rn + (1-a)*r
        g = a*gn + (1-a)*g
        b = a*bn + (1-a)*b
   
    a = 1
    return r,g,b,a

def maxIntensity(voxels):
    maximum = 0
    for x in voxels:
        if x > maximum:
            maximum = x

    r = 1.0*math.exp( -(maximum - 9.0)**2/1.0 ) +  0.1*math.exp( -(maximum - 3.0)**2/0.1 ) +  0.1*math.exp( -(maximum - -3.0)**2/0.5 )
    g = 1.0*math.exp( -(maximum - 9.0)**2/1.0 ) +  1.0*math.exp( -(maximum - 3.0)**2/0.1 ) +  0.1*math.exp( -(maximum - -3.0)**2/0.5 )
    b = 0.1*math.exp( -(maximum - 9.0)**2/1.0 ) +  0.1*math.exp( -(maximum - 3.0)**2/0.1 ) +  1.0*math.exp( -(maximum - -3.0)**2/0.5 )
    a = 1
    return r,g,b,a

def avg(voxels):
    r,g,b,a = [0,0,0,0]
    total = np.sum(voxels)

    average = total/voxels.size

    r = 1.0*math.exp( -(average - 9.0)**2/1.0 ) +  0.1*math.exp( -(average - 3.0)**2/0.1 ) +  0.1*math.exp( -(average - -3.0)**2/0.5 )
    g = 1.0*math.exp( -(average - 9.0)**2/1.0 ) +  1.0*math.exp( -(average - 3.0)**2/0.1 ) +  0.1*math.exp( -(average - -3.0)**2/0.5 )
    b = 0.1*math.exp( -(average - 9.0)**2/1.0 ) +  0.1*math.exp( -(average - 3.0)**2/0.1 ) +  1.0*math.exp( -(average - -3.0)**2/0.5 )
    a = 1
    return r,g,b,a

def cast(voxels, method):
    switcher = {
        "opacity": rayTransfer,
        "max": maxIntensity,
        "average": avg,
    }

    func = switcher.get(method, rayTransfer)
    return func(voxels)

def main():
    """ Volume Rendering """
    
    # Load Datacube
    f = h5.File('datacube.hdf5', 'r')
    datacube = np.array(f['density'])
    
    # Datacube Grid
    Nx, Ny, Nz = datacube.shape
    x = np.linspace(-Nx/2, Nx/2, Nx)
    y = np.linspace(-Ny/2, Ny/2, Ny)
    z = np.linspace(-Nz/2, Nz/2, Nz)
    points = (x, y, z)
    
    # Do Volume Rendering at Different Veiwing Angles
    Nangles = 1
    for i in range(Nangles):
        
        print('Rendering Scene ' + str(i+1) + ' of ' + str(Nangles) + '.\n')
    
        # Camera Grid / Query Points -- rotate camera view
        angle = np.pi/2 * i / Nangles
        N = 180
        c = np.linspace(-N/2, N/2, N)
        qx, qy, qz = np.meshgrid(c,c,c)
        qxR = qx
        qyR = qy * np.cos(angle) - qz * np.sin(angle) 
        qzR = qy * np.sin(angle) + qz * np.cos(angle)
        qi = np.array([qxR.ravel(), qyR.ravel(), qzR.ravel()]).T
        
        # Interpolate onto Camera Grid
        camera_grid = interpn(points, datacube, qi, method='linear').reshape((N,N,N))
        
        # Do Volume Rendering
        image = np.zeros((camera_grid.shape[1],camera_grid.shape[2],3))

        # Run all ray casts in parallel
        with concurrent.futures.ThreadPoolExecutor (max_workers=2) as executor:
             for i in range(image.shape[0]):
                 for j in range(image.shape[1]):
                    future = executor.submit(cast,camera_grid[i,j,:],"max")
                    r = future.result()[0]
                    g = future.result()[1]
                    b = future.result()[2]
                    a = future.result()[3]

                    image[i,j,0] = r
                    image[i,j,1] = g
                    image[i,j,2] = b

        
        image = np.clip(image,0.0,1.0)
        
        # Plot Volume Rendering
        plt.figure(figsize=(4,4), dpi=80)
        
        plt.imshow(image)
        plt.axis('off')
        
        # Save figure
        plt.savefig('volumerender' + str(i) + '.png',dpi=240,  bbox_inches='tight', pad_inches = 0)
    
    
    
    # Plot Simple Projection -- for Comparison
    plt.figure(figsize=(4,4), dpi=80)
    
    plt.imshow(np.log(np.mean(datacube,0)), cmap = 'viridis')
    plt.clim(-5, 5)
    plt.axis('off')
    
    # Save figure
    plt.savefig('projection.png',dpi=240,  bbox_inches='tight', pad_inches = 0)
    plt.show()
    

    return 0
    


  
if __name__== "__main__":
  main()
