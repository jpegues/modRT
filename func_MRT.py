###FUNCTIONS FOR AC299R FINAL PROJECT
###By Jamila Pegues


###----------------------------------------------------------------
###SETUP
##Below Section: Imports necessary modules
import numpy as np
import scipy.ndimage as imger
import scipy.interpolate as inpoler
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull as huller
plt.close()
#




###----------------------------------------------------------------
###CLASSES
###------------------------------------
##INTERPOLATION CLASSES
##CLASS: Interper
##PURPOSE: Represents a base class for interpolation methods.
class Interper():
    ##METHOD: __init__
    ##PURPOSE: Initializes the interpolator.
    def __init__(self, valdict):
        #Record dictionary defining this object
        self._valdict = valdict
    #
    
    ##METHOD: get_value
    ##PURPOSE: Extracts a value for this interpolator, or throws an error if value is not present.
    def get_value(self, value):
        #Retrieve value
        try:
            return self._valdict[value]
        except KeyError:
            raise KeyError("Whoa!  Looks like {0} isn't stored for this class instance.  Check that it was included in the initial dictionary of this class instance during initialization.\nThese are the values stored for this class instance: {1}.".format(value, [key for key in self._valdict]))
    #
    
    ##METHOD: run_interper
    ##PURPOSE: Runs the requested interpolation scheme.
    def run_interper(self, known_xs, known_ys, known_values, new_xs, new_ys):
        pass
    #
#

##CLASS: Interper
##PURPOSE: Performs radial basis function interpolation methods.
class Interper_RBF(Interper):
    ##METHOD: run_interper
    ##PURPOSE: Runs interpolation using a specified radial basis function.
    def run_interper(self, known_xs, known_ys, known_values, new_xs, new_ys):
        #Extract necessary parameters for this interpolator
        verbose = self.get_value("verbose")
        form = self.get_value("form")
        maxpoints = self.get_value("max_inpoints")
        numcaps = self.get_value("num_cappoints")
        
        ##Below Section: Interpolates over and returns new points using RBF
        #Extract indices for selection of points to apply RBF
        if len(known_xs) < maxpoints: #Use all points if not too many
            kps = np.asarray([known_xs, known_ys]).T
            kvals = known_values
        else: #Use subset of points if too many (and thus too slow)
            #First extract all points that form boundary (convex hull)
            hullinds = huller(np.asarray([known_xs, known_ys]).T).vertices
            #Then extract 10 brightest points and 10 darkest points
            sortinds = np.argsort(known_values)
            brightinds = sortinds[::-1][0:numcaps]
            darkinds = sortinds[0:numcaps]
            #Combine these points, which must be included
            sureinds = np.unique(np.concatenate(
                                        (hullinds, brightinds, darkinds)))
            #Next remove boundary points from total selection of points
            leftoverinds = np.setdiff1d(
                                    np.arange(0,len(known_xs),1), sureinds)
            
            #Select subset grid of inner (non-hull) points
            if len(leftoverinds) > maxpoints: #If too many points in subset
                tmpinds = np.round(np.linspace(0, len(leftoverinds),
                                    maxpoints, endpoint=False)).astype(int)
            else: #If allowed to use all points in subset
                tmpinds = np.arange(0, len(leftoverinds), 1)
            
            #Concatenate hull and subset of inner points
            fininds = np.concatenate((sureinds, leftoverinds[tmpinds]))
            kps = np.asarray([known_xs[fininds], known_ys[fininds]]).T
            kvals = known_values[fininds]
        numpoints = kps.shape[0] #Number of points
        
        #Interpolate via RBF
        if len(kvals) == 1: #If single point to interpolate from
            vnew = np.ones(len(new_xs))*kvals #Spread single value
        else: #Otherwise, interpolate using RBF
            rbffunc = inpoler.Rbf(kps[:,0], kps[:,1],
                            np.ones(numpoints), kvals, function=form)
            vnew = rbffunc(new_xs, new_ys, np.ones(len(new_xs)))
        return {"values":vnew}
    #
#


###------------------------------------
##SAMPLING CLASSES
##CLASS: Sampler
##PURPOSE: Represents a base class for sampling methods.
class Sampler():
    ##METHOD: __init__
    ##PURPOSE: Initializes the sampler.
    def __init__(self, valdict):
        #Record dictionary defining this object
        self._valdict = valdict
    #
    
    ##METHOD: get_value
    ##PURPOSE: Extracts a value for this sampler, or throws an error if value is not present.
    def get_value(self, value):
        #Retrieve value
        try:
            return self._valdict[value]
        except KeyError:
            raise KeyError("Whoa!  Looks like {0} isn't stored for this class instance.  Check that it was included in the initial dictionary of this class instance during initialization.\nThese are the values stored for this class instance: {1}.".format(value, [key for key in self._valdict]))
    #
    
    ##METHOD: get_randorder
    ##PURPOSE: Returns a subset of the given indices in requested order.
    def get_randorder(self, inds):
        #Extract sampler parameters
        form = self.get_value("form")
        ninds = len(inds)
        
        #Determine number of samples to extract, based on sampler-type
        nsamps = self.get_numsamples(ninds)
        
        #In grid case, extract
        if form == "grid":
            inv = ninds // nsamps #Intervals between samples
            if inv <= 1: #If basically all samples are being sampled
                return inds
            #Otherwise, return equally-spaced samples
            subinds = inds[np.arange(0, ninds, inv)]
            return subinds
        #Else in random case, randomize given percentage of samples
        elif form == "random":
            subinds = np.random.shuffle(inds.copy())[0:nsamps]
            return subinds
    #
    
    ##METHOD: get_randinds
    ##PURPOSE: Returns an array of randomly selected indices from given range.
    def get_randinds(self, xmin, xmax, ymin, ymax, ndata):
        #Extract sampler parameters
        form = self.get_value("form")
        nsamps = self.get_numsamples(ndata) #Number of samples to extract
        gridl = int(np.floor(np.sqrt(nsamps))) #Num. points for grid length
        gridw = int(np.ceil(nsamps/1.0/gridl)) #Num. points for grid width
        
        #In grid case, extract at fixed intervals
        if form == "grid":
            xsampinds = np.tile(np.linspace(xmin, (xmax-1), gridl,
                        endpoint=True).astype(int), gridw) #Repeat-tile inds
            ysampinds = np.repeat(np.linspace(ymin, (ymax-1), gridw,
                        endpoint=True).astype(int), gridl) #Repeat-tile inds
        #Else in random case, randomize given percentage of samples
        elif form == "random":
            xsampinds = np.random.randint(low=xmin, high=xmax, size=nsamps)
            ysampinds = np.random.randint(low=ymin, high=ymax, size=nsamps)
        #Return random selection
        return {"y":ysampinds, "x":xsampinds}
    #
#

##CLASS: Sampler_Fraction
##PURPOSE: Samples a fixed faction of the given indices.
class Sampler_Fraction(Sampler):
    ##METHOD: get_numsamples
    ##PURPOSE: Determines the number of samples based on the given fraction.
    def get_numsamples(self, totnum):
        #Calculate number of samples
        frac = self.get_value("fraction")
        return int(np.floor(frac*totnum))
#


###------------------------------------
##BOUNDARY-FINDING CLASSES
##CLASS: GridFiller_Recurse
class GridFiller_Recurse():
    def __init__(self, origmatr, verbose, dofilter,
                    dosave=False, testname="test", sigma=None, cutoff=None):
        #Initialize and store grid matrices
        self.origmatr = origmatr
        self.verbose = verbose
        self.dosave = dosave
        self.newmatr = np.ones(shape=origmatr.shape)*np.nan
        self.count = 0 #To count recursions
        
        #Store original rectangular dimensions
        Dy, Dx = origmatr.shape
        self.Dx = Dx
        self.Dy = Dy
        
        #Run recursive algorithm
        self.run_gridfiller(Dx=Dx, Dy=Dy)
        if verbose:
            print("Number of recursive calls to grid-filling function: {0}."
                    .format(self.count))
        
        #Smooth resulting image (e.g., ids), if so desired
        if dofilter:
            unfiltmatr = self.newmatr.copy() #Copy of unfiltered version
            #Iterate through different ids
            uniqids = np.unique(self.newmatr)
            for ai in range(0, len(uniqids)):
                #Skip if current id is 0 (i.e., background)
                if uniqids[ai] == 0:
                    continue
                
                #Generate temporary binary matrix for this id
                tmpmatr = self.newmatr.copy()
                tmpmatr[tmpmatr != uniqids[ai]] = 0.0
                tmpmatr[tmpmatr == uniqids[ai]] = 1.0
                
                #Apply filter to the binary matrix
                tmpmatr = imger.gaussian_filter(tmpmatr, sigma=sigma)
                tmpmatr = tmpmatr*1.0/tmpmatr.max()
                tmpmatr[tmpmatr > cutoff] = 1.0
                tmpmatr[tmpmatr <= cutoff] = 0.0
                #Update actual id matrix with filtered id values
                self.newmatr[tmpmatr == 1] = uniqids[ai]
            
            #Plot tests, if so desired
            if verbose:
                fig = plt.figure()
                ax0 = fig.add_subplot(1, 2, 1)
                ax0.set_title("Unfiltered ids")
                plt.imshow(unfiltmatr, cmap=plt.cm.tab20, origin="lower")
                ax0 = fig.add_subplot(1, 2, 2)
                ax0.set_title("Filtered ids")
                plt.imshow(self.newmatr, cmap=plt.cm.tab20, origin="lower")
                if dosave:
                    plt.savefig(testname+"_id_filter.png")
                    plt.close()
                else:
                    plt.show()
    #
    
    ##METHOD: recurse_gridfiller
    def run_gridfiller(self, Dx, Dy):
        #Recurse with initial rectangular parameters
        self.recurse_gridfiller(x_1=0, x_2=Dx, dx=Dx, y_1=0, y_2=Dy, dy=Dy)
        #Show tests, if so desired
        if self.verbose:
            if self.dosave:
                plt.savefig(testname+"_id_recursion.png")
                plt.close()
            else:
                plt.show()
    #
    
    ##METHOD: recurse_gridfiller
    def recurse_gridfiller(self, x_1, x_2, dx, y_1, y_2, dy):
        self.count += 1
        #Plot boundaries as a check, if so desired
        if self.verbose:
            plt.scatter([x_1, x_1, x_2, x_2], [y_1, y_2, y_1, y_2],
                            alpha=0.1)
        
        #Exit if all nans in this rectangle
        if np.sum(~np.isnan(self.origmatr[y_1:y_2,x_1:x_2])) == 0:
            return
        
        #Check if rectangle has shrunk to pixel size
        if (dx == 1) and (dy == 1):
            #Store original sampled value, then exit
            self.newmatr[y_1,x_1] = self.origmatr[y_1,x_1]
            return
        
        #Check if current rectangle only has one value
        nanminhere = np.nanmin(self.origmatr[y_1:y_2,x_1:x_2])
        nanmaxhere = np.nanmax(self.origmatr[y_1:y_2,x_1:x_2])
        if nanminhere == nanmaxhere:
            #Fill rectangle with this value and exit
            self.newmatr[y_1:y_2,x_1:x_2] = nanmaxhere
            return
        
        #If not, then split the rectangle into four and recurse into them
        newdx = max([dx//2, 1]) #Half the length
        newdy = max([dy//2, 1]) #Half the width
        newx = x_1 + newdx
        newy = y_1 + newdy
        #If new rectangle has length and width > 1, split into 4
        if (newdx > 1) and (newdy > 1):
            #For top-left
            self.recurse_gridfiller(x_1=x_1, x_2=newx, dx=(newx - x_1),
                                    y_1=y_1, y_2=newy, dy=(newy - y_1))
            #For top-right
            self.recurse_gridfiller(x_1=newx, x_2=x_2, dx=(x_2 - newx),
                                    y_1=y_1, y_2=newy, dy=(newy - y_1))
            #For bottom-left
            self.recurse_gridfiller(x_1=x_1, x_2=newx, dx=(newx - x_1),
                                    y_1=newy, y_2=y_2, dy=(y_2 - newy))
            #For bottom-right
            self.recurse_gridfiller(x_1=newx, x_2=x_2, dx=(x_2 - newx),
                                    y_1=newy, y_2=y_2, dy=(y_2 - newy))
        #If new rectangle has L=W=1, split into 4
        elif (newdx == 1) and (newdy == 1):
            #For top-left
            self.recurse_gridfiller(x_1=x_1, x_2=x_1+1, dx=1,
                                    y_1=y_1, y_2=y_1+1, dy=1)
            #For top-right
            if (x_2 + 1) <= self.Dx: #If not past image edge
                self.recurse_gridfiller(x_1=x_1+1, x_2=x_2, dx=1,
                                    y_1=y_1, y_2=y_1+1, dy=1)
            #For bottom-left
            if (y_2 + 1) <= self.Dy: #If not past image edge
                self.recurse_gridfiller(x_1=x_1, x_2=x_1+1, dx=1,
                                    y_1=y_1+1, y_2=y_2, dy=1)
            #For bottom-right
            if ((x_2 + 1) <= self.Dx) and ((y_2 + 1) <= self.Dy):#If < edge
                self.recurse_gridfiller(x_1=x_1+1, x_2=x_2, dx=1,
                                    y_1=y_1+1, y_2=y_2, dy=1)
        #If new rectangle has length = 1, width > 1, and span>1, split into 2
        elif (newdy > 1) and (newdx == 1) and (dx > 1):
            #For top-left
            self.recurse_gridfiller(x_1=x_1, x_2=newx, dx=(newx - x_1),
                                    y_1=y_1, y_2=newy, dy=(newy - y_1))
            #For bottom-left
            self.recurse_gridfiller(x_1=x_1, x_2=newx, dx=(newx - x_1),
                                    y_1=newy, y_2=y_2, dy=(y_2 - newy))
        #If new rectangle has length = 1, width > 1, and span=1, split into 2
        elif (newdy > 1) and (newdx == 1) and (dx == 1):
            #For top-left
            self.recurse_gridfiller(x_1=x_1, x_2=x_2, dx=1,
                                    y_1=y_1, y_2=newy, dy=(newy - y_1))
            #For bottom-left
            self.recurse_gridfiller(x_1=x_1, x_2=x_2, dx=1,
                                    y_1=newy, y_2=y_2, dy=(y_2 - newy))
        #If new rectangle has length > 1, width = 1, and span>1, split into 2
        elif (newdx > 1) and (newdy == 1) and (dy > 1):
            #For bottom-left
            self.recurse_gridfiller(x_1=x_1, x_2=newx, dx=(newx - x_1),
                                    y_1=newy, y_2=y_2, dy=(y_2 - newy))
            #For bottom-right
            self.recurse_gridfiller(x_1=newx, x_2=x_2, dx=(x_2 - newx),
                                    y_1=newy, y_2=y_2, dy=(y_2 - newy))
        #If new rectangle has length > 1, width = 1, and span=1, split into 2
        elif (newdx > 1) and (newdy == 1) and (dy == 1):
            #For bottom-left
            self.recurse_gridfiller(x_1=x_1, x_2=newx, dx=(newx - x_1),
                                    y_1=y_1, y_2=y_2, dy=1)
            #For bottom-right
            self.recurse_gridfiller(x_1=newx, x_2=x_2, dx=(x_2 - newx),
                                    y_1=y_1, y_2=y_2, dy=1)
        else:
            raise ValueError("OH NO! Error! Contact your code provider!")
        
        #Ensure that all nans within total range have been filled
        if np.sum(~np.isnan(self.newmatr[y_1:y_2,x_1:x_2])) == 0:
            return
        elif np.sum(np.isnan(self.newmatr[y_1:y_2,x_1:x_2])) > 0:
            #Extract locations of known ids in this region
            tmpyinds, tmpxinds = np.where(
                            ~np.isnan(self.newmatr[y_1:y_2,x_1:x_2]))
            tmpyinds += y_1
            tmpxinds += x_1
            #Assign nans with ids closest to that corner
            #For top-left section
            if np.sum(np.isnan(self.newmatr[y_1:newy,x_1:newx])) > 0:
                #Find closest known point to top-left corner
                nearind = np.argmin(((tmpyinds - y_1)**2)
                                    + ((tmpxinds - x_1)**2))
                #Assign id of closest point to all nans in this region
                self.newmatr[y_1:newy,x_1:newx] = (
                        self.newmatr[tmpyinds[nearind],tmpxinds[nearind]])
            #For top-right section
            if np.sum(np.isnan(self.newmatr[y_1:newy,newx:x_2])) > 0:
                #Find closest known point to top-right corner
                nearind = np.argmin(((tmpyinds - y_1)**2)
                                    + ((x_2 - tmpxinds)**2))
                #Assign id of closest point to all nans in this region
                self.newmatr[y_1:newy,newx:x_2] = (
                        self.newmatr[tmpyinds[nearind],tmpxinds[nearind]])
            #For bottom-left section
            if np.sum(np.isnan(self.newmatr[newy:y_2,x_1:newx])) > 0:
                #Find closest known point to bottom-left corner
                nearind = np.argmin(((y_2 - tmpyinds)**2)
                                    + ((tmpxinds - x_1)**2))
                #Assign id of closest point to all nans in this region
                self.newmatr[newy:y_2,x_1:newx] = (
                        self.newmatr[tmpyinds[nearind],tmpxinds[nearind]])
            #For bottom-right section
            if np.sum(np.isnan(self.newmatr[newy:y_2,newx:x_2])) > 0:
                #Find closest known point to bottom-right corner
                nearind = np.argmin(((y_2 - tmpyinds)**2)
                                    + ((x_2 - tmpxinds)**2))
                #Assign id of closest point to all nans in this region
                self.newmatr[newy:y_2,newx:x_2] = (
                        self.newmatr[tmpyinds[nearind],tmpxinds[nearind]])
        #Raise error if nans still persist
        if np.sum(np.isnan(self.newmatr[y_1:y_2,x_1:x_2])) > 0:
            raise ValueError("NO! Still nans! Contact your code provider!")
        return
    #
#


###------------------------------------
##RENDERING AND TRACING CLASSES
##CLASS: Renderer
##PURPOSE: Represents a base class for basic rendering methods.
class Renderer():
    ##METHOD: __init__
    ##PURPOSE: Initializes the renderer.
    def __init__(self, scene, sampler, interper, dofilter=True,
                verbose=False, dosave=False, testname="test",
                sigma=10.0, cutoff=0.05):
        #Record rendering components
        self.verbose = verbose
        self.dosave = dosave
        self.sigma = sigma
        self.cutoff = cutoff
        self.testname = testname
        self.dofilter = dofilter
        self.scene = scene
        self.sampler = sampler
        self.interper = interper
        self.world = scene.world
        self.window = scene.window
        self.camera = scene.camera
        self.obj_list = scene.world.obj_list
        self.light_list = scene.world.light_list
        
        #Initialize grid to hold ray-tracing results
        self.grid = np.ones(shape=(scene.window.vspan, scene.window.hspan, 3))
        self.grid[:,:,:] = scene.world.backcolor #Initialize with background color
    #
    
    ##METHOD: reset_grid
    ##PURPOSE: Clears the grid to prepare for a new rendering.
    def reset_grid(self):
        #Override grid for holding ray-tracing results
        self.grid = np.ones(shape=(self.scene.window.vspan,
                                self.scene.window.hspan, 3))
        self.grid[:,:,:] = self.scene.world.backcolor #Init. with back. color
    #
    
    ##METHOD: finalize_grid
    ##PURPOSE: Finalize the grid (e.g., map to 0-1 range) and return the grid.
    def finalize_grid(self):
        #Finalize and return the grid
        self.grid[self.grid > 1] = 1.0
        self.grid[self.grid < 0] = 0.0
        return self.grid
    #
    
    ##METHOD: find_objpoint
    ##PURPOSE: Finds closest object-point of intersection along given ray, if exists.
    def find_objpoint(self, ray, objs=None, blockonly=False, fardist=None):
        #Extract values, if not given
        if objs is None:
            objs = self.obj_list
        #Initialize values
        tmin = np.inf #To hold closest scaling value
        tpoint = None #To hold closest point
        tobj = None #To reference closest object
        tid = None #To reference id of closest object
        
        #Iterate through objects
        for oi in range(0, len(objs)):
            resdict = objs[oi].find_intersection(ray=ray)
            #If no intersection, skip this object
            if resdict is None:
                continue
            
            #If checking for blocking objects, then check if obj-point between given points
            if blockonly:
                if (resdict["t"] > objs[oi].get_value("eps")) and (resdict["t"] < fardist):
                    return resdict["t"]
            #Otherwise, update closest intersection if necessary
            elif resdict["t"] < tmin:
                tmin = resdict["t"]
                tpoint = resdict["intpoint"]
                tobj = objs[oi]
                tid = objs[oi].get_value("id")
        
        #Return results for closest object-point
        if tpoint is None:
            return None
        return {"t":tmin, "intpoint":tpoint, "intobj":tobj, "intid":tid}
    #
    
    ##METHOD: trace_ray
    ##PURPOSE: Traces a given ray and returns desired quantity.
    def trace_ray(self, ray, form_shade, lights=None, objs=None):
        #Extract quantities useful for tracing this ray
        world = self.world
        if lights is None:
            lights = world.light_list
        if objs is None:
            objs = world.obj_list
        
        #Initialize holder of objects and total radiance at origin of ray
        irradiance = 0.0
        tpoint = None
        
        #Iterate through objects and determine point of closest intersection
        objid = 0 #Initiate id to hold intersected object(s)
        for li in range(0, len(lights)): #Iterate through lights
            resdict = self.find_objpoint(ray=ray, objs=objs)
            #If no intersections, then continue
            if resdict is None:
                continue
            
            #Otherwise, determine desired value for point of closest intersection
            objid = resdict["intid"]
            tpoint = resdict["intpoint"]
            tobj = resdict["intobj"]
            #Throw error if unrecognized form
            if form_shade not in ["bare", "matte"]:
                raise ValueError("Whoa! Unrecognized form ({0}) of ray-tracing requested!".format(form_shade))
            
            #For bare ray-tracing (no shading, no lighting)
            if form_shade == "bare":
                return {"color":tobj.get_value("color"), "irradiance":1.0,
                        "id":objid} #CHANGED FROM NONE TO 1.0 for irr.!!!
            
            #For matte (lighting+shading assuming no reflective surfaces)
            elif form_shade == "matte":
                #Determine light and surface parameters
                lcoord = lights[li].get_value("coord") #Coordinates of light
                surfnorm = tobj.get_path(point=tpoint) #Normal to surface
                lvec = lcoord - tpoint #Vector towards light
                ldist = np.linalg.norm(lvec) #Distance to light
                #Accumulate irradiance for shadowless lights
                if lights[li].get_value("iseverywhere"):
                    irradiance += lights[li].calc_irradiance(point=tpoint, path=surfnorm)
                    continue
                
                #Otherwise, determine if point is blocked or not first
                lray = Ray3D(coord=tpoint, path=lvec/ldist)
                blockres = self.find_objpoint(ray=lray, objs=objs, blockonly=True, fardist=ldist)
                #If object-point is not blocked, accumulate radiance
                if blockres is None:
                    irradiance += lights[li].calc_irradiance(point=tpoint, path=surfnorm)
                else:
                    objid += 0.5 #Indicates region on surface that's shadowed
                    continue
        
        #Return None if nothing intersected
        if tpoint is None:
            return None
        #Otherwise, return final color (based on accumulated irradiance)
        color_orig = tobj.get_color(point=tpoint)
        return {"color_new":(color_orig*irradiance),
                "irradiance":irradiance, "id":objid,
                "color_orig":color_orig}
    #
#

##CLASS: Renderer_Standard
##PURPOSE: Represents a standard/traditional ray-tracing renderer.
class Renderer_Standard(Renderer):
    ##METHOD: render_scene
    ##PURPOSE: Runs the renderer for this scene.
    def render_scene(self, form_shade):
        #Reset the grid
        self.reset_grid()
        
        #Extract scene components for this renderer
        scene = self.scene
        window = self.window
        camera = self.camera
        world = self.world
        grid = self.grid
        
        #Ray-trace through each pixel in the window
        hspan = window.hspan
        vspan = window.vspan
        wcoord = window.coord
        cwdist = wcoord[2] - camera.coord[2] #FIXED ALONG Z-AXIS FOR NOW; camera-window distance
        pixcoord = np.array([0, 0, cwdist]) #FIXED FOR AXIS-ALIGNED VIEWING
        irrgrid = np.zeros(shape=(vspan,hspan))
        pixray = Ray3D(coord=pixcoord, path=None)
        for vi in range(0, vspan):
            for hi in range(0, hspan):
                #Determine direction of current pixel
                pixpathraw = np.array([window.pixsize*(hi - 0.5*(hspan - 1.0)), window.pixsize*(vi - 0.5*(vspan - 1.0)), (-1*cwdist)])
                pixpath = pixpathraw/np.linalg.norm(pixpathraw)
                pixray.path = pixpath
                
                #Ray-trace for this pixel
                raydict = self.trace_ray(ray=pixray, form_shade=form_shade)
                if raydict is not None:
                    self.grid[vi,hi,:] = raydict["color_new"]
                    irrgrid[vi,hi] = raydict["irradiance"]
        
        #Finalize and return the grid
        self.irrgrid = irrgrid
        return {"image":self.finalize_grid(), "irr":irrgrid}
    #
#

##CLASS: Renderer_Interpolation
##PURPOSE: Represents a ray-tracing renderer that employs interpolation rather than tracing all rays.
class Renderer_Interpolation(Renderer):
    ##METHOD: render_scene
    ##PURPOSE: Runs the renderer for this scene.
    def render_scene(self, form_shade):
        ##Below Section: Prepares ray-tracing components and inputs
        sigma = self.sigma
        cutoff = self.cutoff
        dofilter = self.dofilter
        verbose = self.verbose
        dosave = self.dosave
        #Reset the grid
        self.reset_grid()
        
        #Extract scene components for this renderer
        scene = self.scene
        window = self.window
        camera = self.camera
        world = self.world
        testname = self.testname
        
        #Extract window components for this renderer
        hspan = window.hspan
        vspan = window.vspan
        wcoord = window.coord
        cwdist = wcoord[2] - camera.coord[2] #FIXED ALONG Z-AXIS FOR NOW; camera-window distance
        pixcoord = np.array([0, 0, cwdist]) #FIXED FOR AXIS-ALIGNED VIEWING
        pixray = Ray3D(coord=pixcoord, path=None)
        
        #Extract grids of values
        fingrid = self.grid
        irrgrid = np.ones(shape=(vspan, hspan))*np.nan
        idcoldict = {} #To hold colors associated with each id
        
        #Set sampler and interpolator
        sampler = self.sampler
        interper = self.interper
        
        ##Below Section: Ray-traces random subset of pixels
        #Extract subset of random samples
        sampindset = sampler.get_randinds(xmin=0, ymin=0, xmax=hspan,
                                            ymax=vspan, ndata=(hspan*vspan))
        ysampinds = sampindset["y"]
        xsampinds = sampindset["x"]
        nsamps = len(ysampinds)
        sampidarr = np.ones(nsamps)*(-1) #To hold intersecting object ids
        if verbose:
            print("Number of pixels: {0}.".format((hspan*vspan)))
            print("Number of samples: {0}.".format(nsamps))
        
        #Ray-trace for randomly selected indices
        for ai in range(0, nsamps):
            #Compute direction of current pixel
            hihere = xsampinds[ai]
            vihere = ysampinds[ai]
            pixpathraw = np.array([
                    window.pixsize*(hihere - 0.5*(hspan - 1.0)),
                    window.pixsize*(vihere - 0.5*(vspan - 1.0)),
                    (-1*cwdist)])
            pixpath = pixpathraw/np.linalg.norm(pixpathraw)
            pixray.path = pixpath
            
            #Ray-trace for this pixel
            raydict = self.trace_ray(ray=pixray, form_shade=form_shade)
            if raydict is not None:
                #Record pixel color and irradiance
                irrgrid[vihere,hihere] = raydict["irradiance"]
                fingrid[vihere,hihere,:] = raydict["color_new"]
                #Record pixel id and associated color
                idhere = raydict["id"]
                sampidarr[ai] = idhere
                if idhere not in idcoldict:
                    idcoldict[idhere] = raydict["color_orig"]
            else:
                #If no intersection, record background id (0)
                sampidarr[ai] = 0
                irrgrid[vihere,hihere] = 0
                fingrid[vihere,hihere,:] = (0*self.scene.world.backcolor)
                if 0 not in idcoldict:
                    idcoldict[0] = self.scene.world.backcolor
        #Raise error if some samples still don't have ids
        if np.min(sampidarr) < 0:
            raise ValueError("Whoa! Not all samples have proper ids!")
        
        ##Below Section: Assigns ids to unknown pixels via boundary divider
        #Prepare incomplete matrix of ids, containing known ids and nans
        idgrid_raw = np.ones(shape=irrgrid.shape)*np.nan
        idgrid_raw[ysampinds,xsampinds] = sampidarr
        
        #Fill in unknown pixels with guessed ids from boundary divider
        uniqids = np.unique(sampidarr)
        gridfiller = GridFiller_Recurse(origmatr=idgrid_raw,
                                    dofilter=dofilter, dosave=dosave,
                                    cutoff=cutoff, sigma=sigma,
                                    verbose=verbose, testname=testname)
        idgrid = gridfiller.newmatr #Full matrix where all pixels have ids
        #Test: Check samples and irradiance calculations so far
        if verbose:
            #Plot irradiance for samples
            plt.imshow(irrgrid, origin="lower")
            plt.colorbar()
            plt.title("Irradiance for sampled points.")
            if dosave:
                plt.savefig(testname+"_irr_sample.png")
                plt.close()
            else:
                plt.show()
            #Plot colors for samples
            plt.imshow(fingrid, origin="lower")
            plt.title("Colors for sampled points.")
            if dosave:
                plt.savefig(testname+"_color_sample.png")
                plt.close()
            else:
                plt.show()
            #Plot ids for samples
            plt.imshow(idgrid_raw, cmap=plt.cm.tab20, origin="lower")
            plt.colorbar()
            plt.title("Ids for sampled points.")
            if dosave:
                plt.savefig(testname+"_id_sample.png")
                plt.close()
            else:
                plt.show()
            #Plot ids for all pixels
            plt.imshow(idgrid, cmap=plt.cm.tab20, origin="lower")
            plt.colorbar()
            plt.title("Ids for all points.")
            if dosave:
                plt.savefig(testname+"_id_all.png")
                plt.close()
            else:
                plt.show()
        
        ##Below Section: Interpolates unknown pixels
        for ai in range(0, len(uniqids)):
            #Extract known pixels with current id
            knownyxindshere = np.where(((idgrid == uniqids[ai])
                                        & ~(np.isnan(irrgrid))))
            kxshere = knownyxindshere[1] #Known x-values
            kyshere = knownyxindshere[0] #Known y-values
            
            #Extract unknown pixels with current id (includes knowns)
            unknownyxindshere = np.where(((idgrid == uniqids[ai])))
                                        #& (np.isnan(irrgrid))))
            uxshere = unknownyxindshere[1] #Unknown x-values
            uyshere = unknownyxindshere[0] #Unknown y-values
            
            #If background pixels, then fill with background
            if uniqids[ai] == 0:
                irrgrid[uyshere,uxshere] = 0
                fingrid[uyshere,uxshere,:] = self.scene.world.backcolor
                continue
            
            #Interpolate unknown pixels with current id
            if len(kxshere) == 0:
                continue
            interpvals = interper.run_interper(known_xs=kxshere,
                            known_ys=kyshere, new_ys=uyshere, new_xs=uxshere,
                            known_values=irrgrid[kyshere, kxshere])["values"]
            irrgrid[uyshere,uxshere] = interpvals
            fingrid[uyshere,uxshere,:] = (
                            np.array([interpvals, interpvals, interpvals]).T
                            *idcoldict[uniqids[ai]])
        
        #Test: Check final irradiance patterns
        if verbose:
            #Plot irradiance for all pixels
            plt.imshow(irrgrid, origin="lower")
            plt.colorbar()
            plt.title("Irradiance for all points.")
            if dosave:
                plt.savefig(testname+"_irr_all.png")
                plt.close()
            else:
                plt.show()
        
        #Finalize and return the grid
        self.irrgrid = irrgrid
        return {"image":self.finalize_grid(), "irr":irrgrid}
    #
#


###------------------------------------
##WORLD AND VIEW CLASSES
##CLASS: Camera
##PURPOSE: Represents a camera in space.
class Camera():
    ##METHOD: __init__
    ##PURPOSE: Initializes the camera representation.
    def __init__(self, coord, angles):
        #Record all components of camera
        self.coord = coord
        self.angles = angles
    #
#

##CLASS: Scene
##PURPOSE: Represents a 3D 'scene' in space.
class Scene():
    ##METHOD: __init__
    ##PURPOSE: Initializes the 'scene' (e.g., world, camera, window).
    def __init__(self, world, camera, window):
        #Record all components of scene
        self.world = world
        self.camera = camera
        self.window = window
    #
#

##CLASS: Window
##PURPOSE: Represents a 2D 'window' in space (e.g., grid of pixels).
class Window():
    ##METHOD: __init__
    ##PURPOSE: Initializes the 'window' (e.g., grid of pixels).
    def __init__(self, hspan, vspan, coord, pixsize):
        #Record description of window
        self.coord = coord #Center of window
        self.hspan = hspan #Number of pixels in horizontal direction
        self.vspan = vspan #Number of pixels in vertical direction
        self.pixsize = pixsize #Size of pixels (length x width = s x s)
    #
#

##CLASS: World
##PURPOSE: Represents a 3D 'world' in space.
class World():
    ##METHOD: __init__
    ##PURPOSE: Initializes the 'world' (e.g., container for objects and lights).
    def __init__(self, obj_list, light_list, backcolor):
        #Record all objects and lights within this world
        self.obj_list = obj_list
        for ai in range(0, len(obj_list)): #Give objects incremental ids
            obj_list[ai].set_value(key="id", value=(ai+1))
        self.light_list = light_list
        self.backcolor = backcolor
    #
#


###------------------------------------
##COORDINATE CLASSES
##CLASS: Ray3D
##PURPOSE: Represents a 3D ray in space.
class Ray3D():
    ##METHOD: __init__
    ##PURPOSE: Initializes the 3D ray.
    def __init__(self, coord, path):
        #Record entire 3D coordinates (x,y,z) and path (direction)
        self.coord = coord
        self.path = path
        #Record individual components of coordinates
        self.x = coord[0]
        self.y = coord[1]
        self.z = coord[2]
    #
#


###------------------------------------
##LIGHT CLASSES
##CLASS: Light3D
##PURPOSE: Represents a 3D light in space.
class Light3D():
    ##METHOD: __init__
    ##PURPOSE: Initializes the 3D light.
    def __init__(self, valdict):
        #Record dictionary defining this object
        self._valdict = valdict
    #
    
    ##METHOD: get_value
    ##PURPOSE: Extracts a value for this light, or throws an error if value is not present.
    def get_value(self, value):
        #Retrieve value
        try:
            return self._valdict[value]
        except KeyError:
            raise KeyError("Whoa!  Looks like {0} isn't stored for this class instance.  Check that it was included in the initial dictionary of this class instance during initialization.\nThese are the values stored for this class instance: {1}.".format(value, [key for key in self._valdict]))
    #
    
    ##METHOD: calc_irradiance
    ##PURPOSE: Determines the irradiance from the light at a given point.
    def calc_irradiance(self, point, path):
        pass
    #
#

##CLASS: Light3D_Ambient
##PURPOSE: Represents 3D ambient light, which is constant everywhere in space.
class Light3D_Ambient(Light3D):
    ##METHOD: calc_irradiance
    ##PURPOSE: Determines the irradiance from the light at a given point.
    def calc_irradiance(self, point, path):
        scaler = self.get_value("scaler") #Scaling value for the strength of the light
        color = self.get_value("color_norm") #Normalized (RGB/255*scale) color of the light
        return (scaler*color)
    #
#

##CLASS: Light3D_Directional
##PURPOSE: Represents 3D directional light, which occurs only along one direction.
class Light3D_Directional(Light3D):
    ##METHOD: calc_irradiance
    ##PURPOSE: Determines the irradiance from the light at a given point.
    def calc_irradiance(self, point, path):
        scaler = self.get_value("scaler") #Scaling value for the strength of the light
        color = self.get_value("color_norm") #Normalized (RGB/255*scale) color of the light
        lpath = self.get_value("path") #Direction of directional light
        cosangle = np.dot(lpath, path)
        if cosangle > 0:
            return (scaler*color*cosangle)
        return 0
    #
#

##CLASS: Light3D_Point
##PURPOSE: Represents a 3D point light with natural distance attenuation.
class Light3D_Point(Light3D):
    ##METHOD: calc_irradiance
    ##PURPOSE: Determines the irradiance from the light at a given point.
    def calc_irradiance(self, point, path):
        scaler = self.get_value("scaler") #Scaling value for the strength of the light
        color = self.get_value("color_norm") #Normalized (RGB/255*scale) color of the light
        coord = self.get_value("coord") #Central location of the light
        ldist = np.linalg.norm(coord - point) #Distance to light
        lpathraw = coord - point #Direction (raw) from surface to light
        lpath = lpathraw/np.linalg.norm(1.0*lpathraw) #Direction (normalized) from surface to light
        cosangle = np.dot(lpath, path)
        if cosangle > 0:
            return (scaler*color*cosangle/(1.0*ldist*ldist))
        return 0
    #
#


###------------------------------------
##OBJECT CLASSES
##CLASS: Object3D
##PURPOSE: Represents a 3D object in space.
class Object3D():
    ##METHOD: __init__
    ##PURPOSE: Initializes the 3D object.
    def __init__(self, valdict):
        #Record dictionary defining this object
        self._valdict = valdict
    #
    
    ##METHOD: set_value
    ##PURPOSE: Set a value for this object.
    def set_value(self, key, value):
        self._valdict[key] = value
    
    ##METHOD: get_value
    ##PURPOSE: Extracts a value for this object, or throws an error if value is not present.
    def get_value(self, value):
        #Retrieve value
        try:
            return self._valdict[value]
        except KeyError:
            raise KeyError("Whoa!  Looks like {0} isn't stored for this class instance.  Check that it was included in the initial dictionary of this object during initialization.\nThese are the values stored for this class instance: {1}.".format(value, [key for key in self._valdict]))
    #
    
    ##METHOD: get_color
    ##PURPOSE: Extracts color at a given point.
    def get_color(self, point):
        #Retrieve color (assumed uniform in this base class)
        return self.get_value("color")
    #
    
    ##METHOD: get_path
    ##PURPOSE: Returns path (normal vector) to current point on surface.
    def get_normal(self, point):
        #Return normal vector
        pass
    #
    
    ##METHOD: find_intersection
    ##PURPOSE: Find nearest point on object surface that intersects given ray.  If no intersection, then None is returned.
    def find_intersection(self, ray):
        pass
    #
#

##CLASS: Plane3D
##PURPOSE: Represents a 3D plane in space.
class Plane3D(Object3D):
    ##METHOD: get_path
    ##PURPOSE: Returns path (normal vector) to current point on surface.
    def get_path(self, point):
        #Return normal vector
        return self.get_value("path")
    #
    
    ##METHOD: find_intersection
    ##PURPOSE: Find nearest point on plane surface that intersects given ray.  If no intersection, then None is returned.
    def find_intersection(self, ray):
        #Extract coordinates and path that define this plane
        coord = self.get_value("coord")
        path = self.get_value("path")
        eps = self.get_value("eps") #Avoids awkward singularities right at 0 during shading
        
        #Calculate scaling constant that defines intersection
        tval = np.dot((coord - ray.coord), path)/1.0/np.dot(ray.path, path)
        
        #If above 'singularity' threshold, return scaling constant and point of intersection
        if tval >= eps:
            return {"t":tval, "intpoint":(ray.coord + (tval*ray.path))}
        #Otherwise, return None
        return None
    #
#

##CLASS: Sphere3D
##PURPOSE: Represents a 3D sphere in space.
class Sphere3D(Object3D):
    ##METHOD: get_path
    ##PURPOSE: Returns path (normal vector) to current point on surface.
    def get_path(self, point):
        #Return normal vector
        pathraw = (point - self.get_value("coord"))
        return (pathraw/np.linalg.norm(1.0*pathraw))
    #
    
    ##METHOD: find_intersection
    ##PURPOSE: Find nearest point(s) on sphere surface that intersects given ray.  If no intersection, then None is returned.
    def find_intersection(self, ray):
        #Extract coordinates and radius that define this sphere
        coord = self.get_value("coord")
        rad = self.get_value("rad") #Radius
        eps = self.get_value("eps") #Avoids awkward singularities right at 0 during shading
        
        ##Below Section: Calculates scaling constant that defines intersection
        #Calculate quadratic constants
        aval = np.dot(ray.path, ray.path)
        bval = 2*np.dot((ray.coord - coord), ray.path)
        cval = np.dot((ray.coord - coord), (ray.coord - coord)) - (rad*rad)
        inpart = (bval*bval) - (4*aval*cval)
        #Throw out this case if no intersection
        if inpart < 0:
            return None
        
        #Calculate roots of quadratic equation for sphere
        sqrtpart = np.sqrt(inpart)
        botpart = 2.0 * aval
        
        #Calculate smaller root first
        tval = (-bval - sqrtpart)/botpart
        #If above 'singularity' threshold, return scaling constant and point of intersection
        if tval >= eps:
            return {"t":tval, "intpoint":(ray.coord + (tval*ray.path))}
        
        #If necessary, calculate larger root next
        tval = (-bval + sqrtpart)/botpart
        #If above 'singularity' threshold, return scaling constant and point of intersection
        if tval >= eps:
            return {"t":tval, "intpoint":(ray.coord + (tval*ray.path))}
        
        #Otherwise, return None
        return None
    #
#








