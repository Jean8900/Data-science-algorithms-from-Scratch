# Convert a list of points to box format
# Input: [[x1,y1], [x2,y2], ...] -> Output: [[x1,x2,...], [y1,y2,...]]
def list_point_to_box(L):
    return [[L[i][0] for i in range(len(L))],[L[i][1] for i in range(len(L))]]

# Convert box format to a list of points
# Input: [[x1,x2,...], [y1,y2,...]] -> Output: [[x1,y1], [x2,y2], ...]
def box_to_list_point(box):
    return [[box[0][i],box[1][i]] for i in range(len(box[0]))]

# Find the intersection point of two lines defined by slope and intercept (y = ax + b)
# D1 and D2 are tuples (a, b) representing lines
# Returns None if lines are parallel, otherwise returns (x, y) intersection point
def intersection_2_lines(D1,D2):
    a1,b1=D1
    a2,b2=D2
    if a1==a2:
        return None
    else:
        return (b2-b1)/(a1-a2),(b2-b1)/(a1-a2)*a1+b1

# Calculate the line equation (y = ax + b) passing through two points
# Returns (a, b) coefficients or None if the line is vertical or points are identical
def line_btw_2_points(P1,P2):
    x1,y1=P1
    x2,y2=P2
    if x1==x2:
        if y1==y2:
            return None
        return None
    return (y1-y2)/(x1-x2),-(y1-y2)/(x1-x2)*x2+y2

# Calculate the perpendicular bisector of the segment between two points
# Returns the line equation (a, b) of the perpendicular line passing through the midpoint
# Returns None if the original line is horizontal
def line_perpendicular_2_points(P1,P2):
    x1,y1=P1
    x2,y2=P2
    a,b=line_btw_2_points(P1,P2)
    xm,ym=(x1+x2)/2,(y1+y2)/2
    if a==0:
        return None
    return -1/a,ym+xm/a
    # cas optimal

# Calculate the centroid (center of mass) of a box of points
# Input: box format [[x1,x2,...], [y1,y2,...]]
# Returns [x_mean, y_mean]
def centroïd(box):
    n=len(box[0])
    return [sum(box[0])/n,sum(box[1])/n]

# Convert a segment from point format to box format
# Input: [[x1,y1],[x2,y2]] -> Output: [[x1,x2],[y1,y2]]
def segment_x2box(S):
    x1,y1=S[0]
    x2,y2=S[1]
    return [[x1,x2],[y1,y2]]

# Convert a segment from box format to point format
# Input: [[x1,x2],[y1,y2]] -> Output: [[x1,y1],[x2,y2]]
def x2box_segment(box):
    x1,x2=box[0]
    y1,y2=box[1]
    return [[x1,y1],[x2,y2]]

# Find the intersection point of two line segments
# S1 and S2 are segments defined by two points each: [[x1,y1],[x2,y2]]
# Returns [x,y] if segments intersect, None otherwise
def intersection_2_segments(S1,S2):
    # collecting the x & y coordinates of the two points per segment
    x1,y1=S1[0]
    x2,y2=S1[1]
    x3,y3=S2[0]
    x4,y4=S2[1]
    # two points
    PS1,QS1=[x1,y1],[x2,y2]
    PS2,QS2=[x3,y3],[x4,y4]

    # making lines from segments
    D1,D2=line_btw_2_points(PS1,QS1),line_btw_2_points(PS2,QS2)

    # manage exception when first segment is vertical
    if type(D1) ==type(None) and not type(D2) ==type(None):
        if max(x3,x4)< x1 or min(x3,x4)>x1:
            return None
        a=(y3-y4)/(x3-x4)
        b=y3-a*x3
        y_inter=a*x1+b
        condition=(y_inter<=max(y1,y2) and y_inter>=min(y1,y2))
        if condition:
            return [x1,y_inter]
        if not condition:
            return None

    # manage exception when second segment is vertical
    if type(D2) ==type(None) and not type(D1) ==type(None):
        if max(x1,x2)< x3 or min(x1,x2)>x3:
            return None
        a=(y1-y2)/(x1-x2)
        b=y1-a*x1
        y_inter=a*x3+b
        condition=(y_inter<=max(y3,y4) and y_inter>=min(y3,y4))
        if condition:
            return [x3,y_inter]
        if not condition:
            return None
    # both segments are vertical
    if type(D2) ==type(None) and type(D1) ==type(None):
        return None

    # computing the two intersections
    X,Y=intersection_2_lines(D1,D2)
    # segment's conditions on those intersections : on x-coordinates on S1 AND S2 segments is enought...
    condition1=(X>=min(x1,x2) and X<=max(x1,x2)) or (Y>=min(y1,y2) and Y<=max(y1,y2) and y1!=y2)
    condition2=(X>=min(x3,x4) and X<=max(x3,x4)) or (Y>=min(y3,y4) and Y<=max(y3,y4) and y3!=y4)
    if condition1 and condition2:
        return [X,Y]

    return None


# Calculate the angle (in radians) from the barycenter to point P
# Returns angle in [0, 2*pi), or None if P equals the barycenter
def angle(baricenter,P):
    xb,yb=baricenter
    xp,yp=P
    xp_,yp_=xp-xb,yp-yb
    if xp_==0:
        if yp_>0:
            return np.pi/2
        if yp_<0:
            return np.pi/2
        else:
            return None
    q=(yp_)/(xp_)
    if xp_>0:
        return float(np.arctan(q))
    else: # xp-xb<0
        return np.pi+float(np.arctan(q))

# Check if two segments are identical (same endpoints, regardless of order)
# Returns True if segments are the same, False otherwise
def identical_segment(S1,S2):
    x1,y1=S1[0]
    x2,y2=S1[1]
    x3,y3=S2[0]
    x4,y4=S2[1]
    if (S1[0]==S2[0] and S1[1]==S2[1]) or (S1[0]==S2[1] and S1[1]==S2[0]):
        return True
    return False

# Remove duplicate segments from a list of segments
# Returns a list with only unique segments
def eliminate_identicals_segments(list_segment):
    result=[]
    for segment in list_segment:
        not_in=True
        for item in result:
            if identical_segment(item,segment):
                not_in=False
        if not_in:
            result.append(segment)
    return result



# Find the index of the minimum value in a list
def min_index_list(l):
    min,i_min=np.inf,0
    for i in range(len(l)):
        if l[i]<min:
            min,i_min=l[i],i
    return i_min

# Get the indices that would sort the list in ascending order
# Returns a list of indices representing the sorted order
def order(l):
    L=list(np.copy(l))
    order=[0]*len(L)
    i=0
    while min(L)<np.inf:
        j=min_index_list(L)
        order[i]=j
        L[j]=np.inf
        i+=1


    return order

# Reorder list L according to the sorted order of list l
# Both lists must have the same length
def order_by(L,l):
    indices=order(l)
    return [L[i] for i in indices]


# Plot a segment by drawing n points along it
# Parameters: segment S, number of points n, color, transparency alpha, marker style, marker size s
def plot_segment(S,n=100,color='grey',alpha=1,marker="s",s=10):
    x1,y1=S[0]
    x2,y2=S[1]
    dx,dy=(x1-x2)/n,(y1-y2)/n
    X=[x2+dx*i for i in range(n+1)]
    Y=[y2+dy*i for i in range(n+1)]
    plt.scatter(X,Y,c=color,alpha=alpha,marker=marker,s=s,)




# Get all segments forming the boundary of a polygon (box)
# Points in the box can be in any order - they will be sorted by angle from centroid
# Returns a list of segments connecting consecutive points in counterclockwise order
def all_segments_of_box(box):
    n=len(box[0])
    # sort all points by their angle to the centroïd
    angles=[angle(centroïd(box),[box[0][i],box[1][i]]) for i in range(len(box[0]))]
    # box ordered
    box[0]=order_by(box[0],angles)
    box[1]=order_by(box[1],angles)
    # points
    points=[[box[0][i],box[1][i]] for i in range(len(box[0]))]
    # segments
    all_segments=[[points[-1],points[0]]]
    for i in range(len(points)-1):
        all_segments.append([points[i],points[i+1]])

    return all_segments





# Find all intersection points between a segment S and the boundary of a polygon (box)
# Returns a list of intersection points [x,y]
def intersection_segment_box(box,S):
    all_segments=all_segments_of_box(box)
    #all segments of the box
    intersections=[]
    for S2 in all_segments:
        inter=intersection_2_segments(S,S2)
        if inter:
            intersections.append(inter)
    return intersections





# Check if point P is inside the polygon (box)
# Returns True if P is inside, False otherwise
# Method: if segment from P to centroid intersects the boundary, P is outside
def P_inside_box(box,P):
    # The segment betwenn P and the centroïd of the box
    S=[P,centroïd(box)]
    # Let's find the two intersection between this line and the box
    intersections=intersection_segment_box(box,S)
    # If there is an intersection between the box and S, it means that P is outside...
    if intersections==[]:
        return True
    return False



# Calculate the perpendicular bisector of the segment between two points (duplicate function)
def line_perpendicular_2_points(P1,P2):
    x1,y1=P1
    x2,y2=P2
    a,b=line_btw_2_points(P1,P2)
    xm,ym=(x1+x2)/2,(y1+y2)/2
    if a==0:
        return None
    return -1/a,ym+xm/a
    # cas optimal

# Find the intersection points between a box and the perpendicular bisector of segment PQ
# Returns intersection points where the perpendicular bisector cuts the box boundary
def slicing_box_intersection(box,P,Q):
    xp,yp=P
    xq,yq=Q
    xmin,xmax,ymin,ymax=min(box[0])-1,max(box[0])+1,min(box[1])-1,max(box[1])+1 # padding 1 just in case
    D=line_perpendicular_2_points(P,Q)
    # let's define a semgent from D
    if D==None: # vertical : yp==yq. S is vertical in x=(xp+xq)/2
        x=(xp+xq)/2
        S=[[x,ymin],[x,ymax]]
    else: # retrieve a and b coefficients
        a,b=D
        S=[[xmin,xmin*a+b],[xmax,xmax*a+b]]

    # then use intersection_segment_box
    intersection=intersection_segment_box(box,S)
    return intersection



# Split a box (polygon) into two parts using the perpendicular bisector of segment PQ
# Returns two boxes: one containing P, one containing Q
def slicing_box(box,P,Q):
    xp,yp=P
    xq,yq=Q
    xmin,xmax,ymin,ymax=min(box[0]),max(box[0]),min(box[1]),max(box[1]) # padding 1 just in case
    D=line_perpendicular_2_points(P,Q)
    # let's define a semgent from D
    if D==None: # vertical : yp==yq. S is vertical in x=(xp+xq)/2
        x=(xp+xq)/2
        S=[[x,ymin],[x,ymax]]
    else: # retrieve a and b coefficients
        a,b=D
        S=[[xmin,xmin*a+b],[xmax,xmax*a+b]]

    # then use intersection_segment_box
    intersection=intersection_segment_box(box,S)
    separation=line_btw_2_points(intersection[0],intersection[1])
    # all points (including intersection - 2 points)
    all_points=[[box[0][i],box[1][i]] for i in range(len(box[0]))]
    # points above/below separation
    upper_points=[]
    below_points=[]
    if separation==None:
        for points in all_points:
            if points[0]<=intersection[0][0]:
                upper_points.append(points)
            if points[0]>=intersection[0][0]:
                below_points.append(points)
        return list_point_to_box(upper_points+intersection),list_point_to_box(below_points+intersection) #upper =left, below =right
    a,b=separation
    for points in all_points:
        x=points[0]
        y=points[1]
        b_=y-a*x
        if b_>=b:
            upper_points.append(points)
        if b_<=b:
            below_points.append(points)
    return list_point_to_box(upper_points+intersection),list_point_to_box(below_points+intersection)

# One iteration of Voronoi diagram construction
# Splits box_frame based on the perpendicular bisector between points i and j
# Returns the half-space containing point i
def one_iter_voronoï(box_frame,box,i,j):
    # slices
    P,Q=[box[0][i],box[1][i]],[box[0][j],box[1][j]]
    up,down=slicing_box(box_frame,P,Q)
    if P_inside_box(up,P):
        return up
    return down


# Compute the Voronoi cell for point i within box_frame
# Returns the region closest to point i compared to all other points
def one_point_voronoï(box_frame,box,i):
    res,res_frame=[list(np.copy(box[0])),list(np.copy(box[1]))],box_frame
    l=[i for i in range(len(box[0]))]
    l.pop(i)
    for j in l:
        P=[box[0][i],box[1][i]]
        Q=[box[0][j],box[1][j]]
        slicing=slicing_box_intersection(res_frame,P,Q)
        if len(slicing)==2:
            res_frame=one_iter_voronoï(res_frame,res,i,j)
    return res_frame


# Compute the complete Voronoi diagram for all points in box
# Returns a list of Voronoi cells (one for each point)
def voronoï(box_frame,box):
    res=[]
    for i in range(len(box[0])):
        res.append(one_point_voronoï(box_frame,box,i))
    return res


# Classify dataset points according to which Voronoi cell they belong to
# Returns a list of labels (indices) indicating which cell each point belongs to
def points_classification(dataset,voronoï):
    n=len(dataset[0])
    labels=[0]*n
    for i in range(n):
        point=[dataset[0][i],dataset[1][i]]
        for j in range(len(voronoï)):
            area=voronoï[j]
            if P_inside_box(area,point):
                labels[i]=j

    return labels


# Calculate the centroid of each cluster defined by the Voronoi diagram
# Returns the mean position of points in each Voronoi cell
def get_means(dataset,voronoï):
    labels=points_classification(dataset,voronoï)
    means=[]
    l=[i for i in range(max(labels)+1)]
    for i in l:
        mask=[labels[j]==i for j in range(len(labels))]
        box_class_i0,box_class_i1=[dataset[0][j] for j in range(len(labels)) if mask[j]],[dataset[1][j] for j in range(len(labels)) if mask[j]]
        box_class_i=[box_class_i0,box_class_i1]
        if len(box_class_i0)!=0:
            mean_i=centroïd(box_class_i)
            means.append(mean_i)
    return list_point_to_box(means)


# Plot the boundary of a polygon (box) by drawing all its segments
def plot_box(box):
    Segments=all_segments_of_box(box)
    for segment in Segments:
        plot_segment(segment)

# Initialize k random points within a rectangular box_frame for Lloyd's algorithm
# Returns k random points in box format
def random_init_lloyd(k,box_frame):
    xmin,xmax=min(box_frame[0]),max(box_frame[0])
    ymin,ymax=min(box_frame[1]),max(box_frame[1])
    init_x=[rd.random()*(xmax-xmin)+xmin for i in range(k)]
    init_y=[rd.random()*(ymax-ymin)+ymin for i in range(k)]
    return [init_x,init_y]


# Visualize the Voronoi diagram with the dataset points colored by cluster
# Plots: data points (colored by cluster), cluster centers (red), Voronoi cells (boundaries)
def plot_voronoï(dataset,voronoï,box_frame):
    colors=points_classification(dataset,voronoï)
    box_clusters=get_means(dataset,voronoï)
    plt.scatter(dataset[0],dataset[1],c=colors)
    plt.scatter(box_clusters[0],box_clusters[1],c='red')
    for area in voronoï:
        plot_box(area)



# Calculate the within-cluster sum of squares (WCSS) for the clustering
# Lower values indicate tighter clusters
# Returns the average squared distance of points to their cluster centers
def within_sum_square(dataset,voronoï):
    means=get_means(dataset,voronoï)
    labels=points_classification(dataset,voronoï)
    result=0
    for i in range(max(labels)+1):
        mask=[labels[j]==i for j in range(len(labels))]
        distances=[(means[0][i]-dataset[0][j])**2+(means[1][i]-dataset[1][j])**2 for j in range(len(labels)) if mask[j]]
        result+=sum(distances)
    return result/len(dataset[0])


# Calculate the silhouette score for the clustering
# Measures how well-separated clusters are
# Score ranges from -1 (poor) to 1 (good clustering)
# Returns the average silhouette coefficient across all points
def silhouette(dataset,v):
    n=len(dataset[0])
    a,b,s=[0]*n,[0]*n,[0]*n
    labels=points_classification(dataset,v)
    for i in range(n):
        eps=10**(-15)
        x,y=dataset[0][i],dataset[1][i]
        mask_a=[labels[j]==labels[i] for j in range(n)]
        mask_b=[not e for e in mask_a]
        n_a=sum([1 for i in range(n) if mask_a[i]])
        n_b=n-n_a
        a[i]=sum([(x-dataset[0][i])**2+(y-dataset[1][i])**2 for i in range(n) if mask_a[i]])/(n_a+eps)
        b[i]=sum([(x-dataset[0][i])**2+(y-dataset[1][i])**2 for i in range(n) if mask_b[i]])/(n_b+eps)
    s=[(b[i]-a[i])/float(max(np.array(b)-np.array(a))) for i in range(n)]
    return sum(s)/n


import numpy as np
import matplotlib.pyplot as plt
import random as rd