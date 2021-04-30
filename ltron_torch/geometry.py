def angle_surrogate(x, dim_1=-2, dim_2=-1):
    '''
    Given a tensor of 3x3 rotation matrices, this will return a matrix of
    values from 0 to 1 representing the angle expressed by the rotation matrix.
    A value of 0 corresponds to a 180 degree rotation, a value of 1 corresponds
    to a 0 degree rotation.  This value is not a linear transform of the angle,
    but is monotonically consistent with it.
    '''
    default_index = [slice(None) for _ in x.shape]
    trace = 0
    for i in range(3):
        index = default_index[:]
        index[dim_1] = i
        index[dim_2] = i
        trace = trace + x[tuple(index)]
    
    return (trace + 1.) / 4.
