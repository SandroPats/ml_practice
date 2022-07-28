def grad_finite_diff(function, w, eps=1e-8):
    e_arr = np.eye(w.shape[0])
    result = np.apply_along_axis(lambda e_i: (function(w + e_i*eps) -
                                              function(w)) / eps, 1, e_arr)
    return result
