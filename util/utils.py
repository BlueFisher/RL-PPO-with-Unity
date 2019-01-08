def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


def is_converged(scalars, smoothing=0.6, threshold=0.02, converged_num=200):
    smoothed_scalars = smooth(scalars, smoothing)
    h, l = smoothed_scalars[-1] + threshold, smoothed_scalars[-1] - threshold
    for i in smoothed_scalars[-2:len(smoothed_scalars) - converged_num - 1:-1]:
        if i > h or i < l:
            return False
    return True
