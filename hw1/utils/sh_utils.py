import torch

# https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#%E2%84%93_=_1_2
SH_C0_0 = 0.28209479177387814  # Y0,0:  1/2*sqrt(1/pi)       plus
SH_C1_0 = -0.4886025119029199  # Y1,-1: sqrt(3/(4*pi))       minus
SH_C1_1 = 0.4886025119029199   # Y1,0:  sqrt(3/(4*pi))       plus
SH_C1_2 = -0.4886025119029199  # Y1,1:  sqrt(3/(4*pi))       minus


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = SH_C0_0 * sh[..., 0]
    
    if deg ==1:
        # Begin code 3.1 ##
        # todo: add support for 1st-degree SH 
        # result = result + SH_C1_0 * ??? + SH_C1_1 * ??? + SH_C1_2 * ???
        pass
        # End code 3.1 ##
    return result