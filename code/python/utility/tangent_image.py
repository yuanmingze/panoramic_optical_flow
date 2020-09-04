import math

# Generate the tangent image with Gnomonic Projection
# https://mathworld.wolfram.com/GnomonicProjection.html
#


# def sphere2tangent():
#     pass

def tangent2sphere(x, y, center_theta = 0, center_phi = 0):
    """
    project the tangent image to equiangular image.
    """
    # Compute the projection back onto sphere
    rho = (x**2 + y**2).sqrt()
    # nu = rho.atan()
    nu = math.atan(rho)
    #out_lat = (nu.cos() * lat.sin() + y * nu.sin() * lat.cos() / rho).asin()
    theta = math.asin(math.cos(nu) * math.sin(center_theta) + y * math.sin(nu) * math.cos(center_theta) /rho)
    phi = math.asin(math.cos(nu) * math.sin(centerP))
    
    # out_lon = lon + torch.atan2(x * nu.sin(), rho * lat.cos() * nu.cos() - y * lat.sin() * nu.sin())
    phi = center_phi + math.atan2(x * math.sin(nu), rho * math.cos(phi) * math.cos(nu) - y * math.sin(phi) * math.sin(nu))

    # Handle the (0,0) case
    out_lat[..., [(kh // 2) * kw + kw // 2]] = lat
    out_lon[..., [(kh // 2) * kw + kw // 2]] = lon

    # Compensate for longitudinal wrap around
    phi = ((phi + math.pi) % (2 * math.pi)) - math.pi


# def forward_gnomonic_projection(lon, lat, center_coord=(0.0, 0.0)):
#     """
#     Computes the projection from the sphere to the map
#     """
#     center_lon, center_lat = parse_center_coord(center_coord)
#     view = match_dims(lon, lat, center_coord)
#     if view is not None:
#         center_lon = center_lon.view(view)
#         center_lat = center_lat.view(view)
#     cos_c = sin(center_lat) * sin(lat) + cos(center_lat) * cos(lat) * cos( lon - center_lon)
#     x = cos(lat) * sin(lon - center_lon) / cos_c
#     y = (cos(center_lat) * sin(lat) - sin(center_lat) * cos(lat) * cos(lon - center_lon)) / cos_c
#     return x, y

#  lon --> phi
#  lat --> theta
# 

# def forward_gnomonic_projection(lon, lat, center_coord=(0.0, 0.0)):
def sphere2tangent(theta, phi, center_theta = 0, center_phi = 0):
    """

    Computes the projection from the sphere to the tangent image
    """
    # center_lon, center_lat = parse_center_coord(center_coord)
    # view = match_dims(phi, lat, center_coord)
    # if view is not None:
    #     center_lon = center_lon.view(view)
    #     center_lat = center_lat.view(view)

    cos_c = math.sin(center_theta) * math.sin(theta) + math.cos(center_theta) * math.cos(theta) * math.cos(phi - center_phi)
    x = math.cos(theta) * math.sin(phi - center_phi) / cos_c
    y = (math.cos(center_theta) * math.sin(theta) - math.sin(center_theta) * math.cos(theta) * math.cos(phi - center_phi)) / cos_c
    return x, y


def get_equirectangular_grid_resolution(shape):
    '''Returns the resolution between adjacency grid indices for an equirectangular grid'''
    H, W = shape
    res_lat = math.pi / (H - 1)
    res_lon = 2 * math.pi / W
    return res_lat, res_lon


def equirectangular_meshgrid(shape):
    H, W = shape
    lat = torch.linspace(-math.pi / 2, math.pi / 2,
                         steps=H).view(-1, 1).expand(-1, W)
    lon = torch.linspace(-math.pi, math.pi,
                         steps=W + 1)[:-1].view(1, -1).expand(H, -1)
    res_lat, res_lon = get_equirectangular_grid_resolution(shape)
    return lat, lon, res_lat, res_lon

def reverse_equirectangular_projection_map(shape, kernel_size):
    # For convenience
    H, W = shape
    kh, kw = kernel_size

    # Get lat/lon mesh grid and resolution
    lat, lon, res_lat, res_lon = equirectangular_meshgrid(shape)

    # Kernel
    x = torch.zeros(kernel_size)
    y = torch.zeros(kernel_size)
    for i in range(kh):
        cur_i = i - (kh // 2)
        for j in range(kw):
            cur_j = j - (kw // 2)

            # Project the equirectangular image onto the tangent plane at the equator
            x[i, j] = cur_j * res_lon
            y[i, j] = cur_i * res_lat

    # Equalize views
    lat = lat.view(H, W, 1)
    lon = lon.view(H, W, 1)
    x = x.view(1, 1, kh * kw)
    y = y.view(1, 1, kh * kw)

    # Compute the projection back onto sphere
    out_lat = y + lat
    out_lon = x / out_lat.cos() + lon

    # Compensate for longitudinal wrap around
    out_lon = ((out_lon + math.pi) % (2 * math.pi)) - math.pi

    # Return OH, OW, KH*KW, 2
    return torch.stack((out_lon, out_lat), -1)


def reverse_gnomonic_projection_map(shape, kernel_size):
    # For convenience
    H, W = shape
    kh, kw = kernel_size

    # Get lat/lon mesh grid and resolution
    lat, lon, res_lat, res_lon = equirectangular_meshgrid(shape)

    # Kernel
    x = torch.zeros(kernel_size)
    y = torch.zeros(kernel_size)
    for i in range(kh):
        cur_i = i - (kh // 2)
        for j in range(kw):
            cur_j = j - (kw // 2)

            # Project the equirectangular image onto the tangent plane at the equator
            x[i, j] = cur_j * res_lon
            y[i, j] = cur_i * res_lat

    # Equalize views
    lat = lat.view(H, W, 1)
    lon = lon.view(H, W, 1)
    x = x.view(1, 1, kh * kw)
    y = y.view(1, 1, kh * kw)

    # Compute the projection back onto sphere
    rho = (x**2 + y**2).sqrt()
    nu = rho.atan()
    out_lat = (nu.cos() * lat.sin() + y * nu.sin() * lat.cos() / rho).asin()
    out_lon = lon + torch.atan2(x * nu.sin(),rho * lat.cos() * nu.cos() - y * lat.sin() * nu.sin())

    # Handle the (0,0) case
    out_lat[..., [kh * kw // 2]] = lat
    out_lon[..., [kh * kw // 2]] = lon

    # Compensate for longitudinal wrap around
    out_lon = ((out_lon + math.pi) % (2 * math.pi)) - math.pi

    # Return OH, OW, KH*KW, 2
    return torch.stack((out_lon, out_lat), -1)



def gnomonic_kernel(spherical_coords, kh, kw, res_lat, res_lon):
    '''
    Creates gnomonic filters of shape (kh, kw) with spatial resolutions given by (res_lon, res_lat) and centers them at each coordinate given by <spherical_coords>

    spherical_coords: H, W, 2 (lon, lat)
    kh: vertical dimension of filter
    kw: horizontal dimension of filter
    res_lat: vertical spatial resolution of filter
    res_lon: horizontal spatial resolution of filter
    '''

    lon = spherical_coords[..., 0]
    lat = spherical_coords[..., 1]
    num_samples = spherical_coords.shape[0]

    # Kernel
    x = torch.zeros(kh * kw)
    y = torch.zeros(kh * kw)
    for i in range(kh):
        cur_i = i - (kh // 2)
        for j in range(kw):
            cur_j = j - (kw // 2)
            # Project the equirectangular image onto the tangent plane at the equator
            x[i * kw + j] = cur_j * res_lon
            y[i * kw + j] = cur_i * res_lat

    # Equalize views
    lat = lat.view(1, num_samples, 1)
    lon = lon.view(1, num_samples, 1)
    x = x.view(1, 1, kh * kw)
    y = y.view(1, 1, kh * kw)

    # Compute the projection back onto sphere
    rho = (x**2 + y**2).sqrt()
    nu = rho.atan()
    out_lat = (nu.cos() * lat.sin() + y * nu.sin() * lat.cos() / rho).asin()
    out_lon = lon + torch.atan2(x * nu.sin(), rho * lat.cos() * nu.cos() - y * lat.sin() * nu.sin())

    # Handle the (0,0) case
    out_lat[..., [(kh // 2) * kw + kw // 2]] = lat
    out_lon[..., [(kh // 2) * kw + kw // 2]] = lon

    # Compensate for longitudinal wrap around
    out_lon = ((out_lon + math.pi) % (2 * math.pi)) - math.pi

    # Return (1, num_samples, kh*kw, 2) map at locations given by <spherical_coords>
    return torch.stack((out_lon, out_lat), -1)

