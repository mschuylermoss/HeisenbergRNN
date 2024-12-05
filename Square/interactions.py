def coord_to_site_bravais(L, x, y, snake=False):
    if snake and (y % 2 == 1):
        return L * y + L - x - 1
    else:
        return L * y + x


def site_to_coord_bravais(L, site, snake=False):
    y = site // L
    if snake and (y % 2 == 1):
        x = L - (site - L * y + 1)
    else:
        x = site - L * y
    return x, y


def buildlattice_square(Lx, Ly, bc="open", snake=False):
    assert Lx == Ly, 'Lx must be equal to Ly'
    assert bc in ['open', 'periodic']
    interactions = []
    coord_fn = lambda x, y: coord_to_site_bravais(Lx, x, y, snake)

    for n in range(Lx - 1):
        for n_ in range(Ly):
            # horizontal square lattice interactions (excluding boundary terms)
            interactions.append([coord_fn(n, n_), coord_fn(n + 1, n_)])

    for n in range(Lx):
        for n_ in range(Ly - 1):
            # vertical square lattice interactions (excluding boundary terms)
            interactions.append([coord_fn(n, n_), coord_fn(n, n_ + 1)])

    if bc == "periodic":
        for n in range(Lx):
            interactions.append([coord_fn(n, Ly - 1), coord_fn(n, 0)])
        for n_ in range(Ly):
            interactions.append([coord_fn(Lx - 1, n_), coord_fn(0, n_)])

    return interactions


def buildlattice_alltoall(L, snake=False):
    interactions = []
    N_spins = L ** 2

    for i in range(N_spins):
        for j in range(i, N_spins):
            xi = i % L
            yi = i // L
            xj = j % L
            yj = j // L
            interaction = [coord_to_site_bravais(L, xi, yi, snake),
                           coord_to_site_bravais(L, xj, yj, snake)]
            interactions.append(interaction)

    return interactions


def buildlattice_alltoall_primitive_vector(L: int, p1, p2, snake=False, periodic=False):
    interactions = {}
    N_spins = L ** 2

    for i in range(N_spins):
        for j in range(i, N_spins):
            xi_square = i % L
            yi_square = i // L

            xj_square = j % L
            yj_square = j // L

            interaction = (coord_to_site_bravais(L, xi_square, yi_square, snake),
                           coord_to_site_bravais(L, xj_square, yj_square, snake))

            delta_x_square = xj_square - xi_square
            delta_y_square = yj_square - yi_square

            if periodic:
                x_ = -1 * (L - delta_x_square)
                y_ = -1 * (L - delta_y_square)
                if x_ ** 2 < delta_x_square ** 2:
                    delta_x_square = x_
                if y_ ** 2 < delta_y_square ** 2:
                    delta_y_square = y_
                delta_x = delta_x_square * p1[0] + delta_y_square * p2[0]
                delta_y = delta_x_square * p1[1] + delta_y_square * p2[1]
                interactions[interaction] = (delta_x, delta_y)
            else:
                delta_x = delta_x_square * p1[0] + delta_y_square * p2[0]
                delta_y = delta_x_square * p1[1] + delta_y_square * p2[1]
                interactions[interaction] = (delta_x, delta_y)

    return interactions


def generate_sublattices_square(Lx, Ly, snake=False):
    A_coords = []
    B_coords = []
    A_sites = []
    B_sites = []
    coord_fn = lambda x, y: coord_to_site_bravais(Lx, x, y, snake)

    for nx in range(Lx):
        for ny in range(Ly):
            if nx % 2 == 0:
                if ny % 2 == 0:
                    A_coords.append((nx, ny))
                    A_sites.append(coord_fn(nx, ny))
                else:
                    B_coords.append((nx, ny))
                    B_sites.append(coord_fn(nx, ny))
            else:
                if ny % 2 == 0:
                    B_coords.append((nx, ny))
                    B_sites.append(coord_fn(nx, ny))
                else:
                    A_coords.append((nx, ny))
                    A_sites.append(coord_fn(nx, ny))

    return A_coords, B_coords, A_sites, B_sites


def generate_lattices_boundary(Lx, Ly, snake=False):
    corner_coords = []
    boundary_coords = []
    bulk_coords = []
    corner_sites = []
    boundary_sites = []
    bulk_sites = []

    coord_fn = lambda x, y: coord_to_site_bravais(Lx, x, y, snake)

    for nx in range(Lx):
        for ny in range(Ly):
            # corners:
            if (nx, ny) in [(0, 0), (0, Ly - 1), (Lx - 1, 0), (Lx - 1, Ly - 1)]:
                corner_coords.append((nx, ny))
                corner_sites.append(coord_fn(nx, ny))
            elif nx in [0, Lx - 1] or ny in [0, Ly - 1]:
                boundary_coords.append((nx, ny))
                boundary_sites.append(coord_fn(nx, ny))
            else:
                bulk_coords.append((nx, ny))
                bulk_sites.append(coord_fn(nx, ny))

    return corner_coords, boundary_coords, bulk_coords, corner_sites, boundary_sites, bulk_sites
