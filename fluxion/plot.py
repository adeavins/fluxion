def plot(results):

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for key, f in results.items():
        fig = plt.figure()
        s = fig.add_subplot(1, 1, 1)

        if len(f.dimensions) == 1:
            s.plot(f.dimensions[0].grid, f.data)
        elif len(f.dimensions) == 2:
            dims = f.dimensions

            if dims[0].uniform and dims[1].uniform:
                s.imshow(
                    f.data.T,
                    extent=(dims[0].grid[0], dims[0].grid[-1], dims[1].grid[0], dims[1].grid[-1]),
                    aspect='auto',
                    origin='lower',
                    cmap=matplotlib.cm.viridis)
            else:
                # use imshow(griddata(...))?
                raise NotImplementedError

        fig.savefig(key + '.pdf')
