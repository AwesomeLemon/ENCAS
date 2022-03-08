'''
These are just some useful commands for debugging
'''

im = images.cpu().detach().numpy()


def stats(x):
    print(f'{x.mean()=:.3f} | {x.std()=:.3f} | {x.min()=:.3f} | {x.max()=:.3f}')


lb = labels.cpu().detach().numpy().transpose((0, 2, 3, 1))


def s(xs):
    fig, axs = plt.subplots(1, len(xs))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    for ax, x in zip(axs, xs):
        ax.imshow(x)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


for i in range(8):
    s([im[i, 0], im[i, 1], lb[i] * 255])
