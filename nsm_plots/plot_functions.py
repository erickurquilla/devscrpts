import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from matplotlib.ticker import AutoLocator, AutoMinorLocator, LogLocator

# Font settings
mpl.rcParams['font.size'] = 22
mpl.rcParams['font.family'] = 'serif'
mpl.rc('text', usetex=False)

# Tick settings
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.major.pad'] = 8
mpl.rcParams['xtick.minor.size'] = 4

mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['ytick.major.size'] = 7
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['ytick.minor.width'] = 2

# Axis linewidth
mpl.rcParams['axes.linewidth'] = 2

# Tick direction and enabling ticks on all sides
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True

def apply_custom_settings(ax, leg=None, log_scale_y=False):

    if log_scale_y:
        # Use LogLocator for the y-axis if it's in log scale
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=100))
    else:
        # Use AutoLocator for regular scales
        ax.yaxis.set_major_locator(AutoLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    # Apply the AutoLocator for the x-axis
    ax.xaxis.set_major_locator(AutoLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    # Legend settings
    if leg is not None:
        leg.get_frame().set_edgecolor('w')
        leg.get_frame().set_linewidth(0.0)

def plot_color_map_with_scattered_points(x, y, z, bh_r, bh_x, bh_y, min_cb, max_cb, x_label, y_label, title, cbar_label, colormap, filename, x_scat, y_scat, size_scat, marker_scat, color_scat, doshow=True, dosave=True):

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot pcolormesh
    c = ax.pcolormesh(x, y, z, shading='auto', cmap=colormap, vmin=min_cb, vmax=max_cb)
    
    # Scatter points
    ax.scatter(x_scat, y_scat, s=size_scat, marker=marker_scat, color=color_scat)

    # Add contour lines
    # contour = ax.contour(x, y, z, colors='black', linewidths=1.5, levels=3)
    # contour = ax.contour(x, y, z, colors='black', linewidths=1.5, levels=[0.5,2.0,3.5])
    # ax.clabel(contour, inline=True, fontsize=15, fmt='%1.1f')
    # ax.clabel(contour, inline=True, fontsize=15, fmt='%1.1e')

    # Plot settings
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # ax.set_title(title+'\nmin: {:.2e}\nmax: {:.2e}'.format(np.nanmin(z), np.nanmax(z)))
    ax.set_title(title)

    # Compute distance from each mesh point to the black hole center
    distance = np.sqrt((x - bh_x)**2 + (y - bh_y)**2)
    mask = distance <= bh_r  # True inside the black hole

    # Create masked array: only show black where mask is True
    bh_overlay = np.ma.masked_where(~mask, mask.astype(float))

    # Plot black hole with pcolormesh using 0=transparent, 1=black
    bh_cmap = plt.cm.gray_r  # white=0, black=1
    bh_cmap.set_bad(color='none')  # transparent for masked points

    ax.pcolormesh(
        x, y, bh_overlay,
        shading='auto',
        cmap=bh_cmap,
        vmin=0, vmax=1,
        alpha=1.0
    )

    # Add color bar
    cbar = fig.colorbar(c, ax=ax, label=cbar_label)
    cbar.ax.yaxis.set_minor_locator(AutoMinorLocator())

    apply_custom_settings(ax, None, False)

    # Ensure equal aspect ratio
    ax.set_aspect('equal', 'box')

    # Save figure
    if dosave:
        fig.savefig(filename, format='png', bbox_inches='tight')

    # Display figure
    if doshow:
        plt.show()
    # display(fig)
    
    # Close figure
    plt.close(fig)

def plot_color_map(x, y, z, min_cb, max_cb, x_label, y_label, title, cbar_label, colormap, filename, doshow=True, dosave=True):

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot pcolormesh
    c = ax.pcolormesh(x, y, z, shading='auto', cmap=colormap, vmin=min_cb, vmax=max_cb)

    # Add contour lines
    # contour = ax.contour(x, y, z, colors='black', linewidths=1.5, levels=3)
    # contour = ax.contour(x, y, z, colors='black', linewidths=1.5, levels=[0.5,2.0,3.5])
    # ax.clabel(contour, inline=True, fontsize=15, fmt='%1.1f')
    # ax.clabel(contour, inline=True, fontsize=15, fmt='%1.1e')

    # Plot settings
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # ax.set_title(title+'\nmin: {:.2e}\nmax: {:.2e}'.format(np.nanmin(z), np.nanmax(z)))
    ax.set_title(title)

    # Add color bar
    cbar = fig.colorbar(c, ax=ax, label=cbar_label)
    cbar.ax.yaxis.set_minor_locator(AutoMinorLocator())

    apply_custom_settings(ax, None, False)

    # Ensure equal aspect ratio
    ax.set_aspect('equal', 'box')

    # Save figure
    if dosave:
        fig.savefig(filename, format='png', bbox_inches='tight')

    # Display figure
    if doshow:
        plt.show()
    # display(fig)
    
    # Close figure
    plt.close(fig)

def plot_pcolormesh_with_contour_and_scatter(
    x1, y1, z1, min_cb1, max_cb1, cbar_label1, colormap1,
    x2, y2, z2, min_cb2, max_cb2, cbar_label2, colormap2,
    x_label, y_label, title, filename,
    x_scatter1, y_scatter1, z_scatter1,
    x_scatter2, y_scatter2, z_scatter2,
    sc_point_color):

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot pcolormesh
    c1 = ax.pcolormesh(x1, y1, z1, shading='auto', cmap=colormap1, vmin=min_cb1, vmax=max_cb1)
    c2 = ax.pcolormesh(x2, y2, z2, shading='auto', cmap=colormap2, vmin=min_cb2, vmax=max_cb2)

    # ax.scatter([2*np.pi/20.0], [0.85], color=sc_point_color, s=800)

    ax.scatter(x_scatter1, y_scatter1, c=z_scatter1, cmap=colormap1, vmin=min_cb1, vmax=max_cb1, s=100, edgecolor='black')
    ax.scatter(x_scatter2, y_scatter2, c=z_scatter2, cmap=colormap2, vmin=min_cb2, vmax=max_cb2, s=100, edgecolor='black')

    # Add contour lines
    # contour1 = ax.contour(x1, y1, z1, colors='black', linewidths=0.5, levels=4)
    # contour2 = ax.contour(x2, y2, z2, colors='black', linewidths=0.5, levels=4)
    # ax.clabel(contour1, inline=True, fontsize=15, fmt='%1.1f')
    # ax.clabel(contour2, inline=True, fontsize=15, fmt='%1.1f')

    # Plot settings
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-1, 1)

    # Add color bar
    cbar1 = fig.colorbar(c1, ax=ax, label=cbar_label1, location='right')
    cbar2 = fig.colorbar(c2, ax=ax, label=cbar_label2, location='right')
    cbar1.ax.yaxis.set_minor_locator(AutoMinorLocator())
    cbar2.ax.yaxis.set_minor_locator(AutoMinorLocator())

    leg = ax.legend(framealpha=0.0, ncol=1, fontsize=15)
    apply_custom_settings(ax, leg, False)

    # Ensure equal aspect ratio
    ax.set_aspect('auto', 'box')

    # Save figure
    fig.savefig(filename, format='png', bbox_inches='tight')

    # Display figure
    plt.show()

    # Close figure
    plt.close(fig)

def plot_pcolormesh_with_contour_and_scatter_one_cbar(
    x1, y1, z1, min_cb1, max_cb1, cbar_label1, colormap1,
    x_label, y_label, title, filename,
    x_scatter1, y_scatter1, z_scatter1, doshow=True, dosave=True):

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot pcolormesh for both, but only use colorbar from c1
    c1 = ax.pcolormesh(x1, y1, z1, shading='auto', cmap=colormap1, vmin=min_cb1, vmax=max_cb1)

    ax.scatter(x_scatter1, y_scatter1, c=z_scatter1, cmap=colormap1, vmin=min_cb1, vmax=max_cb1, s=100, edgecolor='black')

    # Plot settings
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.set_xlim(0 - 0.3, 2 * np.pi + 0.3)
    ax.set_ylim(-1.1, +1.1)

    # Add only one color bar
    cbar1 = fig.colorbar(c1, ax=ax, label=cbar_label1, location='right')
    cbar1.ax.yaxis.set_minor_locator(AutoMinorLocator())

    leg = ax.legend(framealpha=0.0, ncol=1, fontsize=15)
    apply_custom_settings(ax, leg, False)

    ax.set_aspect('auto', 'box')

    if dosave:
        fig.savefig(filename, format='png', bbox_inches='tight')
    if doshow:
        plt.show()

    plt.close(fig)

def plot_pcolormesh_with_contour_and_scatter_one_cbar_with_bh_shadow(
    x1, y1, z1, min_cb1, max_cb1, cbar_label1, colormap1,
    x_label, y_label, title, filename,
    x_scatter1, y_scatter1, z_scatter1, cos_theta_bh_shadow, phi_bh_shadow, costheta_lowlimit, costheta_highlimit, doshow=True, dosave=True):

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot pcolormesh for both, but only use colorbar from c1
    c1 = ax.pcolormesh(x1, y1, z1, shading='auto', cmap=colormap1, vmin=min_cb1, vmax=max_cb1)

    ax.scatter(x_scatter1, y_scatter1, c=z_scatter1, cmap=colormap1, vmin=min_cb1, vmax=max_cb1, s=100, edgecolor='black')

    # Plot BH shadow boundary
    ax.plot(phi_bh_shadow, cos_theta_bh_shadow, color='black', linestyle='solid', linewidth=2)
    
    # Plot settings
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.set_xlim(0 - 0.3, 2 * np.pi + 0.3)
    ax.set_ylim(costheta_lowlimit, costheta_highlimit)

    # Add only one color bar
    cbar1 = fig.colorbar(c1, ax=ax, label=cbar_label1, location='right')
    cbar1.ax.yaxis.set_minor_locator(AutoMinorLocator())

    leg = ax.legend(framealpha=0.0, ncol=1, fontsize=15)
    apply_custom_settings(ax, leg, False)

    ax.set_aspect('auto', 'box')

    if dosave:
        fig.savefig(filename, format='png', bbox_inches='tight')
    if doshow:
        plt.show()

    plt.close(fig)    

def plot_colored_lines(x, y, time_s, xlabel, ylabel, cbarlabel, filename=None, xlog= False, ylog=False, ylowerlimit=None, yupperlimit=None):

    fig, ax = plt.subplots(figsize=(10, 6))
    num_lines = y.shape[0]

    # Normalize time for colormap
    norm = plt.Normalize(time_s.min(), time_s.max())
    cmap = plt.cm.viridis

    for i in range(num_lines):
        color = cmap(norm(time_s[i]))
        ax.plot(x, y[i,:], color=color)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Only needed for matplotlib < 3.1

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(cbarlabel)
    cbar.ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_ylim(ylowerlimit, yupperlimit)

    if xlog:
        ax.set_xscale('log')

    leg = ax.legend(framealpha=0.0, ncol=2, fontsize=20)
    apply_custom_settings(ax, leg, ylog)
    if filename is not None and filename != "":
        plt.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close(fig)