from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.contour import ContourSet
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.path import Path

def extend_res(res: np.ndarray[float], n: int) -> np.ndarray[float]:
    """
    Used to extend an result of m species into an array of n>=m species
    """
    i = int((res.shape[1] - 1) / 2)
    res_exp = np.zeros((res.shape[0], 2 * n + 1))
    res_exp[:, list(range(0, i)) + list(range(n, n + i)) + [-1]] = res
    return res_exp


def array_to_diag(x: np.ndarray[float]):
    """
    from [m, n] to [m, n, n].
    Where the values are on the diagonal. for each m.
    (x, y) -> (x,y,y)

    """
    return np.einsum("ij,jk->ijk", x, np.eye(x.shape[-1]))


def jac_alt_prange(params, Ps, Zs, N):

    if len(Ps.shape) == 1:
        Ps = np.repeat(Ps[None, :], N.shape[0], 0)
    res = extend_res(
        np.concatenate([Ps, Zs, np.zeros((Ps.shape[0], 2))], axis=-1),
        params["sizes"].shape[0],
    )
    n = (res.shape[-1] - 1) // 2
    Pi = res[..., :n]  # [(m,) n]
    Zi = res[..., n:-1]
    N = N  # [(m,) 1]
    dNpi = Pi * params["mu"] * params["k"] / (N + params["k"]) ** 2  # [m, n]
    dNpi = dNpi[..., None] * params["ds"]  # [m, n, n]
    dPiPj = (
        array_to_diag(
            params["mu"] * N / (N + params["k"])
            - params["lambda"]
            - params["g"] * params["K"] * Zi * params["r"] / (Pi + params["K"]) ** 2
        )
        - dNpi
    )
    dPiZj = (
        array_to_diag(-params["g"] * Pi * params["r"] / (Pi + params["K"]))
        - dNpi * params["r"]
    )
    dZiPj = array_to_diag(
        Zi * params["gamma"] * params["g"] * params["K"] / (Pi + params["K"]) ** 2
    )
    dZiZj = array_to_diag(
        params["gamma"] * params["g"] * Pi / (Pi + params["K"]) - params["delta"]
    )
    jac = np.block([[dPiPj, dPiZj], [dZiPj, dZiZj]])
    stab = np.linalg.eigvals(jac).real.max(-1) < 0
    return jac, stab


def Pi_N_Zi_vals(params, i):
    Pc = (
        params["delta"]
        * params["K"]
        / (params["gamma"] * params["g"] - params["delta"])
    )  # [N,M] / [M]
    N = (
        params["lambda"][..., i]
        * params["k"][..., i]
        / (params["mu"][..., i] - params["lambda"][..., i])
    )
    N = N.reshape(-1, 1)
    Zi = (
        (Pc[..., :i] + params["K"][..., :i])
        / (params["g"][..., :i] * params["r"])
        * (
            params["mu"][..., :i] * N / (N + params["k"][..., :i])
            - params["lambda"][..., :i]
        )
    ) 
    return Pc, N, Zi

def Pi_mm_ranges_i(params, i):
    Pc, N, Zi = Pi_N_Zi_vals(params, i) # [N, i]
    a = (Zi[..., :i] * params["ds"][:i] * params["r"]).sum(-1) + (
        Pc[..., :i] * params["ds"][:i]
    ).sum(-1)
    N_min = a + N.reshape(-1)
    N_max = (
        N_min
        + params["delta"][..., i]
        * params["K"][..., i]
        / (params["gamma"][..., i] * params["g"][..., i] - params["delta"][..., i])
        * params["ds"][i]
    )
    jac, stabl = jac_alt_prange(
        params, np.concatenate([Pc[..., :i], Pc[..., [i]] * 0.01], axis=-1), Zi, N
    )
    jac, stabr = jac_alt_prange(
        params, np.concatenate([Pc[..., :i], Pc[..., [i]] * 0.99], axis=-1), Zi, N
    )
    return (N_min, N_max, stabl, stabr)

def Pi_mm_ranges_i_no_stab(params, i):
    Pc, N, Zi = Pi_N_Zi_vals(params, i) # [N, i]
    a = (Zi[..., :i] * params["ds"][:i] * params["r"]).sum(-1) + (
        Pc[..., :i] * params["ds"][:i]
    ).sum(-1)
    N_min = a + N.reshape(-1)
    N_max = (
        N_min
        + params["delta"][..., i]
        * params["K"][..., i]
        / (params["gamma"][..., i] * params["g"][..., i] - params["delta"][..., i])
        * params["ds"][i]
    )
    return (N_min, N_max)


def slice_indices(bool_arr: np.ndarray[bool]):
    if not np.any(bool_arr):
        return 0, bool_arr.shape[0] - 1
    transitions = np.diff(bool_arr.astype(int))

    # If starts at 0, there's no 1 transition at the start
    if bool_arr[0]:
        start = 0
    else:
        start = np.where(transitions == 1)[0][0]

    # If ends at -1, there's no -1 transition at the end
    if bool_arr[-1]:
        end = bool_arr.shape[0] - 1
    else:
        end = np.where(transitions == -1)[0][0] + 1

    return start, end


def interpolate_masked_arrays_general(
    arr1: np.ndarray[float],
    arr2: np.ndarray[float],
    mask1: np.ndarray[bool],
    mask2: np.ndarray[bool],
) -> tuple[np.ndarray[float], np.ndarray[float]]:
    """
    Interpolates values in arr1 and arr2 based on overlapping masks,
    handling cases where either mask can extend further on either side.

    Modifies the array corresponding to the False mask in the outer regions.

    Args:
        arr1: First float array. Assumed arr1 >= arr2 element-wise.
        arr2: Second float array.
        mask1: Boolean mask for arr1. Has a contiguous True slice.
        mask2: Boolean mask for arr2. Has a contiguous True slice that overlaps mask1's.

    Returns:
        A tuple containing the modified arr1 and arr2.
    """
    if not (arr1.shape == arr2.shape == mask1.shape == mask2.shape):
        raise ValueError("All input arrays must have the same shape.")

    # Find the indices of the True slices
    idx1 = np.flatnonzero(mask1)
    idx2 = np.flatnonzero(mask2)
    arr1_mod = arr1.copy()
    arr2_mod = arr2.copy()
    if idx1.size == 0 or idx2.size == 0:
        print("Warning: One or both masks are all False. No interpolation possible.")
        return arr1, arr2, mask1 & mask2

    start1, end1 = idx1[0], idx1[-1]
    start2, end2 = idx2[0], idx2[-1]

    # Find overlap region boundaries
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)

    def range_to_perc(start, end):
        return np.linspace(0, 1, end - start +1, endpoint=True)

    # --- Interpolate LEFT outer region ---
    # Determine which mask starts first
    true_left_start = min(start1, start2)
    if true_left_start < overlap_start:  # Check if there is a left outer region
        int_perc = range_to_perc(true_left_start, overlap_start)
        if start1 < start2:  # mask1 starts first, interpolate arr2
            arr2_mod[true_left_start : overlap_start + 1] = (
                int_perc * arr2[true_left_start : overlap_start + 1]
                + (1 - int_perc) * arr1[true_left_start : overlap_start + 1]
            )
        elif start2 < start1:  # mask2 starts first, interpolate arr1
            arr1_mod[true_left_start : overlap_start + 1] = (
                int_perc * arr1[true_left_start : overlap_start + 1]
                + (1 - int_perc) * arr2[true_left_start : overlap_start + 1]
            )

    # --- Interpolate RIGHT outer region ---
    # Determine which mask ends last
    true_right_end = max(end1, end2)

    if true_right_end > overlap_end:  # Check if there is a right outer region
        int_perc = range_to_perc(overlap_end, true_right_end)
        if end1 > end2:  # mask1 ends last, interpolate arr2
            # Interpolate arr2 from arr2[overlap_end] at overlap_end to arr1[end1] at end1
            arr2_mod[overlap_end : true_right_end + 1] = (
                int_perc * arr1[overlap_end : true_right_end + 1]
                + (1 - int_perc) * arr2[overlap_end : true_right_end + 1]
            )
        elif end2 > end1:  # mask2 ends last, interpolate arr1
            # Interpolate arr1 from arr1[overlap_end] at overlap_end to arr2[end2] at end2
            
            arr1_mod[overlap_end : true_right_end + 1] = (
                int_perc * arr2[overlap_end : true_right_end + 1]
                + (1 - int_perc) * arr1[overlap_end : true_right_end + 1]
            )
    return arr1_mod, arr2_mod, mask1 | mask2

def get_param_dict(params: dict[str, float|str], brd):
    size_boundaries = np.linspace(params["s_min"], params["s_max"], params["n"] + 1, True)
    ds = np.diff(size_boundaries)
    match brd:
        case "r":
            sizes = size_boundaries[1:]
        case "m":
            sizes = (size_boundaries[1:] + size_boundaries[:-1]) / 2
        case _:
            sizes = size_boundaries[:-1]
    
    if (freeparam := params["param_select"].split("_")[0]) in ["mu", "lambda", "k"]:
        if params["param_select"].split("_")[1] == "0":
            def freeval_fn(v):
                return v[:, None] * sizes ** params[freeparam + "_scale"]
        else:
            def freeval_fn(v):
                return params[freeparam + "_0"] * sizes ** v[:, None]
    else:
        if params["param_select"].split("_")[1] == "0":
            def freeval_fn(v):
                return (v[:, None]
                            * (sizes * params["r"]) ** params[freeparam + "_scale"])
        else:
            def freeval_fn(v):
                return (params[freeparam + "_0"] * (sizes * params["r"]) ** v[:, None])

    vars_p = ["mu", "lambda", "k"]
    vars_z = ["g", "K", "gamma", "delta"]
    scaled_params = {"r": params["r"], "sizes": sizes, "ds": ds}

    for v in vars_p:
        if v + "_0" == params["param_select"] or v + "_scale" == params["param_select"]:
            continue
        scaled_params[v] = params[v + "_0"] * sizes ** params[v + "_scale"]
    for v in vars_z:
        if v + "_0" == params["param_select"] or v + "_scale" == params["param_select"]:
            continue
        scaled_params[v] = (
            params[v + "_0"] * (sizes * params["r"]) ** params[v + "_scale"]
        )
    if (freeparam := params["param_select"].split("_")[0]) in ["mu", "lambda", "k"]:
        if params["param_select"].split("_")[1] == "0":
            def freeval_fn(v):
                return v[:, None] * sizes ** params[freeparam + "_scale"]
        else:
            def freeval_fn(v):
                return params[freeparam + "_0"] * sizes ** v[:, None]
    else:
        if params["param_select"].split("_")[1] == "0":
            def freeval_fn(v):
                return (v[:, None]
                            * (sizes * params["r"]) ** params[freeparam + "_scale"])
        else:
            def freeval_fn(v):
                return (params[freeparam + "_0"] * (sizes * params["r"]) ** v[:, None])
    return freeparam, scaled_params, freeval_fn

def solution_regions_bin_search(
    params,
    brd="m",
    progress_frame: tk.Frame = None,
):
    progress_bar = ttk.Progressbar(progress_frame, length=200, mode="determinate")
    progress_bar.pack()
    progress_label = tk.Label(progress_frame, text="Calculating stability regions...")
    progress_label.pack()
    freeparam, scaled_params, freeval_fn = get_param_dict(params, brd)
    regions = []
    test_points = 5
    for i in range(params["n"]):
        progress_bar["value"] = (i / params["n"]) * 100
        progress_frame.update()
        if i > 1 and not np.any(regions[-1][3]):
            break

        testvs = np.linspace(params["fmin"], params["fmax"], test_points * 2, endpoint=True)
        # In this loop we try to refine the boundaries in 3 search loops.
        for t in range(3):
            inputprms = scaled_params | {freeparam: freeval_fn(testvs)}
            # print(inputprms)
            test_range = Pi_mm_ranges_i(inputprms, i)
            left_ind, right_ind = slice_indices(test_range[2] | test_range[3])
            if (
                left_ind <= 1 and right_ind >= testvs.shape[0] - 2
            ):  # Our range is good enough
                break

            # Refine left boundary if needed
            if left_ind > 1:
                left_vs = np.linspace(
                    testvs[left_ind - 1], testvs[left_ind], test_points, endpoint=True
                )
            else:
                left_vs = testvs[: left_ind + 1]
            # Refine right boundary if needed
            if right_ind < testvs.shape[0] - 2:
                right_vs = np.linspace(
                    testvs[right_ind - 1], testvs[right_ind], test_points, endpoint=True
                )
            else:
                right_vs = testvs[right_ind:]

            testvs = np.concatenate([left_vs, right_vs])

        fine_range = np.linspace(testvs[0], testvs[-1], params["m"], endpoint=True)
        fine_vals = Pi_mm_ranges_i(
            scaled_params | {freeparam: freeval_fn(fine_range)}, i
        )
        regions.append(fine_vals + (fine_range,))

    if progress_frame:
        progress_bar.destroy()
        progress_label.destroy()

    return regions
def solution_regions_no_stab(
        params:dict[str, float],
        brd="m",
        progress_frame:tk.Frame=None,
):
    progress_bar = ttk.Progressbar(progress_frame, length=200, mode="determinate")
    progress_bar.pack()
    progress_label = tk.Label(progress_frame, text="Calculating stationary regions...")
    progress_label.pack()
    freeparam, scaled_params, freeval_fn = get_param_dict(params, brd)
    regions = []
    for i in range(params["n"]):
        progress_bar["value"] = (i / params["n"]) * 100
        progress_frame.update()
        testvs = np.linspace(params["fmin"], params["fmax"], params["m"], endpoint=True)
        fine_vals = Pi_mm_ranges_i_no_stab(
            scaled_params | {freeparam: freeval_fn(testvs)}, i
        )
        regions.append(fine_vals + (testvs,))

    if progress_frame:
        progress_bar.destroy()
        progress_label.destroy()
    return regions

def find_left_boundary(paths, y_val, maxx):
    xvals = np.linspace(0, maxx, 20)
    points = np.stack([xvals, np.repeat(y_val, 20)], axis=1)
    valids = np.ones(20, dtype=bool)
    for path in paths:
        valids &= ~path.contains_points(points)
    return xvals[valids][1]

def draw_solution_regions_bin(
    ax: plt.Axes, params: dict[str, float], progress_frame: tk.Frame
):
    ax.cla()
    try:
        regions = solution_regions_bin_search(
            params,
            progress_frame=progress_frame,
        )
        regions_no_stab = solution_regions_no_stab(
            params,
            progress_frame=progress_frame,
        )
    except Exception as e:
        # Clean up any widgets in the progress frame
        for widget in progress_frame.winfo_children():
            widget.destroy()

        ax.text(
            0.5,
            0.5,
            str(e),
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return (False, e)
    
    # Create two alternating colors for P and two for Z
    pcm = getattr(plt.cm, params["P_colormap"])
    zcm = getattr(plt.cm, params["Z_colormap"])
    spread = params["P_spread"]
    P_colors = [
        pcm(np.linspace(0.0, spread, params["n"])),
        pcm(np.linspace(spread, 2 * spread, params["n"])),
    ]
    spread = params["Z_spread"]
    Z_colors = [
        zcm(np.linspace(0.0, spread, params["n"])),
        zcm(np.linspace(spread, 2 * spread, params["n"])),
    ]

    # Stable contours
    contours = []
    contours_Z = []
    # Add first region (P0Z0)
    poly_Zero = list(zip(np.zeros_like(regions[0][0]), regions[0][-1])) + list(
        zip(regions[0][0][::-1], regions[0][-1][::-1])
    )
    contours_Z.append([poly_Zero])

    for i in range(len(regions)):
        if not (np.any(regions[i][2]) and np.any(regions[i][3])):
            continue
        region_left, region_right, mask = interpolate_masked_arrays_general(
            regions[i][0], regions[i][1], regions[i][2], regions[i][3]
        )
        poly_P = list(zip(region_left[mask], regions[i][-1][mask])) + list(
            zip(region_right[mask][::-1], regions[i][-1][mask][::-1])
        )
        contours.append([poly_P])
        if i != len(regions) - 1:
            # Interpolating Z is not as straightforward as the evaluation points are different between the bottom and the top.
            poly_Z = list(
                zip(regions[i][1][regions[i][3]], regions[i][-1][regions[i][3]])
            ) + list(
                zip(
                    regions[i + 1][0][regions[i + 1][2]][::-1],
                    regions[i + 1][-1][regions[i + 1][2]][::-1],
                )
            )
            contours_Z.append([poly_Z])
    # Stationary contours
    contours_s = []
    contours_Z_s = []
    poly_zero_s = list(zip(np.zeros_like(regions_no_stab[0][0]), regions_no_stab[0][-1])) + list(
        zip(regions_no_stab[0][0][::-1], regions_no_stab[0][-1][::-1])
    )
    contours_Z_s.append([poly_zero_s])
    for i in range(len(regions_no_stab)):
        poly_P_s = list(
            zip(regions_no_stab[i][0], regions_no_stab[i][-1])
        ) + list(
            zip(regions_no_stab[i][1][::-1], regions_no_stab[i][-1][::-1])
        )
        contours_s.append([poly_P_s])
        if i != len(regions_no_stab) - 1:
            poly_Z_s = list(
                zip(regions_no_stab[i][1], regions_no_stab[i][-1])
            ) + list(
                zip(regions_no_stab[i + 1][0][::-1], regions_no_stab[i + 1][-1][::-1])
            )
            contours_Z_s.append([poly_Z_s])
    # Alternate colors for P and Z regions
    colors = (
        [P_colors[i % 2][i] for i in range(len(contours_s))]
        + [plt.cm.Greys(0.9)]
        + [Z_colors[i % 2][i] for i in range(1, len(contours_Z_s))]
    )
    hat = None
    hatches = [None] * len(contours) + [hat] * len(contours_Z)
    _labels = [f"P_{i + 1}Z_{i}" for i, _ in enumerate(contours_s)] + [
        f"P_{i}Z_{i}" for i, _ in enumerate(contours_Z_s)
    ]
    hatches_s = ["\\" if i % 2 else "/" for i in range(len(contours_s))] + ["-" if i % 2 else "--" for i in range(len(contours_Z_s))]
    stab_P = len(contours)
    stab_Z = len(contours_Z)
    contours += contours_Z
    contours_s += contours_Z_s
    levels = list(range(len(contours_s) + 1))
    _cs_s = ContourSet(
        ax, levels, contours_s, filled=True, hatches=hatches_s,colors=colors, alpha=.5
    )
    # Create ContourSet with enhanced appearance
    _cs = ContourSet(
        ax, levels[:stab_P+1] + levels[params["n"]:params["n"]+stab_Z], contours, filled=True, hatches=hatches, colors=colors[:stab_P] + colors[params["n"]:params["n"]+stab_Z], alpha=1.0
    )


    # Create legend with the four color types
    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, facecolor=pcm(params["P_spread"] * 0.5), alpha=0.7),
        plt.Rectangle((0, 0), 1, 1, facecolor=pcm(params["P_spread"] * 1.5), alpha=0.7),
        plt.Rectangle((0, 0), 1, 1, facecolor=zcm(params["Z_spread"] * 0.5), alpha=0.7, hatch=hat),
        plt.Rectangle((0, 0), 1, 1, facecolor=zcm(params["Z_spread"] * 1.5), alpha=0.7, hatch=hat),
    ]
    legend_labels = ["P", "P", "Z", "Z"]
    _legend = ax.legend(
        legend_patches,
        legend_labels,
        loc="upper right",  # Position in top right
        bbox_to_anchor=(0.98, 0.98),  # Fine-tune position
        framealpha=0.7,  # Make background semi-transparent
        ncol=2,
    )  # Use 2 columns to make it more compact

    # Enhance grid and spines
    ax.grid(True, linestyle="--", alpha=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # Add labels with larger formatting
    ax.set_xlabel("Nt", fontsize=14, fontweight="bold")
    ax.set_ylabel(params["param_select"], fontsize=14, fontweight="bold")

    # Make tick marks more visible
    ax.tick_params(
        axis="both",  # Apply to both x and y axes
        direction="out",  # Ticks point outward
        length=5,  # Make them longer to be more visible
        width=1,  # Make them thicker
        which="major",  # Apply to major ticks
        grid_alpha=0.7,  # Grid line transparency
        zorder=3,  # Make sure ticks are drawn on top
    )

    # Make sure ticks are enabled
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    # Create a status text that will show the region
    status_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.7),
    )

    paths = [Path(contour[0]) for contour in contours_s]
    def on_mouse_move(event):
        if event.inaxes != ax:
            text = ""
        else:
            x, y = event.xdata, event.ydata
            text = ""

            for i, path in enumerate(paths):
                if path.contains_point((x, y)):
                    if i < len(contours_s) - len(contours_Z_s):
                        text = f"Solution type: #P={i + 1}, #Z={i}"
                    else:
                        z_idx = i - (len(contours_s) - len(contours_Z_s))
                        text = f"Solution type: #P={z_idx}, #Z={z_idx}"
                    break

        if status_text.get_text() != text:  # Only update if text changed
            status_text.set_text(text)
            fig.canvas.draw_idle()

    # Connect the event handler
    fig = ax.figure
    fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)

    return (True, paths)


def setup_buttons(params: dict[str, float], draw: Callable, parent: tk.Frame):
    text_width = 11
    box_width = 7

    # Create frame for parameters
    param_frame = tk.Frame(parent)
    param_frame.pack(fill=tk.BOTH, expand=True, padx=10)

    boxes: dict[str, tk.Widget] = {}

    def make_update_func(param, type=float):
        def update(event=None):
            # Only handle Ctrl+Enter for rendering
            if event and event.state & 0x4:  # 0x4 is the state flag for Control
                new_draw()

        def on_edit(event=None):
            entry = boxes[param]
            try:
                new_value = type(entry.get())
                if new_value != params[param]:
                    params[param] = new_value
                    entry.configure(bg="#90EE90")  # Light green for valid input
                elif (
                    entry.cget("bg") == "white"
                ):  # If value matches and background is white, keep it white
                    pass
                else:  # If value matches but background is not white (was previously changed), keep it green
                    entry.configure(bg="#90EE90")
            except ValueError:
                entry.configure(bg="#FFB6C6")  # Light red for invalid input

        return update, on_edit

    # Handle r parameter with slider
    r_frame = tk.Frame(param_frame)
    r_frame.pack(fill=tk.X, pady=2)
    tk.Label(r_frame, text="r", width=text_width).pack(side=tk.LEFT)

    # Create value label to show current value
    # r_value_label = tk.Label(r_frame, text=str(params["r"]), width=5)
    # r_value_label.pack(side=tk.RIGHT)

    # Create slider
    r_slider = tk.Scale(
        r_frame,
        from_=1,
        to=10,
        orient=tk.HORIZONTAL,
        resolution=0.1,
    )
    r_slider.set(params["r"])
    r_slider.pack(fill=tk.X, expand=True, padx=5)
    boxes["r"] = r_slider

    def on_slider_release(event):
        value = float(r_slider.get())
        params["r"] = value
        r_slider.config(troughcolor="#90EE90")  # Light green for confirmed change

    r_slider.bind(
        "<ButtonRelease-1>", on_slider_release
    )  # Only update when mouse is released

    # Group parameters by base name
    param_groups = {}
    for param, value in params.items():
        if param in ["r", "s_min", "s_max", "n", "m", "fmin", "fmax", "param_select", "P_colormap", "Z_colormap", "P_spread", "Z_spread", "calc_lyap"]:
            continue
        base = param.split("_")[0]
        if base not in param_groups:
            param_groups[base] = []
        param_groups[base].append((param, value))

    # Create shared radio button variable
    selected_param = tk.StringVar(value="delta_0")  # Set delta_0 as default

    # Create entries for each parameter group
    for base in param_groups:
        group_frame = tk.Frame(param_frame)
        group_frame.pack(fill=tk.X, pady=2)

        # Create horizontal layout for each parameter group
        param_row = tk.Frame(group_frame)
        param_row.pack(fill=tk.X)

        # Add label
        tk.Label(param_row, text=base, width=text_width).pack(side=tk.LEFT)

        # Add entries and radio buttons for each parameter
        for param, value in param_groups[base]:
            # Create frame for parameter pair
            param_pair = tk.Frame(param_row)
            param_pair.pack(side=tk.LEFT, padx=2)

            # Add entry
            entry = tk.Entry(param_pair, width=box_width)
            entry.insert(0, str(value))
            entry.configure(bg="white")
            entry.pack(side=tk.LEFT)
            boxes[param] = entry

            update_func, edit_func = make_update_func(param)
            entry.bind("<Return>", update_func)
            entry.bind("<KeyRelease>", edit_func)

            # Add radio button
            radio = tk.Radiobutton(param_pair, variable=selected_param, value=param)
            radio.pack(side=tk.LEFT)

    # Store selected parameter variable in params dict
    params["param_select"] = selected_param.get()
    selected_param.trace_add(
        "write", lambda *args: params.update({"param_select": selected_param.get()})
    )

    # Size range controls
    # Create frames for parameter pairs
    # First row: s min/max + n
    remainin_rows = [
        ("s min/max, n", [("s_min", float), ("s_max", float), ("n", int)]),
        ("f min/max, m", [("fmin", float), ("fmax", float), ("m", int)]),
    ]
    for label, prms in remainin_rows:
        frame = tk.Frame(param_frame)
        frame.pack(fill=tk.X, pady=2)
        tk.Label(frame, text=label, width=text_width).pack(side=tk.LEFT)
        for param, type in prms:
            entry = tk.Entry(frame, width=box_width)
            entry.insert(0, str(params[param]))
            entry.configure(bg="white")
            entry.pack(side=tk.LEFT, padx=2)
            boxes[param] = entry
            update_func, edit_func = make_update_func(param, type)
            entry.bind("<Return>", update_func)
            entry.bind("<KeyRelease>", edit_func)

    # Modify the draw function to reset colors
    def new_draw():
        # Reset all entry colors to white
        for entry in boxes.values():
            if isinstance(entry, tk.Entry):
                entry.configure(bg="white")
            if isinstance(entry, tk.Scale):
                entry.configure(troughcolor="white")
        draw()

    # Update render button with new draw function
    render_button = tk.Button(param_frame, text="Render", command=new_draw)
    render_button.pack(pady=10)
    boxes["render_button"] = render_button

    # Add explanatory text
    help_text = """
Inputs:
The select box is the varied parameter.
Its range is given by the fmin/fmax, 
m is the number of points in that range.
The s min/max is the size range,
n the divisions.
The remaining are input parameters.

Input colors:
• White - Unchanged value
• Green - Confirmed change
• Red - Invalid input

Plot colors:
The solid regions are stable regions,
the dashed regions are unstable.
The color map can be configured
in the Configure Colors button.

Shortcuts:
• Ctrl+Enter - To render

"""
    help_label = tk.Label(param_frame, text=help_text, justify=tk.LEFT, anchor="w")
    help_label.pack(pady=10, padx=5)

    return boxes

def open_color_config(root, params, draw_fn):
    color_window = tk.Toplevel(root)
    color_window.title("Color Configuration")
    color_window.geometry("400x300")

    def reverse_colormap(v: str) -> str:
        if v.endswith("_r"):
            return v[:-2]
        else:
            return v + "_r"

    # Get list of available colormaps
    colormaps = sorted([m for m in plt.colormaps() if not m.endswith("_r")])

    # P colormap selection
    p_frame = tk.Frame(color_window)
    p_frame.pack(fill=tk.X, padx=10, pady=5)
    tk.Label(p_frame, text="P Colormap:").pack(side=tk.LEFT)
    p_colormap = ttk.Combobox(p_frame, values=colormaps, state="readonly")
    p_colormap.set(params["P_colormap"])
    p_colormap.pack(side=tk.LEFT, padx=5)

    rev_button = tk.Button(
        p_frame,
        text="Reverse",
        command=lambda: p_colormap.set(reverse_colormap(p_colormap.get())),
    )
    rev_button.pack(side=tk.LEFT, padx=5)

    # P spread slider
    p_spread_frame = tk.Frame(color_window)
    p_spread_frame.pack(fill=tk.X, padx=10, pady=5)
    tk.Label(p_spread_frame, text="P Spread:").pack(side=tk.LEFT)
    p_spread = tk.Scale(
        p_spread_frame, from_=0, to=0.5, orient=tk.HORIZONTAL, resolution=0.01
    )
    p_spread.set(params["P_spread"])
    p_spread.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

    # Z colormap selection
    z_frame = tk.Frame(color_window)
    z_frame.pack(fill=tk.X, padx=10, pady=5)
    tk.Label(z_frame, text="Z Colormap:").pack(side=tk.LEFT)
    z_colormap = ttk.Combobox(z_frame, values=colormaps, state="readonly")
    z_colormap.set(params["Z_colormap"])
    z_colormap.pack(side=tk.LEFT, padx=5)

    rev_button = tk.Button(
        z_frame,
        text="Reverse",
        command=lambda: z_colormap.set(reverse_colormap(z_colormap.get())),
    )
    rev_button.pack(side=tk.LEFT, padx=5)

    # Z spread slider
    z_spread_frame = tk.Frame(color_window)
    z_spread_frame.pack(fill=tk.X, padx=10, pady=5)
    tk.Label(z_spread_frame, text="Z Spread:").pack(side=tk.LEFT)
    z_spread = tk.Scale(
        z_spread_frame, from_=0, to=0.5, orient=tk.HORIZONTAL, resolution=0.01
    )
    z_spread.set(params["Z_spread"])
    z_spread.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

    def apply_changes():
        params["P_colormap"] = p_colormap.get()
        params["Z_colormap"] = z_colormap.get()
        params["P_spread"] = float(p_spread.get())
        params["Z_spread"] = float(z_spread.get())
        draw_fn()
        color_window.destroy()

    # Apply button
    apply_btn = tk.Button(color_window, text="Apply", command=apply_changes)
    apply_btn.pack(pady=10)

def parameter_domains_interactive():
    root = tk.Tk()
    root.wm_title("Parameter Domains")

    # Create main container with left and right frames
    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Create left frame for plot and toolbar
    left_frame = tk.Frame(main_frame)
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Create right frame for parameters
    right_frame = tk.Frame(main_frame)
    right_frame.pack(side=tk.RIGHT, fill=tk.Y)

    # Create figure with tight layout
    fig = plt.Figure(tight_layout=False)
    ax = fig.add_subplot(111)

    # Create the matplotlib canvas (at the top)
    canvas = FigureCanvasTkAgg(fig, master=left_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # Create a frame for toolbar and progress bar
    bottom_frame = tk.Frame(left_frame)
    bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

    # Add navigation toolbar on the left side of bottom frame
    toolbar = NavigationToolbar2Tk(canvas, bottom_frame)
    toolbar.update()
    toolbar.pack(side=tk.LEFT)

    # Create progress frame on the right side of bottom frame
    progress_frame = tk.Frame(bottom_frame)
    progress_frame.pack(side=tk.RIGHT, fill=tk.X, padx=5)

    params = {
        "mu_0": 5.9,
        "mu_scale": -0.75,
        "lambda_0": 0.017,
        "lambda_scale": -0.6,
        "k_0": 1.0,
        "k_scale": 1.14,
        "g_0": 7.0,
        "g_scale": -0.75,
        "K_0": 1.0,
        "K_scale": 0.24,
        "gamma_0": 0.7,
        "gamma_scale": 0.24,
        "delta_0": 0.17,
        "delta_scale": -0.75,
        "r": 1.0,
        "s_min": 1.0,
        "s_max": 2.0,
        "n": 8,
        "m": 50,
        "param_select": "delta_0",
        "fmin": 0.1,
        "fmax": 2.0,
        # Color configuration parameters
        "P_colormap": "summer",
        "Z_colormap": "autumn",
        "P_spread": 0.3,
        "Z_spread": 0.3,
    }
    # Create a container for the ode variable
    plot_container = {"regions": None}
    
    def draw():
        try:
            ax.clear()
            success, value = draw_solution_regions_bin(ax, params, progress_frame)
            plot_container["regions"] = value
            # Let setup_jitcode decide if it needs to update
            fig.canvas.draw_idle()
            root.update_idletasks()
        except Exception as e:
            print(f"Error during draw: {e}")

    # Setup buttons in the right frame
    setup_buttons(params, draw, right_frame)

    # Add color configuration button
    color_config_btn = tk.Button(
        right_frame, text="Configure Colors", command=lambda : open_color_config(root, params, draw)
    )
    color_config_btn.pack(pady=5)

    # Set initial window size
    root.geometry("1200x800")

    draw()

    # Add window close handler
    def on_closing():
        plt.close(fig)  # Close the figure properly
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


parameter_domains_interactive()
