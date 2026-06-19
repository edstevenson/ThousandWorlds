"""Plotting helpers for ThousandWorlds data."""
from __future__ import annotations

import numpy as np

from .data import DataBundle, CSV_TO_INPUT

_AXIS_LABEL = {
    "T_star": "stellar temperature / K",
    "F_star": "incident stellar flux / W m$^{-2}$",
    "radius": "radius / m",
    "gravity": "gravity / m s$^{-2}$",
    "P_rot": "rotation period / days",
    "P0": "surface pressure / Pa",
    "CO2": "CO$_2$ volume fraction",
    "CH4": "CH$_4$ volume fraction",
}
_CBAR_LABEL = {
    "surface_temperature": "surface temperature / K",
    "temperature": "temperature / K",
    "specific_humidity": "specific humidity / kg kg$^{-1}$",
    "cloud_fraction": "cloud fraction",
    "asr": "ASR / W m$^{-2}$",
    "olr": "OLR / W m$^{-2}$",
    "u": "eastward wind / m s$^{-1}$",
    "v": "northward wind / m s$^{-1}$",
}
_FIELD_CMAP = {
    "surface_temperature": "RdYlBu_r",
    "temperature": "RdYlBu_r",
    "specific_humidity": "viridis",
    "asr": "inferno",
    "olr": "inferno",
    "cloud_fraction": "Blues_r",
    "u": "RdBu_r",
    "v": "RdBu_r",
}
# Vertical coordinate: relative-isobar sigma levels (level 0 ~ surface, last = TOA),
# matching the data build. The dataset is generated on a fixed 10-level grid; subsets
# with fewer levels (complete-obs has 9) simply drop the top level(s), so the first
# `nlev` of the 10-level grid are used.
_P_TOP = 1000.0                   # Pa
_BOTTOM_SQUEEZE_FRACTION = 0.925
_SIGMA_LEVELS = (lambda x: 0.75 * x + 1.75 * x**3 - 1.5 * x**4)(np.linspace(1.0, 0.0, 10))

# Base font size for figures; each plotting function applies it via _use_style().
_BASE_FONTSIZE = 13.0


def _use_style(scale: float = 1.0) -> None:
    import matplotlib.pyplot as plt

    s = _BASE_FONTSIZE * scale
    sizes = {
        "font.size": s,
        "axes.titlesize": s,
        "axes.labelsize": s,            # axis labels + colorbar labels
        "xtick.labelsize": 0.85 * s,
        "ytick.labelsize": 0.85 * s,
        "legend.fontsize": 0.85 * s,
        "figure.labelsize": s,          # supxlabel / supylabel
    }
    plt.rcParams.update({k: v for k, v in sizes.items() if k in plt.rcParams})


def compare_field(
    bundle: DataBundle,
    pred: np.ndarray,
    *,
    field: str = "surface_temperature",
    idx: int = 0,
    split: str = "test",
    cmap: str | None = None,
    residual_cmap: str = "RdBu_r",
    units: str | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Target / prediction / residual triptych for one field of one world.

    *pred* is an array aligned with ``Y_{split}`` (same shape).  Target and
    prediction share one colour scale; the residual (prediction - target) uses
    a diverging scale centred at zero, annotated with the area-weighted RMSE.
    """
    import matplotlib.pyplot as plt

    _use_style()
    fi = bundle.field_names.index(field)
    target = getattr(bundle, f"Y_{split}")[idx, fi]
    prediction = np.asarray(pred)[idx, fi]
    residual = prediction - target

    var = _field_to_variable(field)
    cmap = cmap or _FIELD_CMAP.get(var, "viridis")
    units = units or _CBAR_LABEL.get(var, field)
    lat, lon = _grid_edges(target.shape)
    vmin, vmax = np.nanmin([target, prediction]), np.nanmax([target, prediction])
    rmax = np.nanmax(np.abs(residual))

    fig, axes = plt.subplots(1, 3, figsize=figsize or (13, 3.0), constrained_layout=True)
    for ax, data, title in zip(axes[:2], (target, prediction), ("target", "prediction")):
        mesh = ax.pcolormesh(lon, lat, data, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(mesh, ax=axes[:2], location="bottom", shrink=0.8, aspect=40, label=units)
    axes[0].set_title("target")
    axes[1].set_title("prediction")

    rmse = _weighted_rmse(residual, _lat_weights(target.shape[0]))
    mesh = axes[2].pcolormesh(lon, lat, residual, shading="auto", cmap=residual_cmap, vmin=-rmax, vmax=rmax)
    axes[2].set_title(f"prediction $-$ target  (RMSE {rmse:.3g})")
    fig.colorbar(mesh, ax=axes[2], location="bottom", shrink=0.8, aspect=20, label=f"$\\Delta$ {units}")

    for ax in axes:
        ax.set(xlim=(-180, 180), xticks=[-180, 0, 180], yticks=[-60, 0, 60])
    axes[0].set_ylabel("latitude")
    return fig, axes


def temperature_profile(
    bundle: DataBundle,
    idx: int = 0,
    *,
    split: str = "train",
    profiles: tuple[str, ...] = ("global_mean", "substellar", "antistellar"),
    include_surface: bool = True,
    ax=None,
    title: str | None = None,
):
    """Vertical temperature profile(s) for a single world, on an inverted log-pressure axis.

    Stacks ``temperature_0..N`` for world ``idx`` and reduces to the requested
    ``profiles`` (``global_mean`` / ``substellar`` / ``antistellar``). With
    ``include_surface`` (default), the profile extends down to the surface at the
    world's ``P0`` using the ``surface_temperature`` field.
    """
    import matplotlib.pyplot as plt

    _use_style()
    Yrow = getattr(bundle, f"Y_{split}")[idx]
    P0 = float(getattr(bundle, f"meta_{split}")[CSV_TO_INPUT["P0"]].iloc[idx]) / 1.0e5  # Pa -> bar
    stack = _stack_levels(Yrow, bundle.field_names, "temperature", surface=include_surface)
    plev = _pressure_levels(P0, stack.shape[0] - int(include_surface), surface=include_surface) / 100.0  # hPa
    weights = _lat_weights(stack.shape[1])

    fig, ax = (plt.subplots(figsize=(4, 4), constrained_layout=True) if ax is None else (ax.figure, ax))
    for name in profiles:
        ax.plot(_profile(stack, weights, name), plev, marker="o", ms=3, label=name.replace("_", " "))
    ax.set(xlabel="temperature / K", ylabel="pressure / hPa")
    ax.set_yscale("log")
    ax.invert_yaxis()
    if len(profiles) > 1:
        ax.legend()
    if title is not None:
        ax.set_title(title)
    ax.grid(True, which="both", alpha=0.4)
    return fig, ax


def profile_comparison(
    bundle: DataBundle,
    pred: np.ndarray,
    variable: str = "temperature",
    idx: int = 0,
    *,
    split: str = "test",
    profiles: tuple[str, ...] = ("dayside", "nightside"),
    include_surface: bool = True,
    log_x: bool | None = None,
    ax=None,
    legend: bool = True,
    title: str | None = None,
):
    """Target (solid) vs prediction (dashed) vertical profiles for one world.

    ``pred`` is an array aligned with ``Y_{split}`` (e.g. a model's output). Each
    requested spatial profile is drawn in one colour, target solid and prediction
    dashed. For temperature the profile runs to the surface at P0; specific humidity
    defaults to a log x-axis.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _use_style()
    surf = include_surface and variable == "temperature"
    target = _stack_levels(getattr(bundle, f"Y_{split}")[idx], bundle.field_names, variable, surface=surf)
    prediction = _stack_levels(np.asarray(pred)[idx], bundle.field_names, variable, surface=surf)
    P0 = float(getattr(bundle, f"meta_{split}")[CSV_TO_INPUT["P0"]].iloc[idx]) / 1.0e5  # Pa -> bar
    plev = _pressure_levels(P0, target.shape[0] - int(surf), surface=surf) / 100.0  # hPa
    weights = _lat_weights(target.shape[1])
    log_x = (variable == "specific_humidity") if log_x is None else log_x

    fig, ax = (plt.subplots(figsize=(4, 4), constrained_layout=True) if ax is None else (ax.figure, ax))
    for i, name in enumerate(profiles):
        ax.plot(_profile(target, weights, name), plev, color=f"C{i}", marker="o", ms=3)
        ax.plot(_profile(prediction, weights, name), plev, color=f"C{i}", ls="--", marker="o", ms=3)
    ax.set(xlabel=_CBAR_LABEL.get(variable, variable), ylabel="pressure / hPa")
    ax.set_yscale("log")
    ax.invert_yaxis()
    if log_x:
        ax.set_xscale("log")
    if legend:
        handles = [Line2D([0], [0], color=f"C{i}", label=name.replace("_", " ")) for i, name in enumerate(profiles)]
        handles += [Line2D([0], [0], color="0.4", label="target"),
                    Line2D([0], [0], color="0.4", ls="--", label="prediction")]
        ax.legend(handles=handles)
    if title is not None:
        ax.set_title(title)
    ax.grid(True, which="both", alpha=0.4)
    return fig, ax


def field_map(
    bundle: DataBundle,
    idx: int = 0,
    field: str = "surface_temperature",
    *,
    split: str = "train",
    ax=None,
    cmap: str | None = None,
    colorbar: bool = True,
    units: str | None = None,
    title: str | None = None,
    **kwargs,
):
    """Lon-lat map of one field for one world. Extra kwargs (e.g. ``norm``, ``vmin``) go to pcolormesh."""
    import matplotlib.pyplot as plt

    _use_style()
    var = _field_to_variable(field)
    data = getattr(bundle, f"Y_{split}")[idx, bundle.field_names.index(field)]
    lat, lon = _grid_edges(data.shape)
    fig, ax = (plt.subplots(figsize=(6, 3.2), constrained_layout=True) if ax is None else (ax.figure, ax))
    mesh = ax.pcolormesh(lon, lat, data, shading="auto", cmap=cmap or _FIELD_CMAP.get(var, "viridis"), **kwargs)
    ax.set(xlim=(-180, 180), ylim=(-90, 90), xlabel="longitude", ylabel="latitude",
           xticks=[-180, -90, 0, 90, 180])
    if colorbar:
        fig.colorbar(mesh, ax=ax, label=units or _CBAR_LABEL.get(var, field))
    if title is not None:
        ax.set_title(title)
    return fig, ax


def ice_fraction_map(
    bundle: DataBundle,
    idx: int = 0,
    *,
    split: str = "train",
    freeze_K: float = 273.15,
    ax=None,
    title: str | None = None,
):
    """Surface ice map for one world: cells colder than ``freeze_K`` count as ice.

    Coloured blue (open water) to white (ice).
    """
    import matplotlib.pyplot as plt

    _use_style()
    ts = getattr(bundle, f"Y_{split}")[idx, bundle.field_names.index("surface_temperature")]
    ice = (ts < freeze_K).astype(float)
    lat, lon = _grid_edges(ts.shape)
    fig, ax = (plt.subplots(figsize=(6, 3.2), constrained_layout=True) if ax is None else (ax.figure, ax))
    mesh = ax.pcolormesh(lon, lat, ice, shading="auto", cmap="Blues_r", vmin=0.0, vmax=1.0)
    ax.set(xlim=(-180, 180), ylim=(-90, 90), xlabel="longitude", ylabel="latitude",
           xticks=[-180, -90, 0, 90, 180])
    fig.colorbar(mesh, ax=ax, label="ice fraction", ticks=[0, 1])
    if title is not None:
        ax.set_title(title)
    return fig, ax


def net_radiation_map(
    bundle: DataBundle,
    idx: int = 0,
    *,
    split: str = "train",
    cmap: str = "RdBu_r",
    ax=None,
    title: str | None = None,
):
    """Net top-of-atmosphere radiation (ASR - OLR) for one world.

    Positive is net heating (day side), negative net cooling (night side), on a
    diverging scale centred at zero.
    """
    import matplotlib.pyplot as plt

    _use_style()
    Y = getattr(bundle, f"Y_{split}")
    net = Y[idx, bundle.field_names.index("asr")] - Y[idx, bundle.field_names.index("olr")]
    lat, lon = _grid_edges(net.shape)
    vmax = float(np.nanmax(np.abs(net)))
    fig, ax = (plt.subplots(figsize=(6, 3.2), constrained_layout=True) if ax is None else (ax.figure, ax))
    mesh = ax.pcolormesh(lon, lat, net, shading="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set(xlim=(-180, 180), ylim=(-90, 90), xlabel="longitude", ylabel="latitude",
           xticks=[-180, -90, 0, 90, 180])
    fig.colorbar(mesh, ax=ax, label="ASR $-$ OLR / W m$^{-2}$")
    if title is not None:
        ax.set_title(title)
    return fig, ax


def zonal_cross_section(
    bundle: DataBundle,
    idx: int = 0,
    variable: str = "temperature",
    *,
    split: str = "train",
    include_surface: bool = True,
    ax=None,
    cmap: str | None = None,
):
    """Latitude x pressure cross-section (zonal mean) of a 3D variable for one world.

    For temperature the section extends to the surface at P0 (zonal-mean
    surface_temperature) unless ``include_surface=False``.
    """
    import matplotlib.pyplot as plt

    _use_style()
    Yrow = getattr(bundle, f"Y_{split}")[idx]
    P0 = float(getattr(bundle, f"meta_{split}")[CSV_TO_INPUT["P0"]].iloc[idx]) / 1.0e5  # Pa -> bar
    surf = include_surface and variable == "temperature"
    stack = _stack_levels(Yrow, bundle.field_names, variable, surface=surf)
    lat, _ = _grid(stack.shape[1:])
    plev = _pressure_levels(P0, stack.shape[0] - int(surf), surface=surf) / 100.0  # hPa
    fig, ax = (plt.subplots(figsize=(5, 3), constrained_layout=True) if ax is None else (ax.figure, ax))
    cf = ax.contourf(lat, plev, stack.mean(axis=2), levels=20, cmap=cmap or _FIELD_CMAP.get(variable, "viridis"))
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set(xlabel="latitude", ylabel="pressure / hPa")
    fig.colorbar(cf, ax=ax, label=_CBAR_LABEL.get(variable, variable))
    return fig, ax


def wind_map(
    bundle: DataBundle,
    idx: int = 0,
    *,
    level: int = 0,
    scalar: str = "surface_temperature",
    split: str = "train",
    ax=None,
    step: int = 2,
    title: str | None = None,
):
    """Wind vectors at a model level, quivered over a scalar background field."""
    import matplotlib.pyplot as plt

    _use_style()
    Y = getattr(bundle, f"Y_{split}")
    u = Y[idx, bundle.field_names.index(f"u_{level}")]
    v = Y[idx, bundle.field_names.index(f"v_{level}")]
    bg = Y[idx, bundle.field_names.index(scalar)]
    lat, lon = _grid(bg.shape)                 # centres, for the quiver
    lat_e, lon_e = _grid_edges(bg.shape)       # edges, so the background tiles to the poles
    fig, ax = (plt.subplots(figsize=(6, 3.2), constrained_layout=True) if ax is None else (ax.figure, ax))
    mesh = ax.pcolormesh(lon_e, lat_e, bg, shading="auto", cmap=_FIELD_CMAP.get(_field_to_variable(scalar), "viridis"))
    q = ax.quiver(lon[::step], lat[::step], u[::step, ::step], v[::step, ::step])
    ref = int(np.nanpercentile(np.hypot(u, v), 95)) or 1
    ax.quiverkey(q, 0.85, 1.06, ref, f"{ref} m s$^{{-1}}$", labelpos="E",
                 fontproperties={"size": plt.rcParams["xtick.labelsize"]})
    ax.set(xlim=(-180, 180), ylim=(-90, 90), xlabel="longitude", ylabel="latitude",
           xticks=[-180, -90, 0, 90, 180])
    fig.colorbar(mesh, ax=ax, label=_CBAR_LABEL.get(_field_to_variable(scalar), scalar))
    if title is not None:
        ax.set_title(title)
    return fig, ax


def wind_streamlines(
    bundle: DataBundle,
    idx: int = 0,
    *,
    level: int = 0,
    split: str = "train",
    cmap: str = "plasma",
    density: float = 2.0,
    ax=None,
    colorbar: bool = True,
    title: str | None = None,
):
    """Wind streamlines coloured by speed at a model level (no background field).

    Streamlines on a lon-lat axis, coloured by wind speed. The longitude seam is
    closed so lines wrap across +/-180.
    """
    import matplotlib.pyplot as plt

    _use_style()
    Y = getattr(bundle, f"Y_{split}")
    u = Y[idx, bundle.field_names.index(f"u_{level}")]
    v = Y[idx, bundle.field_names.index(f"v_{level}")]
    lat, lon = _grid(u.shape)

    # streamplot needs an evenly spaced grid: regrid Gaussian -> uniform latitude
    lat_u = np.linspace(lat[0], lat[-1], lat.size)
    u, v = _regrid_lat(u, lat, lat_u), _regrid_lat(v, lat, lat_u)
    # close the longitude seam so streamlines wrap across +/-180
    lon = np.append(lon, lon[0] + 360.0)
    u = np.concatenate([u, u[:, :1]], axis=1)
    v = np.concatenate([v, v[:, :1]], axis=1)

    fig, ax = (plt.subplots(figsize=(6, 3.2), constrained_layout=True) if ax is None else (ax.figure, ax))
    strm = ax.streamplot(lon, lat_u, u, v, color=np.hypot(u, v), cmap=cmap,
                         density=density, linewidth=1.0, arrowsize=0.8)
    ax.set(xlim=(-180, 180), ylim=(-90, 90), xlabel="longitude", ylabel="latitude",
           xticks=[-180, -90, 0, 90, 180])
    if colorbar:
        fig.colorbar(strm.lines, ax=ax, label="wind speed / m s$^{-1}$")
    if title is not None:
        ax.set_title(title)
    return fig, ax


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _stack_levels(Yrow: np.ndarray, field_names: list[str], variable: str, *, surface: bool = False) -> np.ndarray:
    """Stack ``variable_0..N`` planes from one world's field vector into (level, lat, lon).

    With ``surface``, prepend the ``surface_temperature`` plane as the bottom level.
    """
    levels = sorted(
        (int(name.rsplit("_", 1)[1]), Yrow[i])
        for i, name in enumerate(field_names)
        if name.startswith(f"{variable}_") and name.rsplit("_", 1)[1].isdigit()
    )
    arrays = [plane for _, plane in levels]
    if surface:
        arrays.insert(0, Yrow[field_names.index("surface_temperature")])
    return np.stack(arrays, axis=0)


def _pressure_levels(P0: float, nlev: int, *, surface: bool = False) -> np.ndarray:
    """Model-level pressures (Pa) for surface pressure ``P0`` in bar (as in the emulator
    inputs / roxce), truncated to ``nlev``. With ``surface``, prepend ``P0`` as the bottom level.
    """
    P0_pa = P0 * 1.0e5
    plev = (_SIGMA_LEVELS * (_BOTTOM_SQUEEZE_FRACTION * P0_pa - _P_TOP) + _P_TOP)[:nlev]
    return np.concatenate([[P0_pa], plev]) if surface else plev


def _profile(stack: np.ndarray, weights: np.ndarray, which: str) -> np.ndarray:
    """One vertical profile from a (level, lat, lon) stack."""
    nlat, nlon = stack.shape[1:]
    equator = [nlat // 2 - 1, nlat // 2]
    lon = np.linspace(-180.0 + 180.0 / nlon, 180.0 - 180.0 / nlon, nlon)  # substellar at 0
    if which == "global_mean":
        return np.average(stack.mean(axis=2), axis=1, weights=weights)
    if which == "substellar":  # lon 0 (centre column), equator
        return stack[:, equator, nlon // 2].mean(axis=1)
    if which == "antistellar":  # lon -180 (edge column), equator
        return stack[:, equator, 0].mean(axis=1)
    if which == "dayside":  # |lon| < 90 of substellar, area-weighted
        return np.average(stack[:, :, np.abs(lon) < 90.0].mean(axis=2), axis=1, weights=weights)
    if which == "nightside":
        return np.average(stack[:, :, np.abs(lon) >= 90.0].mean(axis=2), axis=1, weights=weights)
    raise ValueError(f"unknown profile {which!r}")


def _grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    nlat, nlon = shape
    lat = np.degrees(np.arcsin(np.polynomial.legendre.leggauss(nlat)[0]))
    lon = np.linspace(-180.0 + 180.0 / nlon, 180.0 - 180.0 / nlon, nlon)  # cell-centred; cells tile [-180, 180]
    return lat, lon


def _grid_edges(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Cell *edges* (lat, lon) for pcolormesh so cells tile [-90, 90] x [-180, 180].

    Longitude edges are uniform; latitude edges are the area-conserving Gaussian-grid
    boundaries (cumulative quadrature weights in sin-latitude), so they reach the poles
    and stay consistent with _lat_weights. Pass with ``shading="auto"`` (-> flat).
    """
    nlat, nlon = shape
    mu = np.concatenate([[-1.0], -1.0 + np.cumsum(_lat_weights(nlat))])
    lat = np.degrees(np.arcsin(np.clip(mu, -1.0, 1.0)))
    lon = np.linspace(-180.0, 180.0, nlon + 1)
    return lat, lon


def _regrid_lat(field: np.ndarray, lat_src: np.ndarray, lat_dst: np.ndarray) -> np.ndarray:
    """Linearly interpolate a (lat, lon) field from one latitude axis to another."""
    return np.stack([np.interp(lat_dst, lat_src, col) for col in field.T], axis=1)


def _lat_weights(nlat: int) -> np.ndarray:
    """Gaussian-quadrature latitude (area) weights for a T-grid field."""
    return np.polynomial.legendre.leggauss(nlat)[1]


def _weighted_rmse(diff: np.ndarray, lat_weights: np.ndarray) -> float:
    w = np.broadcast_to(lat_weights[:, None], diff.shape)
    finite = np.isfinite(diff)
    return float(np.sqrt(np.sum(w[finite] * diff[finite] ** 2) / np.sum(w[finite])))


def _field_to_variable(name: str) -> str:
    parts = name.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else name
