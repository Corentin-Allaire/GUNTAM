import glob
import argparse
from typing import List, Optional
import numpy as np
import pandas as pd

# Define the masks as constants
kVolumeMask = 0xFF00000000000000
kBoundaryMask = 0x00FF000000000000
kLayerMask = 0x0000FFF000000000
kApproachMask = 0x0000000FF0000000
kSensitiveMask = 0x000000000FFFFF00
kExtraMask = 0x00000000000000FF


def extract_masked_values(value: int) -> tuple[int, int, int, int]:
    """Extract geometry components from a packed geometry_id.

    Args:
        value: Packed integer geometry identifier.

    Returns:
        A 4-tuple (volume, layer, sensitive, extra) as integers.
    """
    volume: int = (value & kVolumeMask) >> 56
    layer: int = (value & kLayerMask) >> 36
    sensitive: int = (value & kSensitiveMask) >> 8
    extra: int = value & kExtraMask
    return (int(volume), int(layer), int(sensitive), int(extra))


def _create_particle_id_column(hits: pd.DataFrame, particles: pd.DataFrame) -> None:
    """Create a 'particle_id' column in the hits and particles DataFrames
       based on the five existing particle ID columns.

    Args:
        hits: DataFrame containing hit information
        particles: DataFrame containing particle information
    """

    id_columns = [
        "particle_id_pv",
        "particle_id_sv",
        "particle_id_part",
        "particle_id_gen",
        "particle_id_subpart",
    ]

    # Ensure particle_id exists and is stable
    particles["particle_id"] = particles.index.astype(int)

    # Build composite keys (tuple of the 5 values)
    particles["_key"] = list(map(tuple, particles[id_columns].to_numpy()))
    hits["_key"] = list(map(tuple, hits[id_columns].to_numpy()))

    # Create mapping: composite_key -> particle_id
    key_to_particle_id = dict(zip(particles["_key"], particles["particle_id"]))

    # Assign particle_id to hits
    hits["particle_id"] = hits["_key"].map(key_to_particle_id)

    # Optional cleanup
    particles.drop(columns="_key", inplace=True)
    hits.drop(columns="_key", inplace=True)


def _process_hits_data(data: pd.DataFrame, R_max: float = 500, Z_max: float = 1000) -> pd.DataFrame:
    """Process hits data, applying spatial filters and calculating eta and phi.

    Args:
        data: Raw hits DataFrame
        R_max: Maximum radial distance filter (default: 500)
        Z_max: Maximum z-coordinate filter (default: 1000)

    Returns:
        Processed hits DataFrame with computed r, eta, phi and geometry fields
    """
    # Drop unnecessary columns early to reduce memory usage
    columns_to_keep = [
        "particle_id",
        "particle_id_pv",
        "particle_id_sv",
        "particle_id_part",
        "particle_id_gen",
        "particle_id_subpart",
        "geometry_id",
        "tx",
        "ty",
        "tz",
    ]
    existing_columns = [col for col in columns_to_keep if col in data.columns]
    data = data[existing_columns].copy()

    # Add hit_id before any filtering to maintain original indices
    data["hit_id"] = data.index.astype(int)

    # Check for duplicates based on spatial coordinates and remove them early to reduce data size
    duplicate_subset = [c for c in ["tx", "ty", "tz"] if c in data.columns]
    if duplicate_subset:
        pre_dup_count = len(data)
        data = data.drop_duplicates(subset=duplicate_subset, keep="first").copy()
        removed = pre_dup_count - len(data)
        if removed > 0:
            print(
                f"[preprocess] Removed {removed} duplicate hits ("
                f"{removed / pre_dup_count:.2%} of raw hits) based on {duplicate_subset}"
            )

    # Apply spatial filters early to reduce data size
    spatial_mask = (data["tz"] < Z_max) & (data["tz"] > -1 * Z_max) & ((data["tx"] ** 2 + data["ty"] ** 2) < R_max**2)
    data = data[spatial_mask].copy()

    # Vectorized calculations for better performance
    tx_sq = data["tx"] ** 2
    ty_sq = data["ty"] ** 2
    tz_sq = data["tz"] ** 2
    data["r"] = np.sqrt(tx_sq + ty_sq)
    data["d"] = np.sqrt(tx_sq + ty_sq + tz_sq)

    # Compute eta and phi using vectorized operations
    rho = np.sqrt(tx_sq + ty_sq)
    theta = np.arctan2(rho, data["tz"])
    data["eta"] = -np.log(np.tan(theta / 2))
    data["phi"] = np.arctan2(data["ty"], data["tx"])

    # Rename tx, ty, tz to x, y, z for clarity
    data.rename(columns={"tx": "x", "ty": "y", "tz": "z"}, inplace=True)

    # Add space point based collumns
    data["varR"] = 0
    data["varZ"] = 0
    data["badSP"] = 0

    # Sort by distance from origin
    data = data.sort_values("d", ascending=True)
    data = data.drop(columns=["d"])

    # Use the bit map to extract the volume, layer, sensitive and extra values
    data["volume"], data["layer"], data["sensitive"], data["extra"] = zip(*data["geometry_id"].map(extract_masked_values))
    data = data.drop(columns=["geometry_id"])

    return data


def _process_particles_data(particles: pd.DataFrame, valid_particle_ids: pd.Index) -> pd.DataFrame:
    """Compute pT, eta, and phi for particles.

    Args:
        particles: Raw particles DataFrame
        valid_particle_ids: Valid particle IDs to keep

    Returns:
        Processed particles DataFrame with computed pT, eta, phi (filters out zero pz)
    """
    # Filter particles early
    particles = particles[particles["particle_id"].isin(valid_particle_ids) & (particles["pz"] != 0)].copy()

    if len(particles) == 0:
        return particles

    # Vectorized calculations
    px_sq = particles["px"] ** 2
    py_sq = particles["py"] ** 2
    particles["pT"] = np.sqrt(px_sq + py_sq)

    # Compute eta and phi for particles
    p_rho = np.sqrt(px_sq + py_sq)
    p_theta = np.arctan2(p_rho, particles["pz"])
    particles["eta"] = -np.log(np.tan(p_theta / 2))
    particles["phi"] = np.arctan2(particles["py"], particles["px"])
    particles["d0"] = particles["vx"] ** 2 + particles["vy"] ** 2
    particles["z0"] = particles["vz"]

    return particles


def _process_space_points_data(space_points: pd.DataFrame, hit_measurement_map: pd.DataFrame, hits: pd.DataFrame) -> pd.DataFrame:
    """Process space point data, merging with hit measurement map and hits to compute r, eta, phi.

    Args:
        space_points: Raw space points DataFrame
        hit_measurement_map: DataFrame mapping space points to hits
        hits: Processed hits DataFrame
    Returns:
        Processed space points DataFrame with computed r, eta, phi and geometry fields
    """

    columns_to_keep = [
        "measurement_id_1",
        "measurement_id_2",
        "x",
        "y",
        "z",
        "var_r",
        "var_z",
    ]

    existing_columns = [col for col in columns_to_keep if col in space_points.columns]
    space_points = space_points[existing_columns]
    # Merge to get hit_id for measurement_id_1
    space_points = (
        space_points.merge(
            hit_measurement_map, left_on="measurement_id_1", right_on="measurement_id", how="left", suffixes=("", "_1")
        )
        .rename(columns={"hit_id": "hit_id_1"})
        .drop(columns=["measurement_id"])
    )

    # Drop space points where hit_id_1 is NaN (not mapped to any hit)
    space_points = space_points.dropna(subset=["hit_id_1"])

    # Merge to get hit_id for measurement_id_2
    space_points = (
        space_points.merge(
            hit_measurement_map, left_on="measurement_id_2", right_on="measurement_id", how="left", suffixes=("", "_2")
        )
        .rename(columns={"hit_id": "hit_id_2"})
        .drop(columns=["measurement_id"])
    )

    # Merge with hits to get particle ID columns and geometry for hit_id_1
    hit_columns_1 = [
        "event_id",
        "hit_id",
        "particle_id",
        "particle_id_pv",
        "particle_id_sv",
        "particle_id_part",
        "particle_id_gen",
        "particle_id_subpart",
        "volume",
        "layer",
        "sensitive",
        "extra",
    ]
    space_points = space_points.merge(
        hits[hit_columns_1],
        left_on="hit_id_1",
        right_on="hit_id",
        how="left",
        suffixes=("", "_1"),
    ).drop(columns=["hit_id"])

    # Merge with hits to get particle_id for hit_id_2
    space_points = space_points.merge(
        hits[["hit_id", "particle_id"]],
        left_on="hit_id_2",
        right_on="hit_id",
        how="left",
        suffixes=("", "_2"),
    ).drop(columns=["hit_id"])

    # Check if particle IDs match, mark as bad if they don't
    space_points["badSP"] = (space_points["particle_id"] != space_points["particle_id_2"]).astype(int)
    space_points["badSP"] = (~space_points["particle_id_2"].isna()).astype(int)

    # Compute r, eta, phi from space_point coordinates (x, y, z)
    x = space_points["x"]
    y = space_points["y"]
    z = space_points["z"]

    x_sq = x**2
    y_sq = y**2
    z_sq = z**2
    space_points["r"] = np.sqrt(x_sq + y_sq)
    space_points["d"] = np.sqrt(x_sq + y_sq + z_sq)

    space_points = space_points.sort_values("d", ascending=True)
    space_points = space_points.drop(columns=["d"])

    rho = np.sqrt(x_sq + y_sq)
    theta = np.arctan2(rho, z)
    space_points["eta"] = -np.log(np.tan(theta / 2))
    space_points["phi"] = np.arctan2(y, x)

    # Rename varR and varZ to var_r and var_z
    space_points.rename(columns={"var_r": "varR", "var_z": "varZ"}, inplace=True)
    space_points.rename(columns={"hit_id_1": "hit_id"}, inplace=True)
    # Drop columns with _1 and _2 suffixes
    space_points = space_points.drop(columns=[col for col in space_points.columns if col.endswith("_1") or col.endswith("_2")])

    # Drop space points that don't have an event_id (failed to merge with hits)
    space_points = space_points.dropna(subset=["event_id"])

    return space_points


def read_acts_csv(args: argparse.Namespace) -> None:
    """Preprocess CSV data from ACTS G4 simulation with the ODD detector.

    This function reads hits, particles, and optionally space points data from
    specified directories, preprocesses them, and combines them into output files.

    Processing includes:
    - Spatial filtering (R_max, Z_max)
    - Particle ID mapping and filtering by minimum hit count
    - Computation of r, eta, phi for hits, particles, and space points
    - Geometry ID unpacking (volume, layer, sensitive, extra)
    - Duplicate hit removal
    - Space point validation and particle matching

    Args:
        args: Namespace containing:
            - input_path: Base path to input data directories
            - dir_start, dir_end: Optional directory range (odd_full_chain_N)
            - file_number: Optional suffix for output files
            - use_space_point: If True, process space points instead of raw hits
            - min_hits_per_particle: Minimum hits required per particle (default: 9)
            - output_format: List of output formats ('csv', 'h5', or both)

    Outputs:
        CSV and/or H5 files containing processed particles and hits/space points.
    """

    # Determine output suffix
    file_suffix = f"_{args.file_number}" if args.file_number is not None else ""

    # Build directory pattern based on range
    if args.dir_start is not None and args.dir_end is not None:
        # User specified a range - collect files from each directory in range
        hit_files = []
        particle_files = []
        space_point_files = []
        hit_measurement_map_files = []

        for dir_num in range(args.dir_start, args.dir_end + 1):
            dir_pattern = f"{args.input_path}/odd_full_chain_{dir_num}"
            hit_files.extend(sorted(glob.glob(f"{dir_pattern}/event*-hits.csv")))
            particle_files.extend(sorted(glob.glob(f"{dir_pattern}/event*-particles_selected.csv")))
            if args.use_space_point:
                space_point_files.extend(sorted(glob.glob(f"{dir_pattern}/event*-spacepoint.csv")))
                hit_measurement_map_files.extend(sorted(glob.glob(f"{dir_pattern}/event*-measurement-simhit-map.csv")))

        print(f"Processing directories: odd_full_chain_{args.dir_start} to odd_full_chain_{args.dir_end}")
    else:
        # Default behavior - process all odd* directories
        hit_files = sorted(glob.glob(f"{args.input_path}/odd*/event*-hits.csv"))
        particle_files = sorted(glob.glob(f"{args.input_path}/odd*/event*-particles_selected.csv"))
        if args.use_space_point:

            space_point_files = sorted(glob.glob(f"{args.input_path}/odd*/event*-spacepoint.csv"))
            hit_measurement_map_files = sorted(glob.glob(f"{args.input_path}/odd*/event*-measurement-simhit-map.csv"))
        else:
            space_point_files = []
            hit_measurement_map_files = []

        print(f"Processing all directories matching: {args.input_path}/odd*")

    print(
        f"Found {len(hit_files)} hit files, "
        f"{len(particle_files)} particle files, "
        f"{len(space_point_files) if args.use_space_point else 'N/A'} space point files, and "
        f"{len(hit_measurement_map_files) if args.use_space_point else 'N/A'} hit measurement map files."
    )

    total_files = len(hit_files)

    # Pre-allocate lists for better performance
    all_data: List[Optional[pd.DataFrame]] = [None] * total_files
    all_particles: List[Optional[pd.DataFrame]] = [None] * total_files
    all_space_points: List[Optional[pd.DataFrame]] = [None] * total_files if args.use_space_point else []

    # Create iterator based on whether space points are used
    if args.use_space_point:
        file_iterator = zip(hit_files, particle_files, space_point_files, hit_measurement_map_files)
    else:
        # When not using space points, create dummy None values for space_point and hit_measurement files
        file_iterator = zip(hit_files, particle_files, [None] * len(hit_files), [None] * len(hit_files))  # type: ignore

    for counter, (file, particle_file, space_point_file, hit_measurement_map_file) in enumerate(file_iterator):
        # Progress reporting
        if counter % 10 == 0 or counter == total_files - 1:
            print(f"Processing event {counter} / {total_files} ({counter / total_files * 100:.1f}%)")

        # Read CSV files
        data = pd.read_csv(file, dtype={"geometry_id": np.int64})
        particles = pd.read_csv(particle_file, dtype={"particle_id": np.int64})

        if args.use_space_point:
            space_points = pd.read_csv(space_point_file)
            hit_measurement_map = pd.read_csv(hit_measurement_map_file)

        # From the id barcode create a single particle_id column in both hits and particles dataframes
        _create_particle_id_column(data, particles)

        # Filter particles with IDs associated with at less than the minimum number of hits (default: 9)
        particle_counts = data["particle_id"].value_counts()
        valid_particle_ids = particle_counts[particle_counts >= args.min_hits_per_particle].index

        # Process each dataset with optimized functions
        data = _process_hits_data(data)
        particles = _process_particles_data(particles, valid_particle_ids)

        # Add event ID to all datasets
        data["event_id"] = counter
        particles["event_id"] = counter

        # Set particle_id to -1 for hits that don't match any remaining particle after filtering
        # This handles both NaN values and particle_ids that were filtered out during processing
        particle_ids = set(particles["particle_id"].values)
        mask_id = data["particle_id"].isna() | ~data["particle_id"].isin(particle_ids)
        data.loc[mask_id, "particle_id"] = -1

        if args.use_space_point:
            space_points = _process_space_points_data(space_points, hit_measurement_map, data)

        # Assign to pre-allocated lists instead of appending
        all_data[counter] = data
        all_particles[counter] = particles
        if args.use_space_point:
            all_space_points[counter] = space_points

    print("Concatenating all data...")

    # Concatenate all DataFrames at once - much more efficient
    full_data = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    full_particles = pd.concat(all_particles, ignore_index=True) if all_particles else pd.DataFrame()
    if args.use_space_point:
        full_space_points = pd.concat(all_space_points, ignore_index=True) if all_space_points else pd.DataFrame()

    print("Concatenation completed")

    # Print statistics
    print("Final data shapes:")
    print(f"  Hits: {full_data.shape}")
    print(f"  Particles: {full_particles.shape}")
    if args.use_space_point:
        print(f"  Space Points: {full_space_points.shape}")

    print("Writing output files...")

    # Write the new files with optional numbering
    hits_filename = f"{args.input_path}/hits_small{file_suffix}.csv"
    particles_filename = f"{args.input_path}/particles_small{file_suffix}.csv"
    space_points_filename = f"{args.input_path}/space_points_small{file_suffix}.csv"
    hdf_filename = f"{args.input_path}/processed_data{file_suffix}.h5"

    files_written = []

    # Write CSV files if requested
    if "csv" in args.output_format:
        full_particles.to_csv(particles_filename, index=False)
        files_written.extend([particles_filename])
        if args.use_space_point:
            full_space_points.to_csv(space_points_filename, index=False)
            files_written.append(space_points_filename)
        else:
            full_data.to_csv(hits_filename, index=False)
            files_written.append(hits_filename)

    # Write H5 files if requested
    if "h5" in args.output_format:
        with pd.HDFStore(hdf_filename, mode="w") as store:
            store.put("particles", full_particles, format="table")
            if args.use_space_point:
                store.put("space_points", full_space_points, format="table")
            else:
                store.put("hits", full_data, format="table")
        files_written.append(hdf_filename)

    print("Files written successfully:")
    for filename in files_written:
        print(f"  {filename}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Preprocess ACTS simulation data")
    parser.add_argument(
        "--file-number",
        type=int,
        default=None,
        help="Optional number to append to output filenames (e.g., --file-number 1 produces hits_small_1.csv)",
    )
    parser.add_argument(
        "--dir-start",
        type=int,
        default=None,
        help="Starting directory number (inclusive) for odd_full_chain_N directories",
    )
    parser.add_argument(
        "--dir-end", type=int, default=None, help="Ending directory number (inclusive) for odd_full_chain_N directories"
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="/data/atlas/callaire/Acts/ODD_data",
        help="Base path to input data directory (default: /data/atlas/callaire/Acts/ODD_data)",
    )
    parser.add_argument("--use-space-point", action="store_true", help="If set, use space point data instead of hit data")

    parser.add_argument(
        "--min-hits-per-particle",
        type=int,
        default=9,
        help="Minimum number of hits required for a particle to be included (default: 9)",
    )

    parser.add_argument(
        "--output-format",
        nargs="+",
        default=["csv"],
        choices=["csv", "h5"],
        help="Output file format(s): 'csv' (default), 'h5', or both (e.g., --output-format csv h5)",
    )

    args = parser.parse_args()
    read_acts_csv(args)
