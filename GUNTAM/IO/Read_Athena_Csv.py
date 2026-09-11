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


def _create_particle_id_column(target: pd.DataFrame, particles: pd.DataFrame) -> None:
    """Create a 'particle_id' column in `target` and `particles` DataFrames
       based on the five existing particle ID columns.

    Args:
        target: DataFrame containing the same five particle ID columns as
            `particles` (e.g. the measurement-to-particle map)
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
    target["_key"] = list(map(tuple, target[id_columns].to_numpy()))

    # Create mapping: composite_key -> particle_id
    key_to_particle_id = dict(zip(particles["_key"], particles["particle_id"]))

    # Assign particle_id to target
    target["particle_id"] = target["_key"].map(key_to_particle_id).astype(int)

    # Optional cleanup
    particles.drop(columns="_key", inplace=True)
    target.drop(columns="_key", inplace=True)


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


def _process_space_points_data(space_points: pd.DataFrame, measurement_particles_map: pd.DataFrame) -> pd.DataFrame:
    """Process space point data, merging directly with the measurement-to-particle map
       to compute r, eta, phi and geometry fields.


    Args:
        space_points: Raw space points DataFrame
        measurement_particles_map: DataFrame mapping measurement_id to particle IDs
            (must already contain a computed 'particle_id' column)
    Returns:
        Processed space points DataFrame with computed r, eta, phi and geometry fields
    """

    columns_to_keep = [
        "measurement_id_1",
        "measurement_id_2",
        "geometry_id_1",
        "geometry_id_2",
        "x",
        "y",
        "z",
        "var_r",
        "var_z",
    ]

    existing_columns = [col for col in columns_to_keep if col in space_points.columns]
    space_points = space_points[existing_columns]

    map_columns = [
        "measurement_id",
        "particle_id",
        "particle_id_pv",
        "particle_id_sv",
        "particle_id_part",
        "particle_id_gen",
        "particle_id_subpart",
    ]
    existing_map_columns = [col for col in map_columns if col in measurement_particles_map.columns]
    measurement_particles_map = measurement_particles_map[existing_map_columns]

    # Merge to get particle info (particle_id + the 5 raw id columns) for measurement_id_1
    space_points = space_points.merge(
        measurement_particles_map,
        left_on="measurement_id_1",
        right_on="measurement_id",
        how="left",
        suffixes=("", "_1"),
    ).drop(columns=["measurement_id"])

    # Classify space points with no measurement_id_1 match as orphans instead of dropping them
    space_points["particle_id"] = space_points["particle_id"].fillna(-1)
    measurement_particles_map["measurement_id"] = measurement_particles_map["measurement_id"].astype(np.float64)
    # Merge to get particle_id for measurement_id_2
    space_points = space_points.merge(
        measurement_particles_map[["measurement_id", "particle_id"]],
        left_on="measurement_id_2",
        right_on="measurement_id",
        how="left",
        suffixes=("", "_2"),
    ).drop(columns=["measurement_id"])

    # Check if particle IDs match, mark as bad if they don't
    space_points["badSP"] = (space_points["particle_id"] != space_points["particle_id_2"]).astype(int)
    space_points["badSP"] = (~space_points["particle_id_2"].isna()).astype(int)

    # Use the bit map to extract the volume, layer, sensitive and extra values directly
    # from the space point's own geometry_id (no need to go through hits)
    if "geometry_id_1" in space_points.columns:
        space_points["volume"], space_points["layer"], space_points["sensitive"], space_points["extra"] = zip(
            *space_points["geometry_id_1"].map(extract_masked_values)
        )
        space_points = space_points.drop(columns=["geometry_id_1"])
    if "geometry_id_2" in space_points.columns:
        space_points = space_points.drop(columns=["geometry_id_2"])

    # Compute r, eta, phi from space_point coordinates (x, y, z)
    x = space_points["x"]
    y = space_points["y"]
    z = space_points["z"]

    x_sq = x**2
    y_sq = y**2
    z_sq = z**2
    space_points["r"] = np.sqrt(x_sq + y_sq)
    space_points["d"] = np.sqrt(x_sq + y_sq + z_sq)

    space_points = space_points.sort_values("r", ascending=True)
    space_points = space_points.drop(columns=["d"])

    rho = np.sqrt(x_sq + y_sq)
    theta = np.arctan2(rho, z)
    space_points["eta"] = -np.log(np.tan(theta / 2))
    space_points["phi"] = np.arctan2(y, x)

    # Rename var_r and var_z to varR and varZ
    space_points.rename(columns={"var_r": "varR", "var_z": "varZ"}, inplace=True)

    # Drop columns with _2 suffix (only needed for badSP computation)
    space_points = space_points.drop(columns=[col for col in space_points.columns if col.endswith("_2")])

    return space_points


def read_athena_csv(args: argparse.Namespace) -> None:
    """Preprocess CSV data from Athena simulation.

    This function reads particles, space points, and a measurement-to-particle map
    from specified directories, preprocesses them, and combines them into output files.

    Processing includes:
    - Particle ID mapping and filtering by minimum measurement count
    - Computation of pT, eta, phi for particles
    - Computation of r, eta, phi for space points
    - Geometry ID unpacking (volume, layer, sensitive, extra)
    - Space point validation and particle matching

    Args:
        args: Namespace containing:
            - input_path: Base path to input data directories
            - dir_start, dir_end: Optional directory range (odd_full_chain_N)
            - file_number: Optional suffix for output files
            - min_hits_per_particle: Minimum measurements required per particle (default: 9)
            - output_format: List of output formats ('csv', 'h5', or both)

    Outputs:
        CSV and/or H5 files containing processed particles and space points.
    """

    # Determine output suffix
    file_suffix = f"_{args.file_number}" if args.file_number is not None else ""

    # Build directory pattern based on range
    if args.dir_start is not None and args.dir_end is not None:
        # User specified a range - collect files from each directory in range
        particle_files = []
        space_point_files = []
        measurement_particles_map_files = []

        for dir_num in range(args.dir_start, args.dir_end + 1):
            dir_pattern = f"{args.input_path}/itk_athena_{dir_num}"
            particle_files.extend(sorted(glob.glob(f"{dir_pattern}/event*-particles.csv")))
            space_point_files.extend(sorted(glob.glob(f"{dir_pattern}/event*-spacepoint.csv")))
            measurement_particles_map_files.extend(sorted(glob.glob(f"{dir_pattern}/event*-measurement-particles-map.csv")))

        print(f"Processing directories: odd_full_chain_{args.dir_start} to odd_full_chain_{args.dir_end}")
    else:
        # Default behavior - process all odd* directories
        particle_files = sorted(glob.glob(f"{args.input_path}/itk*/event*-particles.csv"))
        space_point_files = sorted(glob.glob(f"{args.input_path}/itk*/event*-spacepoint.csv"))
        measurement_particles_map_files = sorted(glob.glob(f"{args.input_path}/itk*/event*-measurement-particles-map.csv"))

        print(f"Processing all directories matching: {args.input_path}/itk*")

    print(
        f"Found {len(particle_files)} particle files, "
        f"{len(space_point_files)} space point files, and "
        f"{len(measurement_particles_map_files)} measurement-particles map files."
    )

    total_files = len(particle_files)

    # Pre-allocate lists for better performance
    all_particles: List[Optional[pd.DataFrame]] = [None] * total_files
    all_space_points: List[Optional[pd.DataFrame]] = [None] * total_files

    file_iterator = zip(particle_files, space_point_files, measurement_particles_map_files)

    for counter, (particle_file, space_point_file, measurement_particles_map_file) in enumerate(file_iterator):
        # Progress reporting
        if counter % 10 == 0 or counter == total_files - 1:
            print(f"Processing event {counter} / {total_files} ({counter / total_files * 100:.1f}%)")

        # Read CSV files
        particles = pd.read_csv(particle_file, dtype={"particle_id": np.int64})
        space_points = pd.read_csv(
            space_point_file,
        )

        measurement_particles_map = pd.read_csv(measurement_particles_map_file, dtype={"measurement_id": np.int64})

        # From the id barcode create a single particle_id column in both the measurement map and particles dataframes
        _create_particle_id_column(measurement_particles_map, particles)

        # Filter particles with IDs associated with less than the minimum number of measurements (default: 9)
        measurement_counts = measurement_particles_map["particle_id"].value_counts()
        valid_particle_ids = measurement_counts[measurement_counts >= args.min_hits_per_particle].index

        # Process particles and space points
        particles = _process_particles_data(particles, valid_particle_ids)
        space_points = _process_space_points_data(space_points, measurement_particles_map)

        # Add event ID to all datasets
        particles["event_id"] = counter
        space_points["event_id"] = counter

        # Set particle_id to -1 for space points that don't match any remaining particle after filtering
        # This handles both NaN values and particle_ids that were filtered out during processing
        particle_ids = set(particles["particle_id"].values)
        mask_id = space_points["particle_id"].isna() | ~space_points["particle_id"].isin(particle_ids)
        space_points.loc[mask_id, "particle_id"] = -1

        # Assign to pre-allocated lists instead of appending
        all_particles[counter] = particles
        all_space_points[counter] = space_points

    print("Concatenating all data...")

    # Concatenate all DataFrames at once - much more efficient
    full_particles = pd.concat(all_particles, ignore_index=True) if all_particles else pd.DataFrame()
    full_space_points = pd.concat(all_space_points, ignore_index=True) if all_space_points else pd.DataFrame()

    print("Concatenation completed")

    # Print statistics
    print("Final data shapes:")
    print(f"  Particles: {full_particles.shape}")
    print(f"  Space Points: {full_space_points.shape}")

    print("Writing output files...")

    # Write the new files with optional numbering
    particles_filename = f"{args.input_path}/particles_small{file_suffix}.csv"
    space_points_filename = f"{args.input_path}/space_points_small{file_suffix}.csv"
    hdf_filename = f"{args.input_path}/processed_data{file_suffix}.h5"

    files_written = []

    # Write CSV files if requested
    if "csv" in args.output_format:
        full_particles.to_csv(particles_filename, index=False)
        full_space_points.to_csv(space_points_filename, index=False)
        files_written.extend([particles_filename, space_points_filename])

    # Write H5 files if requested
    if "h5" in args.output_format:
        with pd.HDFStore(hdf_filename, mode="w", complevel=9, complib="blosc") as store:
            store.put("particles", full_particles, format="table")
            store.put("space_points", full_space_points, format="table")
        files_written.append(hdf_filename)

    print("Files written successfully:")
    for filename in files_written:
        print(f"  {filename}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Preprocess Athena simulation data")
    parser.add_argument(
        "--file-number",
        type=int,
        default=None,
        help="Optional number to append to output filenames (e.g., --file-number 1 produces particles_small_1.csv)",
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

    parser.add_argument(
        "--min-hits-per-particle",
        type=int,
        default=9,
        help="Minimum number of measurements required for a particle to be included (default: 9)",
    )

    parser.add_argument(
        "--output-format",
        nargs="+",
        default=["csv"],
        choices=["csv", "h5"],
        help="Output file format(s): 'csv' (default), 'h5', or both (e.g., --output-format csv h5)",
    )

    args = parser.parse_args()
    read_athena_csv(args)
