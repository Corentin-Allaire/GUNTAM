import pandas as pd
import numpy as np

from GUNTAM.IO.Read_ACTS_Csv import (
    extract_masked_values,
    _create_particle_id_column,
    _process_hits_data,
    _process_particles_data,
    _process_space_points_data,
)


class TestExtractMaskedValues:
    def test_basic_extraction(self):
        # Test with known packed geometry_id
        # Volume=1, Layer=2, Sensitive=3, Extra=4
        geometry_id = (1 << 56) | (2 << 36) | (3 << 8) | 4
        volume, layer, sensitive, extra = extract_masked_values(geometry_id)
        assert volume == 1
        assert layer == 2
        assert sensitive == 3
        assert extra == 4

    def test_zero_values(self):
        # All zeros
        volume, layer, sensitive, extra = extract_masked_values(0)
        assert volume == 0
        assert layer == 0
        assert sensitive == 0
        assert extra == 0

    def test_max_values(self):
        # Test with maximum values for each field
        # Volume: 8 bits (255), Layer: 12 bits (4095), Sensitive: 20 bits (1048575), Extra: 8 bits (255)
        geometry_id = (255 << 56) | (4095 << 36) | (1048575 << 8) | 255
        volume, layer, sensitive, extra = extract_masked_values(geometry_id)
        assert volume == 255
        assert layer == 4095
        assert sensitive == 1048575
        assert extra == 255


class TestCreateParticleIdColumn:
    def test_matching_particles_and_hits(self):
        # Create test data with matching particle IDs
        particles = pd.DataFrame({
            "particle_id_pv": [0, 0, 1],
            "particle_id_sv": [0, 0, 0],
            "particle_id_part": [1, 2, 3],
            "particle_id_gen": [0, 0, 0],
            "particle_id_subpart": [0, 0, 0],
        })
        
        hits = pd.DataFrame({
            "particle_id_pv": [0, 0, 1, 0],
            "particle_id_sv": [0, 0, 0, 0],
            "particle_id_part": [1, 2, 3, 1],
            "particle_id_gen": [0, 0, 0, 0],
            "particle_id_subpart": [0, 0, 0, 0],
        })
        
        _create_particle_id_column(hits, particles)
        
        # Check particles got particle_id
        assert "particle_id" in particles.columns
        assert list(particles["particle_id"]) == [0, 1, 2]
        
        # Check hits got mapped particle_id
        assert "particle_id" in hits.columns
        assert list(hits["particle_id"]) == [0, 1, 2, 0]
        
        # Check temporary key column was removed
        assert "_key" not in particles.columns
        assert "_key" not in hits.columns

    def test_unmapped_hits(self):
        # Test hits that don't match any particle
        particles = pd.DataFrame({
            "particle_id_pv": [0],
            "particle_id_sv": [0],
            "particle_id_part": [1],
            "particle_id_gen": [0],
            "particle_id_subpart": [0],
        })
        
        hits = pd.DataFrame({
            "particle_id_pv": [0, 1],
            "particle_id_sv": [0, 0],
            "particle_id_part": [1, 2],  # Second hit doesn't match any particle
            "particle_id_gen": [0, 0],
            "particle_id_subpart": [0, 0],
        })
        
        _create_particle_id_column(hits, particles)
        
        # First hit should map, second should be NaN
        assert hits["particle_id"].iloc[0] == 0
        assert pd.isna(hits["particle_id"].iloc[1])


class TestProcessHitsData:
    def test_basic_processing(self):
        # Create minimal test data
        data = pd.DataFrame({
            "particle_id": [0, 1, 2],
            "particle_id_pv": [0, 0, 1],
            "particle_id_sv": [0, 0, 0],
            "particle_id_part": [1, 2, 3],
            "particle_id_gen": [0, 0, 0],
            "particle_id_subpart": [0, 0, 0],
            "geometry_id": [
                (1 << 56) | (2 << 36) | (3 << 8) | 4,
                (2 << 56) | (3 << 36) | (4 << 8) | 5,
                (3 << 56) | (4 << 36) | (5 << 8) | 6,
            ],
            "tx": [5.0, 10.0, 20.0],
            "ty": [5.0, 10.0, 20.0],
            "tz": [10.0, 20.0, 30.0],
        })
        
        result = _process_hits_data(data)
        
        # Check required columns exist
        assert "x" in result.columns
        assert "y" in result.columns
        assert "z" in result.columns
        assert "r" in result.columns
        assert "eta" in result.columns
        assert "phi" in result.columns
        assert "volume" in result.columns
        assert "layer" in result.columns
        assert "sensitive" in result.columns
        assert "extra" in result.columns
        assert "hit_id" in result.columns
        assert "varR" in result.columns
        assert "varZ" in result.columns
        assert "badSP" in result.columns
        
        # Check geometry extraction
        assert result["volume"].iloc[0] == 1
        assert result["layer"].iloc[0] == 2
        assert result["sensitive"].iloc[0] == 3
        assert result["extra"].iloc[0] == 4
        
        # Check spatial calculations
        assert np.isclose(result["r"].iloc[0], np.sqrt(50))  # sqrt(5^2 + 5^2)
        assert np.isclose(result["r"].iloc[1], np.sqrt(200))  # sqrt(10^2 + 10^2)

    def test_spatial_filtering(self):
        # Test that hits outside R_max and Z_max are filtered
        data = pd.DataFrame({
            "particle_id": [0, 1, 2],
            "particle_id_pv": [0, 0, 1],
            "particle_id_sv": [0, 0, 0],
            "particle_id_part": [1, 2, 3],
            "particle_id_gen": [0, 0, 0],
            "particle_id_subpart": [0, 0, 0],
            "geometry_id": [
                (1 << 56) | (2 << 36) | (3 << 8) | 4,
                (2 << 56) | (3 << 36) | (4 << 8) | 5,
                (3 << 56) | (4 << 36) | (5 << 8) | 6,
            ],
            "tx": [5.0, 600.0, 10.0],  # Second hit beyond R_max=500
            "ty": [5.0, 0.0, 10.0],
            "tz": [10.0, 20.0, 2000.0],  # Third hit beyond Z_max=1000
        })
        
        result = _process_hits_data(data, R_max=500, Z_max=1000)
        
        # Only first hit should remain
        assert len(result) == 1
        assert result["particle_id"].iloc[0] == 0

    def test_duplicate_removal(self):
        # Test that duplicate hits (same tx, ty, tz) are removed
        data = pd.DataFrame({
            "particle_id": [0, 1],
            "particle_id_pv": [0, 0],
            "particle_id_sv": [0, 0],
            "particle_id_part": [1, 1],
            "particle_id_gen": [0, 0],
            "particle_id_subpart": [0, 0],
            "geometry_id": [
                (1 << 56) | (2 << 36) | (3 << 8) | 4,
                (1 << 56) | (2 << 36) | (3 << 8) | 4,
            ],
            "tx": [10.0, 10.0],  # Duplicate position
            "ty": [20.0, 20.0],
            "tz": [30.0, 30.0],
        })
        
        result = _process_hits_data(data)
        
        # Should have only one hit after deduplication
        assert len(result) == 1


class TestProcessParticlesData:
    def test_basic_processing(self):
        # Create test particles
        particles = pd.DataFrame({
            "particle_id": [0, 1, 2],
            "px": [1.0, 2.0, 3.0],
            "py": [0.0, 0.0, 4.0],
            "pz": [5.0, 6.0, 0.0],  # Third particle has pz=0, should be filtered
            "vx": [0.1, 0.2, 0.3],
            "vy": [0.1, 0.2, 0.3],
            "vz": [1.0, 2.0, 3.0],
        })
        
        valid_ids = pd.Index([0, 1, 2])
        result = _process_particles_data(particles, valid_ids)
        
        # Third particle should be filtered (pz=0)
        assert len(result) == 2
        
        # Check computed columns
        assert "pT" in result.columns
        assert "eta" in result.columns
        assert "phi" in result.columns
        assert "d0" in result.columns
        assert "z0" in result.columns
        
        # Check pT calculation
        assert np.isclose(result.iloc[0]["pT"], 1.0)  # sqrt(1^2 + 0^2)
        assert np.isclose(result.iloc[1]["pT"], 2.0)  # sqrt(2^2 + 0^2)

    def test_filtering_by_valid_ids(self):
        # Test that only valid particle IDs are kept
        particles = pd.DataFrame({
            "particle_id": [0, 1, 2],
            "px": [1.0, 2.0, 3.0],
            "py": [0.0, 0.0, 4.0],
            "pz": [5.0, 6.0, 7.0],
            "vx": [0.1, 0.2, 0.3],
            "vy": [0.1, 0.2, 0.3],
            "vz": [1.0, 2.0, 3.0],
        })
        
        # Only keep particles 0 and 2
        valid_ids = pd.Index([0, 2])
        result = _process_particles_data(particles, valid_ids)
        
        assert len(result) == 2
        assert 0 in result["particle_id"].values
        assert 2 in result["particle_id"].values
        assert 1 not in result["particle_id"].values


class TestProcessSpacePointsData:
    def test_basic_processing(self):
        # Create test space points
        space_points = pd.DataFrame({
            "measurement_id_1": [0, 1],
            "measurement_id_2": [1, 2],
            "x": [10.0, 20.0],
            "y": [10.0, 20.0],
            "z": [30.0, 40.0],
            "var_r": [0.1, 0.2],
            "var_z": [0.1, 0.2],
        })
        
        # Create hit measurement map
        hit_measurement_map = pd.DataFrame({
            "measurement_id": [0, 1, 2],
            "hit_id": [10, 11, 12],
        })
        
        # Create hits with required columns
        hits = pd.DataFrame({
            "event_id": [0, 0, 0],
            "hit_id": [10, 11, 12],
            "particle_id": [0, 0, 1],
            "particle_id_pv": [0, 0, 1],
            "particle_id_sv": [0, 0, 0],
            "particle_id_part": [1, 1, 2],
            "particle_id_gen": [0, 0, 0],
            "particle_id_subpart": [0, 0, 0],
            "volume": [1, 1, 2],
            "layer": [2, 2, 3],
            "sensitive": [3, 3, 4],
            "extra": [4, 4, 5],
        })
        
        result = _process_space_points_data(space_points, hit_measurement_map, hits)
        
        # Check required columns exist
        assert "x" in result.columns
        assert "y" in result.columns
        assert "z" in result.columns
        assert "r" in result.columns
        assert "eta" in result.columns
        assert "phi" in result.columns
        assert "particle_id" in result.columns
        assert "varR" in result.columns
        assert "varZ" in result.columns
        assert "badSP" in result.columns
        assert "event_id" in result.columns
        
        # Check spatial calculations
        assert np.isclose(result.iloc[0]["r"], np.sqrt(10.0**2 + 10.0**2))
        
        # Check that badSP marks mismatched particle IDs
        assert "badSP" in result.columns

    def test_filters_unmapped_space_points(self):
        # Create space points with measurement IDs that don't map to hits
        space_points = pd.DataFrame({
            "measurement_id_1": [0, 99],  # 99 doesn't exist in map
            "measurement_id_2": [1, 2],
            "x": [10.0, 20.0],
            "y": [10.0, 20.0],
            "z": [30.0, 40.0],
            "var_r": [0.1, 0.2],
            "var_z": [0.1, 0.2],
        })
        
        hit_measurement_map = pd.DataFrame({
            "measurement_id": [0, 1, 2],
            "hit_id": [10, 11, 12],
        })
        
        hits = pd.DataFrame({
            "event_id": [0, 0, 0],
            "hit_id": [10, 11, 12],
            "particle_id": [0, 0, 1],
            "particle_id_pv": [0, 0, 1],
            "particle_id_sv": [0, 0, 0],
            "particle_id_part": [1, 1, 2],
            "particle_id_gen": [0, 0, 0],
            "particle_id_subpart": [0, 0, 0],
            "volume": [1, 1, 2],
            "layer": [2, 2, 3],
            "sensitive": [3, 3, 4],
            "extra": [4, 4, 5],
        })
        
        result = _process_space_points_data(space_points, hit_measurement_map, hits)
        
        # Only first space point should remain (measurement_id_1=0 is valid)
        assert len(result) == 1
