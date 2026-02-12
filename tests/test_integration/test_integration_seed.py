import os
import pytest
import shutil
import tempfile
import subprocess
import sys
from pathlib import Path


class TestFullIntegration:
    """
    Full integration test for the complete GUNTAM pipeline:
    1. Read ACTS CSV data in both hits and spacepoints modes
    2. Write output to both CSV and H5 formats
    3. Run preprocessing on all outputs
    4. Run training (2 epochs)
    5. Clean up all generated files
    """

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs"""
        temp_dir = tempfile.mkdtemp(prefix="guntam_integration_test_")
        yield temp_dir
        # Cleanup after test
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def test_data_dir(self, temp_dir):
        """
        Path to test data directory with proper structure.
        Creates an 'odd_output' subdirectory and copies test files there.
        """
        source_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        test_input_dir = os.path.join(temp_dir, "test_input")
        odd_output_dir = os.path.join(test_input_dir, "odd_output")
        os.makedirs(odd_output_dir, exist_ok=True)
        
        # Copy all test data files to the odd_output directory
        for filename in os.listdir(source_dir):
            if filename.endswith(".csv"):
                shutil.copy(
                    os.path.join(source_dir, filename),
                    os.path.join(odd_output_dir, filename)
                )
        
        return test_input_dir

    @pytest.mark.parametrize(
        "use_space_point,output_format",
        [
            (False, "csv"),
            (False, "h5"),
            (True, "csv"),
            (True, "h5"),
        ],
        ids=["hits_csv", "hits_h5", "spacepoints_csv", "spacepoints_h5"]
    )
    def test_read_and_preprocessing(self, temp_dir, test_data_dir, use_space_point, output_format):
        """Test read and preprocessing for all data modes and output formats"""
        mode = "sp" if use_space_point else "hits"
        suffix = f"{mode}_{output_format}"
        
        # Step 1: Read ACTS CSV data
        print(f"\n{'=' * 80}")
        print(f"STEP 1: Reading ACTS CSV data ({suffix})")
        print(f"{'=' * 80}")
        
        self._run_read_acts_csv(
            test_data_dir=test_data_dir,
            use_space_point=use_space_point,
            output_format=output_format
        )
        
        # Step 2: Run preprocessing
        print(f"\n{'=' * 80}")
        print(f"STEP 2: Running preprocessing ({suffix})")
        print(f"{'=' * 80}")
        
        preprocessing_output = os.path.join(temp_dir, f"preprocessing_output_{suffix}")
        os.makedirs(preprocessing_output, exist_ok=True)
        
        self._run_preprocessing(
            input_path=test_data_dir,
            output_path=preprocessing_output,
            input_format=output_format,
            dataset_name=f"test_{suffix}",
            max_events=1
        )
        
        # Step 3: Verify read and preprocessing outputs
        print(f"\n{'=' * 80}")
        print(f"STEP 3: Verifying outputs ({suffix})")
        print(f"{'=' * 80}")
        
        self._verify_read_and_preprocessing_outputs(
            read_output=test_data_dir,
            preprocessing_output=preprocessing_output,
            output_format=output_format,
            use_space_point=use_space_point
        )
        
        print(f"\n{'=' * 80}")
        print(f"READ AND PREPROCESSING COMPLETED SUCCESSFULLY ({suffix})")
        print(f"{'=' * 80}\n")

    # The training test is separate to allow verifying read/preprocessing outputs
    # before training, and to isolate training issues if they arise.
    def test_full_pipeline_with_training(self, temp_dir, test_data_dir):
        """Test full pipeline including training for spacepoints with CSV format"""
        use_space_point = True
        output_format = "csv"
        suffix = "sp_csv"
        
        # Step 1: Read ACTS CSV data
        print(f"\n{'=' * 80}")
        print(f"STEP 1: Reading ACTS CSV data ({suffix})")
        print(f"{'=' * 80}")
        
        self._run_read_acts_csv(
            test_data_dir=test_data_dir,
            use_space_point=use_space_point,
            output_format=output_format
        )
        
        # Step 2: Run preprocessing
        print(f"\n{'=' * 80}")
        print(f"STEP 2: Running preprocessing ({suffix})")
        print(f"{'=' * 80}")
        
        preprocessing_output = os.path.join(temp_dir, f"preprocessing_output_{suffix}")
        os.makedirs(preprocessing_output, exist_ok=True)
        
        self._run_preprocessing(
            input_path=test_data_dir,
            output_path=preprocessing_output,
            input_format=output_format,
            dataset_name=f"test_{suffix}"
        )
        
        # Step 3: Run training
        print(f"\n{'=' * 80}")
        print(f"STEP 3: Running training ({suffix})")
        print(f"{'=' * 80}")
        
        model_path = os.path.join(temp_dir, f"model_{suffix}.pt")
        
        self._run_training(
            input_path=test_data_dir,
            input_tensor_path=preprocessing_output,
            model_path=model_path,
            dataset_name=f"test_{suffix}",
            input_format=output_format
        )
        
        # Step 4: Verify all outputs including model
        print(f"\n{'=' * 80}")
        print(f"STEP 4: Verifying outputs ({suffix})")
        print(f"{'=' * 80}")
        
        self._verify_outputs(
            read_output=test_data_dir,
            preprocessing_output=preprocessing_output,
            model_path=model_path,
            output_format=output_format,
            use_space_point=use_space_point
        )
        
        print(f"\n{'=' * 80}")
        print(f"FULL PIPELINE WITH TRAINING COMPLETED SUCCESSFULLY ({suffix})")
        print(f"{'=' * 80}\n")

    def _run_read_acts_csv(self, test_data_dir, use_space_point, output_format):
        """
        Run the Read_ACTS_Csv.py script
        
        Args:
            test_data_dir: Directory containing test CSV files (also used as output)
            use_space_point: Whether to use spacepoints
            output_format: Output format ('csv' or 'h5')
        """
        # Build the command
        cmd = [
            sys.executable,
            "-m", "GUNTAM.IO.Read_ACTS_Csv",
            "--input-path", test_data_dir,
            "--output-format", output_format,
            "--min-hits-per-particle", "9",
            "--file-number", "0"
        ]
        
        if use_space_point:
            cmd.append("--use-space-point")
        
        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        print(f"Read ACTS CSV command: {' '.join(cmd)}")
        print(f"Return code: {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        
        # Check if command succeeded
        assert result.returncode == 0, f"Read_ACTS_Csv failed with return code {result.returncode}"

    def _run_preprocessing(self, input_path, output_path, input_format, dataset_name, max_events=5):
        """
        Run the PrepareTensor.py script
        
        Args:
            input_path: Directory containing input data
            output_path: Directory for output tensors
            input_format: Input format ('csv' or 'h5')
            dataset_name: Name for the dataset
            max_events: Maximum number of events to process (default: 5)
        """
        # Build the command
        cmd = [
            sys.executable,
            "-m", "GUNTAM.IO.PrepareTensor",
            "--input_path", input_path,
            "--input_format", input_format,
            "--input_tensor_path", output_path,
            "--dataset_name", dataset_name,
            "--events_per_file", "1",
            "--max_events", str(max_events),
            "--binning_strategy", "neighbor",
            "--bin_width", "0.02",
            "--max_hit_input", "1200",
            "--eta_range", "-3.0", "3.0",
            "--vertex_cuts", "10", "200",
            "--hit_features", "x", "y", "z", "r", "phi", "eta",
            "--particle_features", "d0", "z0", "phi", "eta", "pT", "q", "m"
        ]
        
        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        print(f"PrepareTensor command: {' '.join(cmd)}")
        print(f"Return code: {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        
        # Check if command succeeded
        assert result.returncode == 0, f"PrepareTensor failed with return code {result.returncode}"

    def _run_training(self, input_path, input_tensor_path, model_path, dataset_name, input_format):
        """
        Run the Train.py script
        
        Args:
            input_path: Directory containing raw CSV/H5 files
            input_tensor_path: Directory containing preprocessed tensors
            model_path: Path to save the trained model
            dataset_name: Name of the dataset
            input_format: Input format ('csv' or 'h5')
        """
        # Build the command
        cmd = [
            sys.executable,
            "-m", "GUNTAM.Seed.Train",
            "--input_path", input_path,
            "--input_format", input_format,
            "--input_tensor_path", input_tensor_path,
            "--dataset_name", dataset_name,
            "--model_path", model_path,
            "--epoch_nb", "2",
            "--test_fraction", "0.2",
            "--batch_size", "1",
            "--nb_layers_t", "2",  # Small model for testing
            "--dim_embedding", "64",  # Small embedding for testing
            "--nb_heads", "2",
            "--learning_rate", "1e-4",
            "--events_per_file", "1",
            "--max_events", "5",
            "--binning_strategy", "neighbor",
            "--bin_width", "0.02",
            "--max_hit_input", "1200",
            "--eta_range", "-3.0", "3.0",
            "--vertex_cuts", "10", "200"
        ]
        
        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        print(f"Training command: {' '.join(cmd)}")
        print(f"Return code: {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        
        # Check if command succeeded
        assert result.returncode == 0, f"Training failed with return code {result.returncode}"

    def _verify_read_and_preprocessing_outputs(self, read_output, preprocessing_output,
                                               output_format, use_space_point):
        """
        Verify that read and preprocessing output files were created

        Args:
            read_output: Directory with Read_ACTS_Csv outputs
            preprocessing_output: Directory with PrepareTensor outputs
            output_format: Output format used ('csv' or 'h5')
            use_space_point: Whether spacepoints were used
        """
        # Check Read_ACTS_Csv outputs
        if output_format == "csv":
            expected_particles = os.path.join(read_output, "particles_small_0.csv")
            assert os.path.exists(expected_particles), f"Particles CSV not found: {expected_particles}"

            if use_space_point:
                expected_data = os.path.join(read_output, "space_points_small_0.csv")
                assert os.path.exists(expected_data), f"Spacepoints CSV not found: {expected_data}"
            else:
                expected_data = os.path.join(read_output, "hits_small_0.csv")
                assert os.path.exists(expected_data), f"Hits CSV not found: {expected_data}"

        elif output_format == "h5":
            expected_h5 = os.path.join(read_output, "processed_data_0.h5")
            assert os.path.exists(expected_h5), f"H5 file not found: {expected_h5}"

        # Check PrepareTensor outputs
        metadata_files = list(Path(preprocessing_output).glob("metadata_*.pt"))
        assert len(metadata_files) > 0, f"No metadata files found in {preprocessing_output}"

        tensor_files = list(Path(preprocessing_output).rglob("tensor_data_*.pt"))
        assert len(tensor_files) > 0, f"No tensor data files found in {preprocessing_output}"

        print("  Read and preprocessing outputs verified successfully")
        print(f"  - Read output: {read_output}")
        print(f"  - Preprocessing output: {preprocessing_output}")

    def _verify_outputs(self, read_output, preprocessing_output, model_path,
                        output_format, use_space_point):
        """
        Verify that all expected output files were created
        
        Args:
            read_output: Directory with Read_ACTS_Csv outputs
            preprocessing_output: Directory with PrepareTensor outputs
            model_path: Path to trained model
            output_format: Output format used ('csv' or 'h5')
            use_space_point: Whether spacepoints were used
        """
        # Check Read_ACTS_Csv outputs
        if output_format == "csv":
            expected_particles = os.path.join(read_output, "particles_small_0.csv")
            assert os.path.exists(expected_particles), f"Particles CSV not found: {expected_particles}"
            
            if use_space_point:
                expected_data = os.path.join(read_output, "space_points_small_0.csv")
                assert os.path.exists(expected_data), f"Spacepoints CSV not found: {expected_data}"
            else:
                expected_data = os.path.join(read_output, "hits_small_0.csv")
                assert os.path.exists(expected_data), f"Hits CSV not found: {expected_data}"
        
        elif output_format == "h5":
            expected_h5 = os.path.join(read_output, "processed_data_0.h5")
            assert os.path.exists(expected_h5), f"H5 file not found: {expected_h5}"
        
        # Check PrepareTensor outputs
        # Find metadata file
        metadata_files = list(Path(preprocessing_output).glob("metadata_*.pt"))
        assert len(metadata_files) > 0, f"No metadata files found in {preprocessing_output}"
        
        # Check that at least one tensor data file exists
        tensor_files = list(Path(preprocessing_output).rglob("tensor_data_*.pt"))
        assert len(tensor_files) > 0, f"No tensor data files found in {preprocessing_output}"
        
        # Check model file
        assert os.path.exists(model_path), f"Model file not found: {model_path}"
        
        # Verify model file is not empty
        model_size = os.path.getsize(model_path)
        assert model_size > 0, f"Model file is empty: {model_path}"
        
        print("  All output files verified successfully")
        print(f"  - Read output: {read_output}")
        print(f"  - Preprocessing output: {preprocessing_output}")
        print(f"  - Model: {model_path} ({model_size / 1024:.2f} KB)")


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "-s"])
