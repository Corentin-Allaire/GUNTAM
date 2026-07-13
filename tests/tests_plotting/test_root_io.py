"""Tests for GUNTAM.Plotting.RootIO."""

from pathlib import Path

import numpy as np
import pytest
import uproot.containers

from GUNTAM.Plotting import RootIO

DATA_DIR = Path(__file__).parent.parent / "data" / "geant4_pu200_ttbar_odd_5000evt"
SEEDING_FILE = str(DATA_DIR / "guntam" / "performance_seeding.root")


class TestPatchUprootAsVector:
    def test_idempotent(self):
        patched_once = uproot.containers.AsVector.read
        RootIO.patch_uproot_asvector()
        patched_twice = uproot.containers.AsVector.read
        assert patched_once is patched_twice

    def test_unrelated_not_implemented_error_reraised(self, monkeypatch):
        def fake_orig_read(self, chunk, cursor, context, file, selffile, parent, header=True):
            raise NotImplementedError("some unrelated failure")

        monkeypatch.setattr(uproot.containers.AsVector, "read", fake_orig_read)
        monkeypatch.setattr(RootIO, "_patched", False)
        RootIO.patch_uproot_asvector()

        with pytest.raises(NotImplementedError, match="unrelated failure"):
            uproot.containers.AsVector.read(None, None, None, None, None, None, None)

    def test_target_not_implemented_error_is_handled(self, monkeypatch):
        def fake_orig_read(self, chunk, cursor, context, file, selffile, parent, header=True):
            raise NotImplementedError("streamerless memberwise serialization of class AsVector(pair<double,double>)")

        class FakeCursor:
            def skip(self, n):
                pass

            def field(self, chunk, struct_obj, context):
                return 0

        monkeypatch.setattr(uproot.containers.AsVector, "read", fake_orig_read)
        monkeypatch.setattr(RootIO, "_patched", False)
        RootIO.patch_uproot_asvector()

        result = uproot.containers.AsVector.read(None, None, FakeCursor(), None, None, None, None)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0


CKF_FILE = str(DATA_DIR / "guntam" / "performance_finding_ckf.root")


class TestListPlottableKeys:
    def test_excludes_non_plottable_classes(self):
        keys = RootIO.list_plottable_keys(SEEDING_FILE)
        assert "nFakeTracks_vs_eta" not in keys  # TH2F
        assert "eff_tracks" not in keys  # TVectorT<float>

    def test_includes_expected_teff_and_tprofile_keys(self):
        keys = RootIO.list_plottable_keys(SEEDING_FILE)
        assert keys["trackeff_vs_eta"] == "TEfficiency"
        assert keys["nDuplicated_vs_pT"] == "TProfile"

    def test_excludes_two_dimensional_teff_sharing_the_same_classname(self):
        """trackeff_vs_eta_phi/trackeff_vs_eta_pt are TEfficiency like trackeff_vs_eta, but 2D."""
        keys = RootIO.list_plottable_keys(CKF_FILE)
        assert "trackeff_vs_eta_phi" not in keys
        assert "trackeff_vs_eta_pt" not in keys
        assert "trackeff_vs_eta" in keys

    def test_deduped_by_base_name(self):
        keys = RootIO.list_plottable_keys(SEEDING_FILE)
        assert all(";" not in k for k in keys)


class TestReadEfficiency:
    def test_shape_and_range(self):
        data = RootIO.read_efficiency(SEEDING_FILE, "trackeff_vs_eta")
        assert data.x.shape == data.eff.shape == data.err_lo.shape == data.err_hi.shape
        valid = ~np.isnan(data.eff)
        assert np.all(data.eff[valid] >= 0.0)
        assert np.all(data.eff[valid] <= 1.0)


class TestClopperPearson:
    def test_scalar(self):
        eff, err_lo, err_hi = RootIO.clopper_pearson(50, 100)
        assert eff == pytest.approx(0.5)
        assert err_lo > 0
        assert err_hi > 0

    def test_array(self):
        eff, err_lo, err_hi = RootIO.clopper_pearson(np.array([0, 50, 100]), np.array([100, 100, 100]))
        assert eff[0] == pytest.approx(0.0)
        assert eff[1] == pytest.approx(0.5)
        assert eff[2] == pytest.approx(1.0)

    def test_n_pass_equals_n_tot_upper_error_is_zero_not_nan(self):
        eff, err_lo, err_hi = RootIO.clopper_pearson(100, 100)
        assert eff == pytest.approx(1.0)
        assert err_hi == 0.0
        assert not np.isnan(err_hi)

    def test_zero_total_is_nan(self):
        eff, err_lo, err_hi = RootIO.clopper_pearson(0, 0)
        assert np.isnan(eff)


class TestFoldAbsolute:
    def test_pools_symmetric_bins(self):
        x = np.array([-2.0, -1.0, 1.0, 2.0])
        n_pass = np.array([10.0, 20.0, 15.0, 5.0])
        n_tot = np.array([20.0, 40.0, 30.0, 10.0])

        abs_x, folded_pass, folded_tot = RootIO.fold_absolute(x, n_pass, n_tot)

        assert list(abs_x) == [1.0, 2.0]
        # bin at -1 (20/40) pools with bin at +1 (15/30) -> pass=35, tot=70
        idx1 = list(abs_x).index(1.0)
        idx2 = list(abs_x).index(2.0)
        assert folded_pass[idx1] == pytest.approx(35.0)
        assert folded_tot[idx1] == pytest.approx(70.0)
        assert folded_pass[idx2] == pytest.approx(15.0)
        assert folded_tot[idx2] == pytest.approx(30.0)


class TestPooledEfficiency:
    def test_cross_checked_against_manual_pooling(self):
        _, n_pass, n_tot = RootIO.read_efficiency_raw(SEEDING_FILE, "trackeff_vs_eta")
        pooled = RootIO.pooled_efficiency(SEEDING_FILE, "trackeff_vs_eta")

        expected_eff = n_pass.sum() / n_tot.sum()
        assert pooled.eff == pytest.approx(expected_eff)
        assert pooled.n_pass_total == pytest.approx(float(n_pass.sum()))
        assert pooled.n_tot_total == pytest.approx(float(n_tot.sum()))


class TestInverseVarianceCombine:
    def test_hand_computed_example(self):
        values = np.array([2.0, 4.0])
        errors = np.array([1.0, 2.0])
        # w1 = 1, w2 = 0.25 -> mean = (1*2 + 0.25*4) / 1.25 = 3/1.25 = 2.4
        combined = RootIO.inverse_variance_combine(values, errors)
        assert combined.mean == pytest.approx(2.4)
        assert combined.err == pytest.approx(1.0 / np.sqrt(1.25))

    def test_excludes_zero_error_bins(self):
        values = np.array([2.0, 100.0])
        errors = np.array([1.0, 0.0])  # second bin is an empty TProfile bin (value=0, error=0 pattern)
        combined = RootIO.inverse_variance_combine(values, errors)
        assert combined.mean == pytest.approx(2.0)

    def test_all_invalid_returns_nan(self):
        combined = RootIO.inverse_variance_combine(np.array([np.nan]), np.array([0.0]))
        assert np.isnan(combined.mean)
        assert np.isnan(combined.err)

    def test_smoke_against_real_sample_file(self):
        combined = RootIO.combine_profile_inverse_variance(SEEDING_FILE, "nDuplicated_vs_pT")
        assert not np.isnan(combined.mean)
        assert combined.err > 0
