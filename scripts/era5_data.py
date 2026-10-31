"""
scripts/era5_data.py
====================
Data management, normalization, padding, and Sequence generator for ERA5 weather forecasting.
Ensures 100% strict non-leakage chronological partitions and zero data snooping.
Supports cyclical temporal encodings (sin/cos hour, sin/cos DOY) and terrain elevation features.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import config

class ERA5DataManager:
    def __init__(self, data_dir: Path = config.DATA_DIR):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.year_files = {}
        self.timestamps_by_year = {}
        self.all_timestamps = None
        self.year_cache = {}  # Cache year arrays: y -> float32 array (T, 28, 34, 6)
        self.max_cached_years = 3
        
        # Spatial mask (32, 48): 1.0 on valid physical domain, 0.0 on padded cells
        self.spatial_mask = np.zeros(config.PADDED_SHAPE, dtype=np.float32)
        lat_start, lat_end = config.PAD_LAT[0], config.PAD_LAT[0] + config.ORIGINAL_LAT
        lon_start, lon_end = config.PAD_LON[0], config.PAD_LON[0] + config.ORIGINAL_LON
        self.spatial_mask[lat_start:lat_end, lon_start:lon_end] = 1.0

        self._discover_files()
        self._build_time_index()

    def _discover_files(self):
        for y in range(2011, 2026):
            fp = self.data_dir / f"{y}_final.nc"
            if not fp.exists():
                raise FileNotFoundError(f"Missing required final NetCDF file: {fp}")
            self.year_files[y] = fp

    def _build_time_index(self):
        all_times = []
        for y in sorted(self.year_files.keys()):
            with xr.open_dataset(self.year_files[y]) as ds:
                time_var = "valid_time" if "valid_time" in ds.coords else "time"
                times = pd.DatetimeIndex(ds[time_var].values)
                self.timestamps_by_year[y] = times
                all_times.append(times)
        self.all_timestamps = pd.DatetimeIndex(np.concatenate(all_times))

    def get_year_data(self, year: int) -> np.ndarray:
        """
        Loads and caches a year's array of shape (T, 28, 34, 6).
        """
        if year in self.year_cache:
            return self.year_cache[year]
            
        if len(self.year_cache) >= self.max_cached_years:
            # Evict first cached year
            evict_y = next(iter(self.year_cache))
            del self.year_cache[evict_y]
            
        fp = self.year_files[year]
        with xr.open_dataset(fp) as ds:
            # Stack all 6 variables into shape (T, 28, 34, 6)
            var_arrays = [ds[v].values.astype(np.float32) for v in config.INPUT_VARIABLES]
            year_arr = np.stack(var_arrays, axis=-1)
            self.year_cache[year] = year_arr
            return year_arr

    def compute_normalization_stats(self, train_years: list = config.TRAIN_YEARS) -> dict:
        """
        Computes mean, std, min, and max strictly from the training years (2011-2021).
        Guarantees zero leakage from validation or test partitions.
        """
        print(f"Computing normalization statistics strictly from training years: {train_years[0]} to {train_years[-1]}...")
        stats = {}
        for idx, var_name in enumerate(config.INPUT_VARIABLES):
            all_vals = []
            for y in train_years:
                arr = self.get_year_data(y)[..., idx]
                all_vals.append(arr.ravel())
            concat_vals = np.concatenate(all_vals)
            
            v_mean = float(np.mean(concat_vals))
            v_std = float(np.std(concat_vals))
            v_min = float(np.min(concat_vals))
            v_max = float(np.max(concat_vals))
            v_range = float(v_max - v_min)
            
            stats[var_name] = {
                "mean": v_mean,
                "std": v_std if v_std > 1e-6 else 1.0,
                "min": v_min,
                "max": v_max,
                "range": v_range if v_range > 1e-6 else 1.0
            }
            print(f"  {var_name:5s} -> Mean: {v_mean:.2f}, Std: {v_std:.2f}, Min: {v_min:.2f}, Max: {v_max:.2f}")

        # Save stats
        stats_path = config.OUTPUTS_DIR / "normalization_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Normalization statistics saved to {stats_path}")
        return stats

    def pad_spatial(self, arr: np.ndarray) -> np.ndarray:
        """
        Symmetrically edge-pads array from (..., 28, 34, ...) to (..., 32, 48, ...).
        Preserves original geographic coordinates without distortion.
        """
        ndim = arr.ndim
        # Expected latitude at -3, longitude at -2, or latitude at -2, longitude at -1
        if ndim == 5:
            # (batch, lags, lat, lon, feats) or (lags, lat, lon, feats)
            pad_width = [(0, 0), (0, 0), config.PAD_LAT, config.PAD_LON, (0, 0)]
        elif ndim == 4:
            # (lags, lat, lon, feats) or (batch, lat, lon, feats)
            pad_width = [(0, 0), config.PAD_LAT, config.PAD_LON, (0, 0)]
        elif ndim == 3:
            # (lat, lon, feats) or (batch, lat, lon)
            pad_width = [config.PAD_LAT, config.PAD_LON, (0, 0)]
        elif ndim == 2:
            pad_width = [config.PAD_LAT, config.PAD_LON]
        else:
            raise ValueError(f"Unsupported array ndim for padding: {ndim}")
            
        return np.pad(arr, pad_width, mode="edge")

    def unpad_spatial(self, arr: np.ndarray) -> np.ndarray:
        """
        Crops padded dimensions back to exact physical grid (28, 34).
        """
        lat_s, lat_e = config.PAD_LAT[0], config.PAD_LAT[0] + config.ORIGINAL_LAT
        lon_s, lon_e = config.PAD_LON[0], config.PAD_LON[0] + config.ORIGINAL_LON
        
        if arr.ndim == 5:
            return arr[:, :, lat_s:lat_e, lon_s:lon_e, :]
        elif arr.ndim == 4:
            return arr[:, lat_s:lat_e, lon_s:lon_e, :]
        elif arr.ndim == 3:
            return arr[lat_s:lat_e, lon_s:lon_e, :]
        elif arr.ndim == 2:
            return arr[lat_s:lat_e, lon_s:lon_e]
        else:
            raise ValueError(f"Unsupported array ndim for unpadding: {arr.ndim}")

    def compute_climatology(self, train_years: list = config.TRAIN_YEARS) -> dict:
        """
        Computes historical training average t2m for each calendar day and 2-hour UTC step.
        Provides a climatological baseline for test evaluation.
        """
        climatology = {}
        for y in train_years:
            arr = self.get_year_data(y)[..., config.TARGET_INDEX]
            times = self.timestamps_by_year[y]
            for i, t in enumerate(times):
                key = (t.month, t.day, t.hour)
                if key not in climatology:
                    climatology[key] = []
                climatology[key].append(arr[i])
                
        # Mean across years for each (month, day, hour)
        clim_mean_map = {k: np.mean(v, axis=0) for k, v in climatology.items()}
        return clim_mean_map


class ERA5Sequence(tf.keras.utils.Sequence):
    """
    Keras Sequence generator delivering mini-batches of:
    - Inputs: (batch, 6, 32, 48, C_in) for 3D Broad-UNet or (batch, 32, 48, 6*C_in) for 2D UNet
    - Targets: (batch, 1, 32, 48, num_horizons)
    Supports:
    - include_cyclical_time: adds sin/cos hour-of-day and sin/cos day-of-year (+4 channels)
    - include_terrain: adds normalized ln(msl / sp) surface geopotential proxy (+1 channel)
    """
    def __init__(
        self,
        data_manager: ERA5DataManager,
        years: list,
        norm_stats: dict,
        horizon: int = None,  # None means all 5 horizons [1, 2, 4, 8, 12]
        batch_size: int = config.BATCH_SIZE,
        dim_model: str = "3d",
        norm_method: str = config.NORM_METHOD,
        include_cyclical_time: bool = False,
        include_terrain: bool = False,
        shuffle: bool = False,
        seed: int = 42
    ):
        super().__init__()
        self.dm = data_manager
        self.years = years
        self.norm_stats = norm_stats
        self.horizon = horizon
        self.batch_size = batch_size
        self.dim_model = dim_model
        self.norm_method = norm_method
        self.include_cyclical_time = include_cyclical_time
        self.include_terrain = include_terrain
        self.shuffle = shuffle
        self.seed = seed
        
        self.lags = config.LAGS  # 6
        self.horizons = [horizon] if horizon is not None else config.HORIZONS
        self.max_lead = max(self.horizons)
        self.required_window = self.lags + self.max_lead
        
        # Build window indices: (year, start_idx, target_timestamps_list)
        self.windows = []
        self._build_windows()
        self.indices = np.arange(len(self.windows))
        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(self.indices)

    def _build_windows(self):
        for y in self.years:
            y_times = self.dm.timestamps_by_year[y]
            n_steps = len(y_times)
            # Ensure window fits within the year
            for i in range(n_steps - self.required_window + 1):
                input_end_t = y_times[i + self.lags - 1]
                target_times = [y_times[i + self.lags + h - 1] for h in self.horizons]
                self.windows.append((y, i, input_end_t, target_times))

    def __len__(self):
        return int(np.floor(len(self.windows) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_window_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        x_batch = []
        y_batch = []
        
        for w_idx in batch_window_indices:
            y, start_i, input_end_t, _ = self.windows[w_idx]
            year_arr = self.dm.get_year_data(y)
            
            # Slice input frames: (6, 28, 34, 6)
            x_raw = year_arr[start_i : start_i + self.lags].copy()
            
            # Slice target frames: shape (1, 28, 34, len(horizons))
            y_raw_list = []
            for h in self.horizons:
                target_idx = start_i + self.lags + h - 1
                y_raw_list.append(year_arr[target_idx : target_idx + 1, :, :, config.TARGET_INDEX : config.TARGET_INDEX + 1])
            y_raw = np.concatenate(y_raw_list, axis=-1)
            
            # Normalize inputs using training statistics
            x_norm = np.empty_like(x_raw)
            for c_idx, var_name in enumerate(config.INPUT_VARIABLES):
                if self.norm_method == "zscore":
                    mean = self.norm_stats[var_name]["mean"]
                    std = self.norm_stats[var_name]["std"]
                    x_norm[..., c_idx] = (x_raw[..., c_idx] - mean) / std
                else:
                    v_min = self.norm_stats[var_name]["min"]
                    v_range = self.norm_stats[var_name]["range"]
                    x_norm[..., c_idx] = (x_raw[..., c_idx] - v_min) / v_range

            extra_channels = []
            # S2: Cyclical time encodings: sin/cos hour-of-day, sin/cos day-of-year
            if self.include_cyclical_time:
                h_val = input_end_t.hour
                d_val = input_end_t.dayofyear
                h_sin = np.sin(2.0 * np.pi * h_val / 24.0)
                h_cos = np.cos(2.0 * np.pi * h_val / 24.0)
                d_sin = np.sin(2.0 * np.pi * d_val / 365.25)
                d_cos = np.cos(2.0 * np.pi * d_val / 365.25)
                # Broadcast across (lags, 28, 34, 4)
                time_feats = np.zeros((self.lags, config.ORIGINAL_LAT, config.ORIGINAL_LON, 4), dtype=np.float32)
                time_feats[..., 0] = h_sin
                time_feats[..., 1] = h_cos
                time_feats[..., 2] = d_sin
                time_feats[..., 3] = d_cos
                extra_channels.append(time_feats)

            # S4: Static terrain channel: proxy ln(msl / sp)
            if self.include_terrain:
                # msl index = 2, sp index = 3
                msl_val = np.maximum(x_raw[..., 2:3], 100.0)
                sp_val = np.maximum(x_raw[..., 3:4], 100.0)
                terrain = np.log(msl_val / sp_val)
                # Normalize terrain proxy (~ mean 0.03, std 0.04)
                terrain_norm = (terrain - 0.03) / 0.04
                extra_channels.append(terrain_norm.astype(np.float32))

            if len(extra_channels) > 0:
                x_norm = np.concatenate([x_norm] + extra_channels, axis=-1)
                    
            # Normalize target (t2m)
            t_var = config.TARGET_VARIABLE
            if self.norm_method == "zscore":
                t_mean = self.norm_stats[t_var]["mean"]
                t_std = self.norm_stats[t_var]["std"]
                y_norm = (y_raw - t_mean) / t_std
            else:
                t_min = self.norm_stats[t_var]["min"]
                t_range = self.norm_stats[t_var]["range"]
                y_norm = (y_raw - t_min) / t_range
                
            # Pad spatial dimensions to (32, 48)
            x_padded = self.dm.pad_spatial(x_norm)  # (6, 32, 48, C_in)
            y_padded = self.dm.pad_spatial(y_norm)  # (1, 32, 48, num_horizons)
            
            if self.dim_model == "3d":
                x_batch.append(x_padded)
                y_batch.append(y_padded)
            elif self.dim_model == "2d":
                # Flatten temporal lags into channels
                x_2d = np.transpose(x_padded, (1, 2, 0, 3)).reshape((config.PADDED_LAT, config.PADDED_LON, -1))
                y_2d = y_padded[0]
                x_batch.append(x_2d)
                y_batch.append(y_2d)

        return np.asarray(x_batch, dtype=np.float32), np.asarray(y_batch, dtype=np.float32)

    def get_sample_raw(self, sample_idx: int):
        """
        Returns raw (unnormalized, unpadded) physical values in Kelvin:
        (x_raw, y_raw_dict, persistence_dict, input_end_time, target_times_dict)
        """
        y, start_i, input_end_t, target_times = self.windows[sample_idx]
        year_arr = self.dm.get_year_data(y)
        
        # Raw historical input: (6, 28, 34, 6)
        x_raw = year_arr[start_i : start_i + self.lags].copy()
        
        # Persistence prediction is the last observed t2m frame: x_raw[-1, :, :, TARGET_INDEX]
        last_observed_t2m = x_raw[-1, :, :, config.TARGET_INDEX].copy()  # (28, 34)
        
        y_raw_dict = {}
        target_times_dict = {}
        for h_idx, h in enumerate(self.horizons):
            target_i = start_i + self.lags + h - 1
            y_raw_dict[h] = year_arr[target_i, :, :, config.TARGET_INDEX].copy()  # (28, 34)
            target_times_dict[h] = target_times[h_idx]
            
        return x_raw, y_raw_dict, last_observed_t2m, input_end_t, target_times_dict
