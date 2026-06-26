import warnings
from pathlib import Path
from typing import Optional, Literal, Union
import pandas as pd
from cwa_reader_rs import read_cwa_file, read_header


class CWADataset:
    """
    Loader for CWA files organized in participant-specific folders.

    This loader is based on the Rust implementation from the Mobilise-D project
    via the `cwa_reader_rs` Python bindings, providing fast and reliable reading of
    accelerometer CWA files.

    Each participant folder should contain CWA files for one or more sensor positions
    (e.g., wrist, lower back). Sensor type and sampling rate are inferred automatically
    from the file header, and sensor position is inferred from the filename.

    Folder structure expected:
        base_folder/
            participant_id_1/
                *.cwa
            participant_id_2/
                *.cwa
            ...

    Notes:
        -Dependencies: cwa_reader_rs, not included in pyproject.toml for now, need manual installation for this part of the code.
    """

    def __init__(
        self,
        base_folder: Union[str, Path],
        missing_sensor_error_type: Literal["raise", "warn", "ignore"] = "raise",
    ):
        """
        Initialize the dataset loader.

        Args:
            base_folder (str | Path): Path to the top-level folder containing participant subfolders.
            missing_sensor_error_type (str): Behavior when sensor info or file is missing.
                - "raise": raise an exception
                - "warn": log a warning
                - "ignore": silently skip
        """

        self.base_folder = Path(base_folder)
        self.missing_sensor_error_type = missing_sensor_error_type
        self.error_log = []

    def _process_file(
        self, file_path: Path
    ) -> tuple[Optional[pd.DataFrame], str, Optional[float]]:
        """
        Read a single CWA file and extract sensor data, sampling rate, and hardware type.

        Args:
            file_path (Path): Path to the CWA file.

        Returns:
            tuple:
                - pd.DataFrame or None: DataFrame with time and sensor readings.
                - str: Hardware type (e.g., 'AX6') inferred from header.
                - float or None: Sampling rate in Hz inferred from header.
        """

        try:
            header = read_header(str(file_path))
            sensor_type = header.get("hardware_type", "Unknown")
            sampling_rate = header.get("sample_rate_hz", None)

            data = read_cwa_file(
                str(file_path),
                include_magnetometer=False,
                include_temperature=False,
                include_light=False,
                include_battery=False,
            )
            df = pd.DataFrame(data)
            df["time"] = (df["timestamp"].astype("int64") * 1000).astype(
                "datetime64[ns]"
            )
            df = df[["time"] + [c for c in df.columns if c != "time"]]
            df = df.drop(columns=["timestamp"])
            return df, sensor_type, sampling_rate
        except Exception as e:
            msg = f"Error reading {file_path}: {e}"
            self.error_log.append(msg)
            if self.missing_sensor_error_type == "raise":
                raise RuntimeError(msg)
            elif self.missing_sensor_error_type == "warn":
                warnings.warn(msg)
            return None, "Unknown", None

    def _get_sensor_position_from_filename(self, file_path: Path) -> str:
        """
        Infer sensor position from the CWA file name.

        Args:
            file_path (Path): Path to the CWA file.

        Returns:
            str: Sensor position (e.g., "Wrist", "LowerBack") or "Unknown" if not recognized.
        """

        fname = file_path.name.lower()
        if any(k in fname for k in ["wrist", "wr"]):
            return "Wrist"
        elif any(k in fname for k in ["lowback", "lb"]):
            return "LowerBack"
        else:
            msg = f"Unknown sensor position for file {file_path}"
            self.error_log.append(msg)
            if self.missing_sensor_error_type == "raise":
                raise ValueError(msg)
            elif self.missing_sensor_error_type == "warn":
                warnings.warn(msg)
            return "Unknown"

    @property
    def participant_ids(self) -> list[str]:
        """
        List all participant IDs based on folder names.

        Returns:
            list[str]: List of participant folder names.
        """

        return [f.name for f in self.base_folder.iterdir() if f.is_dir()]

    def load(self) -> dict[str, dict[str, dict[str, Union[pd.DataFrame, float, str]]]]:
        """
        Load all CWA files for all participants.

        Returns:
            dict: Nested dictionary structured as:
                {participant_id:
                    {sensor_position:
                        {"data": pd.DataFrame, "sampling_rate": float, "hardware_type": str}
                    }
                }
        """

        result = {}
        for pid in self.participant_ids:
            result[pid] = {}
            folder = self.base_folder / pid
            for file in folder.glob("*.cwa"):
                df, sensor_type, sampling_rate = self._process_file(file)
                sensor_pos = self._get_sensor_position_from_filename(file)
                if df is not None:
                    result[pid][sensor_pos] = {
                        "data": df,
                        "sampling_rate": sampling_rate,
                        "hardware_type": sensor_type,
                    }
        return result

    def get_sensor_data(
        self, participant_id: str, sensor: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Return the DataFrame for a given participant and sensor.

        If the sensor is not available, a warning is issued. Valid options are 'Wrist' and 'LowerBack'.
        """
        data = self.load()
        if participant_id not in data:
            warnings.warn(f"No data found for participant {participant_id}")
            return None

        # If sensor not specified, return the first available
        if sensor is None:
            sensor = next(iter(data[participant_id].keys()), None)

        if sensor not in data[participant_id]:
            warnings.warn(
                f"Sensor '{sensor}' not available for participant {participant_id}. "
                f"Available sensors: {list(data[participant_id].keys())}"
            )
            return None

        return data[participant_id][sensor]["data"]

    def get_sampling_rate(
        self, participant_id: str, sensor: Optional[str] = None
    ) -> Optional[float]:
        """
        Retrieve the sampling rate for a participant's sensor.

        Args:
            participant_id (str): Participant ID.
            sensor (str, optional): Sensor position. Defaults to first available sensor if None.

        Returns:
            float or None: Sampling rate in Hz.
        """

        data = self.load()
        if participant_id not in data:
            warnings.warn(f"No data found for participant {participant_id}")
            return None
        if sensor is None:
            sensor = next(iter(data[participant_id].keys()), None)
        return data[participant_id].get(sensor, {}).get("sampling_rate")

    def get_hardware_type(
        self, participant_id: str, sensor: Optional[str] = None
    ) -> Optional[str]:
        """
        Retrieve the hardware type for a participant's sensor.

        Args:
            participant_id (str): Participant ID.
            sensor (str, optional): Sensor position. Defaults to first available sensor if None.

        Returns:
            str or None: Hardware type (e.g., 'AX6').
        """

        data = self.load()
        if participant_id not in data:
            warnings.warn(f"No data found for participant {participant_id}")
            return None
        if sensor is None:
            sensor = next(iter(data[participant_id].keys()), None)
        return data[participant_id].get(sensor, {}).get("hardware_type")
