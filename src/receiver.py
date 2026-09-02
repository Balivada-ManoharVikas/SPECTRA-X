import numpy as np


class RFReceiver:
    def __init__(
        self,
        total_bandwidth_mhz=500,
        instantaneous_bandwidth_mhz=50,
        scan_time_ms=10
    ):
        self.total_bandwidth_mhz = total_bandwidth_mhz
        self.instantaneous_bandwidth_mhz = instantaneous_bandwidth_mhz
        self.scan_time_ms = scan_time_ms

        self.num_windows = int(
            np.ceil(
                total_bandwidth_mhz /
                instantaneous_bandwidth_mhz
            )
        )

    def get_scan_windows(self):
        windows = []

        start = 0

        while start < self.total_bandwidth_mhz:

            end = min(
                start + self.instantaneous_bandwidth_mhz,
                self.total_bandwidth_mhz
            )

            windows.append((start, end))

            start = end

        return windows

    def scan(self, environment, time_slot, window):

        start_mhz, end_mhz = window

        total_bands = environment.shape[1]

        bands_per_mhz = total_bands / self.total_bandwidth_mhz

        start_band = int(start_mhz * bands_per_mhz)

        end_band = int(
            np.ceil(end_mhz * bands_per_mhz)
        )

        end_band = min(
            end_band,
            total_bands
        )

        observations = environment[
            time_slot,
            start_band:end_band
        ]

        hit = int(np.any(observations == 1))

        return {
            "time_slot": time_slot,
            "start_frequency_mhz": start_mhz,
            "end_frequency_mhz": end_mhz,
            "start_band": start_band,
            "end_band": end_band - 1,
            "hit": hit
        }


if __name__ == "__main__":

    receiver = RFReceiver(
        total_bandwidth_mhz=500,
        instantaneous_bandwidth_mhz=50,
        scan_time_ms=10
    )

    print("\nRF RECEIVER")
    print("-----------")

    print(
        f"Total bandwidth: "
        f"{receiver.total_bandwidth_mhz} MHz"
    )

    print(
        f"Receiver bandwidth: "
        f"{receiver.instantaneous_bandwidth_mhz} MHz"
    )

    print(
        f"Scan time: "
        f"{receiver.scan_time_ms} ms"
    )

    print(
        f"Number of windows: "
        f"{receiver.num_windows}"
    )

    print("\nScan windows:")

    for i, window in enumerate(
        receiver.get_scan_windows(),
        start=1
    ):
        print(
            f"Window {i}: "
            f"{window[0]} - {window[1]} MHz"
        )