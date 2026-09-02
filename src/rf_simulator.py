import numpy as np
import pandas as pd


class RFEnvironment:
    def __init__(
        self,
        num_bands=50,
        num_time_slots=1000,
        seed=42
    ):
        self.num_bands = num_bands
        self.num_time_slots = num_time_slots
        self.rng = np.random.default_rng(seed)

    def generate(self):
        """
        Generate synthetic RF environment.

        1 = transmission
        0 = no transmission
        """

        environment = np.zeros(
            (self.num_time_slots, self.num_bands),
            dtype=int
        )

        # Generate activity for each simulated emitter/band.
        for band in range(self.num_bands):

            activity_probability = self.rng.uniform(
                0.05, 0.30
            )

            activity = (
                self.rng.random(self.num_time_slots)
                < activity_probability
            )

            environment[:, band] = activity.astype(int)

        return environment

    def to_dataframe(self, environment):
        records = []

        for time_slot in range(self.num_time_slots):
            for band in range(self.num_bands):

                records.append({
                    "time_slot": time_slot,
                    "band": band,
                    "transmission": environment[
                        time_slot, band
                    ]
                })

        return pd.DataFrame(records)


if __name__ == "__main__":

    simulator = RFEnvironment(
        num_bands=50,
        num_time_slots=1000
    )

    environment = simulator.generate()

    df = simulator.to_dataframe(environment)

    print("\nRF Environment created")
    print("----------------------")
    print(f"Frequency bands : {simulator.num_bands}")
    print(f"Time slots      : {simulator.num_time_slots}")
    print(f"Total samples   : {len(df)}")

    print("\nFirst observations:")
    print(df.head(20))

    df.to_csv(
        "data/rf_environment.csv",
        index=False
    )

    print("\nSaved:")
    print("data/rf_environment.csv")