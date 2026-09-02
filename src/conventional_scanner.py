from receiver import RFReceiver
from rf_simulator import RFEnvironment


def run_conventional_scan():

    print("Starting RF simulation...")

    # Create RF environment
    simulator = RFEnvironment(
        num_bands=50,
        num_time_slots=100
    )

    environment = simulator.generate()

    print("RF environment created.")

    # Create receiver
    receiver = RFReceiver(
        total_bandwidth_mhz=500,
        instantaneous_bandwidth_mhz=50,
        scan_time_ms=10
    )

    windows = receiver.get_scan_windows()

    print(f"Total scan windows: {len(windows)}")
    print("Starting conventional scan...\n")

    total_time = 0
    scan_count = 0

    # Scan time slots
    for time_slot in range(environment.shape[0]):

        # Scan frequency windows sequentially
        for window in windows:

            result = receiver.scan(
                environment,
                time_slot,
                window
            )

            scan_count += 1
            total_time += receiver.scan_time_ms

            print(
                f"Scan {scan_count:03d} | "
                f"Time slot: {time_slot:03d} | "
                f"Frequency: "
                f"{window[0]}-{window[1]} MHz | "
                f"Result: "
                f"{'HIT' if result['hit'] else 'MISS'}"
            )

            # Stop at first detected signal
            if result["hit"] == 1:

                print("\n==============================")
                print("       SIGNAL DETECTED")
                print("==============================")
                print(f"Frequency: {window[0]}-{window[1]} MHz")
                print(f"Time slot: {time_slot}")
                print(f"Scans performed: {scan_count}")
                print(f"Detection time: {total_time} ms")
                print("==============================")

                return


if __name__ == "__main__":
    run_conventional_scan()