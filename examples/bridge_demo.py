# Demo file for the Python-LaTeX bridge.

def calculate_gravity(mass1, mass2, distance):
    """A simple function to demonstrate the bridge."""
    # BRIDGEBLOCK_START demo-concept-1
    G = 6.67430e-11  # Gravitational constant
    force = (G * mass1 * mass2) / (distance ** 2)
    return force
    # BRIDGEBLOCK_END demo-concept-1

if __name__ == "__main__":
    f = calculate_gravity(100, 200, 10)
    print(f"Calculated force: {f}")
