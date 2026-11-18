"""Time-dependent speed, load-dependent energy, and inverse energy calculations; operating time is in minutes."""

from __future__ import annotations
import numpy as np


class EnergyModel:
    def __init__(
        self,
        phi_motor=1.184692,
        phi_battery=1.112434,
        g=9.8,
        theta_ij=0,
        C_r=0.012,
        L=3000,
        R=0.7,
        A=3.8,
        rho=1.2041,
        clock_mode="legacy",
    ):
        """Initialize the energy model.

        phi_motor: Motor output efficiency factor (phi^d).
        phi_battery: Battery output efficiency factor (phi^a).
        g: Gravitational acceleration (m/s^2).
        theta_ij: Road slope (radians).
        C_r: Rolling resistance coefficient.
        L: Vehicle curb weight (kg).
        R: Aerodynamic drag coefficient (Cd).
        A: Frontal area (m^2).
        rho: Air density (kg/m^3)."""
        self.phi_motor = phi_motor
        self.phi_battery = phi_battery
        self.g = g
        self.theta_ij = theta_ij
        self.C_r = C_r
        self.L = L
        self.R = R
        self.A = A
        self.rho = rho
        if clock_mode not in ("legacy", "relative_07_v1"):
            raise ValueError("Unknown traffic clock convention")
        self.clock_mode = clock_mode
        self.start_hour = 7.0 if clock_mode == "relative_07_v1" else 0.0

    def calculate_energy_consumption(self, u_ijk, v_ijk_R, t_ijk_R):
        """Compute energy consumed on arc (i, j).

        u_ijk is current payload (kg), v_ijk_R is speed (km/h),
        and t_ijk_R is travel time (h). Return consumed energy."""
        resistance_energy = (
            (self.g * np.sin(self.theta_ij) + self.C_r * self.g * np.cos(self.theta_ij))
            * (self.L + u_ijk)
            / 3600.0
        )
        aerodynamic_drag_energy = self.R * self.A * self.rho * v_ijk_R**2 / 76140.0
        e_ijk_R = (
            self.phi_motor
            * self.phi_battery
            * (resistance_energy + aerodynamic_drag_energy)
            * v_ijk_R
            * t_ijk_R
        )
        return e_ijk_R

    def calculate_one_node_energy_time(
        self, distance, departure_time, u_ijk, *, segments=None
    ):
        max_minutes = 1440
        period_length = 60
        total_energy = 0.0
        total_travel_time = 0.0
        remaining_distance = distance
        travel_speed = 0
        while remaining_distance > 0:
            current_time = departure_time % max_minutes
            period = int(current_time // period_length)
            period_end = (period + 1) * period_length
            current_hour = departure_time / 60 % 24
            congested = 0 <= current_hour <= 2 or 10 <= current_hour <= 12
            if self.clock_mode == "relative_07_v1":
                hour = (current_hour + self.start_hour) % 24
                congested = 7 <= hour < 9 or 17 <= hour < 19
            if congested:
                travel_speed = 30
            else:
                travel_speed = 60
            time_available = period_end - current_time
            max_travel_time = remaining_distance / travel_speed * 60.0
            t_ijk_R = min(time_available, max_travel_time)
            energy_this_period = self.calculate_energy_consumption(
                u_ijk, travel_speed, t_ijk_R / 60.0
            )
            if segments is not None:
                segments.append(
                    {
                        "start_time_min": float(departure_time),
                        "duration_min": float(t_ijk_R),
                        "distance_km": float(travel_speed * t_ijk_R / 60.0),
                        "speed_kmh": float(travel_speed),
                        "energy_kwh": float(energy_this_period),
                    }
                )
            total_energy += energy_this_period
            total_travel_time += t_ijk_R
            remaining_distance -= travel_speed * (t_ijk_R / 60.0)
            departure_time += t_ijk_R
        return (total_energy, total_travel_time, travel_speed)

    def calculate_charging_time(self, q_ik):
        charging_time = max(q_ik, 0.0)
        return charging_time * 60 / (0.9 * 60)
