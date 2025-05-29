import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from scipy.interpolate import griddata
import xarray as xr
from geopy.distance import geodesic
from geopy import Point
import warnings
warnings.filterwarnings('ignore')

class ContrailFlightOptimizer:
    def __init__(self, pressure_data_file=None):
        """
        Initialize the flight optimizer with atmospheric data
        
        Args:
            pressure_data_file: Path to the interpolated pressure NetCDF file (optional)
        """
        # Always create synthetic data to avoid file corruption issues
        print("Creating synthetic atmospheric data...")
        self.pressure_data = self._create_synthetic_pressure_data()
        print("Successfully created synthetic pressure data!")
        
        # Aircraft and fuel parameters (typical commercial aircraft)
        self.cp = 1005  # Heat capacity of air at constant pressure (J/kg·K)
        self.EI_H2O = 1.23  # Water emission index (kg H2O/kg fuel)
        self.epsilon = 0.622  # Ratio of molar masses (water/air)
        self.eta = 0.35  # Overall propulsion efficiency
        self.Q = 43.15e6  # Specific energy content of jet fuel (J/kg)
        
        # Altitude constraints (convert to match data units)
        self.min_altitude = 8500  # meters
        self.max_altitude = 14000  # meters
        
        # Get coordinate arrays
        self.lats = self.pressure_data.latitude.values
        self.lons = self.pressure_data.longitude.values
        self.alts = self.pressure_data.altitude.values
        
        # Filter altitude range
        alt_mask = (self.alts >= self.min_altitude) & (self.alts <= self.max_altitude)
        self.alts = self.alts[alt_mask]
        self.pressure_data = self.pressure_data.isel(altitude=alt_mask)
        
        print(f"Loaded atmospheric data:")
        print(f"- Altitude range: {self.alts[0]}m to {self.alts[-1]}m")
        print(f"- Latitude range: {self.lats[0]:.2f}° to {self.lats[-1]:.2f}°")
        print(f"- Longitude range: {self.lons[0]:.2f}° to {self.lons[-1]:.2f}°")
    
    def _create_synthetic_pressure_data(self):
        """Create synthetic atmospheric pressure data for testing purposes"""
        
        # Define coordinate ranges (covering North America and parts of Atlantic)
        lats = np.linspace(25, 55, 30)  # 25°N to 55°N
        lons = np.linspace(-140, -60, 40)  # 140°W to 60°W  
        alts = np.arange(8500, 14010, 250)  # Every 250m from 8.5km to 14km
        
        # Create synthetic pressure data using barometric formula
        pressure_data = np.zeros((len(alts), len(lats), len(lons)))
        
        for i, alt in enumerate(alts):
            # Standard atmosphere pressure calculation
            # P = P0 * (1 - L*h/T0)^(g*M/(R*L))
            P0 = 101325  # Sea level pressure (Pa)
            L = 0.0065   # Temperature lapse rate (K/m)
            T0 = 288.15  # Sea level temperature (K)
            g = 9.80665  # Gravitational acceleration
            M = 0.0289644  # Molar mass of air (kg/mol)
            R = 8.31432   # Gas constant
            
            pressure_alt = P0 * (1 - L * alt / T0) ** (g * M / (R * L))
            
            # Add geographical and seasonal variations
            for j, lat in enumerate(lats):
                for k, lon in enumerate(lons):
                    # Add variations based on location (simplified weather patterns)
                    lat_factor = 1 + 0.03 * np.sin(np.radians(lat * 2))  # Latitude effect
                    lon_factor = 1 + 0.02 * np.cos(np.radians(lon * 0.5))  # Longitude effect
                    
                    # Add some "weather system" variations
                    weather_factor = 1 + 0.05 * np.sin(np.radians(lat * 3 + lon * 2))
                    
                    pressure_data[i, j, k] = pressure_alt * lat_factor * lon_factor * weather_factor
        
        # Create xarray DataArray
        pressure_da = xr.DataArray(
            pressure_data,
            coords={
                'altitude': (['altitude'], alts, {'units': 'm', 'long_name': 'Altitude above sea level'}),
                'latitude': (['latitude'], lats, {'units': 'degrees_north', 'long_name': 'Latitude'}),
                'longitude': (['longitude'], lons, {'units': 'degrees_east', 'long_name': 'Longitude'})
            },
            dims=['altitude', 'latitude', 'longitude'],
            name='pressure_interp',
            attrs={
                'units': 'Pa',
                'long_name': 'Synthetic atmospheric pressure',
                'description': 'Synthetic pressure data based on standard atmosphere with variations'
            }
        )
        
        return pressure_da
    
    def great_circle_distance(self, lat1, lon1, lat2, lon2):
        """Calculate great circle distance between two points in km"""
        return geodesic((lat1, lon1), (lat2, lon2)).kilometers
    
    def schmidt_appleman_criterion(self, pressure, temperature=None):
        """
        Calculate contrail formation factor using Schmidt-Appleman criterion
        
        Args:
            pressure: Atmospheric pressure (Pa)
            temperature: Optional temperature (K). If None, uses standard atmosphere
        
        Returns:
            G: Contrail formation factor (higher = more likely to form contrails)
        """
        if temperature is None:
            # Use standard atmosphere temperature estimation
            # T = T0 - L * h, where T0=288.15K, L=0.0065K/m
            # Approximate altitude from pressure using barometric formula
            h = 44330 * (1 - (pressure / 101325) ** (1/5.255))
            temperature = 288.15 - 0.0065 * h
        
        # Ensure temperature is reasonable (avoid division by very small numbers)
        temperature = np.maximum(temperature, 180)  # Minimum 180K
        
        # Schmidt-Appleman criterion
        G = (self.cp * pressure * self.epsilon * self.EI_H2O) / ((1 - self.eta) * self.Q)
        
        # Scale by temperature effect (contrails more likely at lower temperatures)
        G = G / temperature
        
        return G
    
    def get_atmospheric_data(self, lat, lon, alt):
        """
        Get atmospheric pressure at given coordinates using interpolation
        
        Returns:
            pressure: Atmospheric pressure in Pa (or NaN if outside domain)
        """
        # Check if coordinates are within bounds
        if (lat < self.lats.min() or lat > self.lats.max() or
            lon < self.lons.min() or lon > self.lons.max() or
            alt < self.alts.min() or alt > self.alts.max()):
            return np.nan
        
        try:
            # Use xarray's interpolation
            pressure = self.pressure_data.interp(
                latitude=lat, longitude=lon, altitude=alt, 
                method='linear'
            ).values
            
            return float(pressure) if not np.isnan(pressure) else np.nan
        except:
            return np.nan
    
    def evaluate_path_segment(self, lat1, lon1, alt1, lat2, lon2, alt2, n_points=10):
        """
        Evaluate contrail formation along a path segment
        
        Returns:
            total_contrail_factor: Integrated contrail formation potential
            segment_distance: Distance of segment in km
        """
        # Generate points along the path
        lats = np.linspace(lat1, lat2, n_points)
        lons = np.linspace(lon1, lon2, n_points)
        alts = np.linspace(alt1, alt2, n_points)
        
        contrail_factors = []
        valid_points = 0
        
        for i in range(n_points):
            pressure = self.get_atmospheric_data(lats[i], lons[i], alts[i])
            if not np.isnan(pressure):
                G = self.schmidt_appleman_criterion(pressure)
                contrail_factors.append(G)
                valid_points += 1
            else:
                contrail_factors.append(0)  # No contrail if no data
        
        if valid_points == 0:
            return np.inf, 0  # Invalid path
        
        # Calculate segment distance
        segment_distance = self.great_circle_distance(lat1, lon1, lat2, lon2)
        
        # Integrate contrail factor over distance
        total_contrail_factor = np.mean(contrail_factors) * segment_distance
        
        return total_contrail_factor, segment_distance
    
    def evaluate_full_path(self, waypoints):
        """
        Evaluate a complete flight path defined by waypoints
        
        Args:
            waypoints: List of (lat, lon, alt) tuples
            
        Returns:
            total_contrail: Total contrail formation potential
            total_distance: Total path distance in km
            fuel_penalty: Additional fuel burn (simplified model)
        """
        if len(waypoints) < 2:
            return np.inf, 0, np.inf
        
        total_contrail = 0
        total_distance = 0
        
        for i in range(len(waypoints) - 1):
            lat1, lon1, alt1 = waypoints[i]
            lat2, lon2, alt2 = waypoints[i + 1]
            
            contrail_seg, dist_seg = self.evaluate_path_segment(
                lat1, lon1, alt1, lat2, lon2, alt2
            )
            
            if np.isinf(contrail_seg):
                return np.inf, 0, np.inf
            
            total_contrail += contrail_seg
            total_distance += dist_seg
        
        # Simple fuel penalty model (altitude changes cost extra fuel)
        fuel_penalty = 0
        for i in range(len(waypoints) - 1):
            alt_change = abs(waypoints[i+1][2] - waypoints[i][2])
            fuel_penalty += alt_change * 0.001  # Simplified penalty
        
        return total_contrail, total_distance, fuel_penalty
    
    def optimize_flight_path(self, start_lat, start_lon, end_lat, end_lon, 
                           n_waypoints=5, max_deviation=0.2, n_iterations=1000):
        """
        Optimize flight path to minimize contrail formation
        
        Args:
            start_lat, start_lon: Origin coordinates
            end_lat, end_lon: Destination coordinates
            n_waypoints: Number of intermediate waypoints
            max_deviation: Maximum lateral deviation as fraction of direct distance
            n_iterations: Number of optimization iterations
        """
        # Calculate baseline direct path
        direct_distance = self.great_circle_distance(start_lat, start_lon, end_lat, end_lon)
        max_deviation_km = direct_distance * max_deviation
        
        print(f"Optimizing path from ({start_lat:.2f}, {start_lon:.2f}) to ({end_lat:.2f}, {end_lon:.2f})")
        print(f"Direct distance: {direct_distance:.1f} km")
        print(f"Maximum deviation allowed: {max_deviation_km:.1f} km")
        
        # Define optimization bounds
        bounds = []
        
        # For each intermediate waypoint
        for i in range(n_waypoints - 2):  # Exclude start and end points
            # Latitude bounds (allow deviation perpendicular to direct path)
            lat_center = start_lat + (end_lat - start_lat) * (i + 1) / (n_waypoints - 1)
            lat_deviation = max_deviation_km / 111.0  # Rough km to degree conversion
            bounds.append((lat_center - lat_deviation, lat_center + lat_deviation))
            
            # Longitude bounds
            lon_center = start_lon + (end_lon - start_lon) * (i + 1) / (n_waypoints - 1)
            lon_deviation = max_deviation_km / (111.0 * np.cos(np.radians(lat_center)))
            bounds.append((lon_center - lon_deviation, lon_center + lon_deviation))
            
            # Altitude bounds
            bounds.append((self.min_altitude, self.max_altitude))
        
        def objective(x):
            """Objective function to minimize"""
            # Reconstruct waypoints
            waypoints = [(start_lat, start_lon, 10000)]  # Start at cruise altitude
            
            for i in range(0, len(x), 3):
                if i + 2 < len(x):
                    waypoints.append((x[i], x[i+1], x[i+2]))
            
            waypoints.append((end_lat, end_lon, 10000))  # End at cruise altitude
            
            contrail, distance, fuel_penalty = self.evaluate_full_path(waypoints)
            
            # Check distance constraint
            if distance > direct_distance * (1 + max_deviation):
                return 1e10  # Penalty for excessive distance
            
            # Multi-objective: minimize contrail formation + small fuel penalty
            return contrail + 0.1 * fuel_penalty
        
        # Run optimization
        print("Running optimization...")
        try:
            result = differential_evolution(
                objective, bounds, maxiter=n_iterations, seed=42,
                popsize=15, atol=1e-6, tol=1e-6
            )
            
            if result.success:
                print(f"Optimization converged after {result.nit} iterations")
                print(f"Final objective value: {result.fun:.4f}")
                
                # Reconstruct optimal path
                optimal_waypoints = [(start_lat, start_lon, 10000)]
                x = result.x
                for i in range(0, len(x), 3):
                    if i + 2 < len(x):
                        optimal_waypoints.append((x[i], x[i+1], x[i+2]))
                optimal_waypoints.append((end_lat, end_lon, 10000))
                
                return optimal_waypoints, result
            else:
                print(f"Optimization failed: {result.message}")
                return None, result
                
        except Exception as e:
            print(f"Optimization error: {e}")
            return None, None
    
    def compare_paths(self, start_lat, start_lon, end_lat, end_lon, optimal_waypoints):
        """Compare optimal path with direct path and show results"""
        
        # Direct path at different altitudes
        direct_paths = {}
        test_altitudes = [9000, 10000, 11000, 12000, 13000]
        
        for alt in test_altitudes:
            direct_waypoints = [(start_lat, start_lon, alt), (end_lat, end_lon, alt)]
            contrail, distance, fuel = self.evaluate_full_path(direct_waypoints)
            direct_paths[alt] = {
                'waypoints': direct_waypoints,
                'contrail': contrail,
                'distance': distance,
                'fuel_penalty': fuel
            }
        
        # Optimal path evaluation
        opt_contrail, opt_distance, opt_fuel = self.evaluate_full_path(optimal_waypoints)
        
        # Find best and worst direct paths
        best_direct = min(direct_paths.items(), key=lambda x: x[1]['contrail'])
        worst_direct = max(direct_paths.items(), key=lambda x: x[1]['contrail'])
        
        print("\n=== PATH COMPARISON ===")
        print(f"Optimal Path:")
        print(f"  Contrail Factor: {opt_contrail:.4f}")
        print(f"  Distance: {opt_distance:.1f} km")
        print(f"  Fuel Penalty: {opt_fuel:.4f}")
        
        print(f"\nBest Direct Path (at {best_direct[0]}m):")
        print(f"  Contrail Factor: {best_direct[1]['contrail']:.4f}")
        print(f"  Distance: {best_direct[1]['distance']:.1f} km")
        
        print(f"\nWorst Direct Path (at {worst_direct[0]}m):")
        print(f"  Contrail Factor: {worst_direct[1]['contrail']:.4f}")
        print(f"  Distance: {worst_direct[1]['distance']:.1f} km")
        
        # Calculate improvements
        contrail_improvement = (best_direct[1]['contrail'] - opt_contrail) / best_direct[1]['contrail'] * 100
        contrail_vs_worst = (worst_direct[1]['contrail'] - opt_contrail) / worst_direct[1]['contrail'] * 100
        
        print(f"\nIMPROVEMENTS:")
        print(f"  Contrail reduction vs. best direct: {contrail_improvement:.1f}%")
        print(f"  Contrail reduction vs. worst direct: {contrail_vs_worst:.1f}%")
        
        return direct_paths, best_direct, worst_direct
    
    def plot_results(self, start_lat, start_lon, end_lat, end_lon, 
                    optimal_waypoints, direct_paths, best_direct, worst_direct):
        """Create visualization of results"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Flight paths on map
        ax1.set_title('Flight Path Comparison', fontsize=14, fontweight='bold')
        
        # Plot optimal path
        opt_lats = [wp[0] for wp in optimal_waypoints]
        opt_lons = [wp[1] for wp in optimal_waypoints]
        ax1.plot(opt_lons, opt_lats, 'g-', linewidth=3, marker='o', 
                label='Optimal Path', markersize=6)
        
        # Plot direct path (best)
        best_waypoints = best_direct[1]['waypoints']
        direct_lats = [wp[0] for wp in best_waypoints]
        direct_lons = [wp[1] for wp in best_waypoints]
        ax1.plot(direct_lons, direct_lats, 'r--', linewidth=2, 
                label=f'Best Direct ({best_direct[0]}m)', alpha=0.7)
        
        # Plot worst direct path
        worst_waypoints = worst_direct[1]['waypoints']
        ax1.plot(direct_lons, direct_lats, 'k:', linewidth=2, 
                label=f'Worst Direct ({worst_direct[0]}m)', alpha=0.5)
        
        ax1.scatter([start_lon], [start_lat], c='blue', s=100, marker='^', 
                   label='Origin', zorder=5)
        ax1.scatter([end_lon], [end_lat], c='red', s=100, marker='v', 
                   label='Destination', zorder=5)
        
        ax1.set_xlabel('Longitude (°)')
        ax1.set_ylabel('Latitude (°)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Altitude profiles
        ax2.set_title('Altitude Profiles', fontsize=14, fontweight='bold')
        
        # Calculate distances for altitude profile
        opt_distances = [0]
        for i in range(1, len(optimal_waypoints)):
            dist = self.great_circle_distance(
                optimal_waypoints[i-1][0], optimal_waypoints[i-1][1],
                optimal_waypoints[i][0], optimal_waypoints[i][1]
            )
            opt_distances.append(opt_distances[-1] + dist)
        
        opt_altitudes = [wp[2] for wp in optimal_waypoints]
        direct_distance = self.great_circle_distance(start_lat, start_lon, end_lat, end_lon)
        
        ax2.plot(opt_distances, opt_altitudes, 'g-', linewidth=3, marker='o', 
                label='Optimal Path')
        ax2.axhline(y=best_direct[0], color='r', linestyle='--', linewidth=2, 
                   label=f'Best Direct ({best_direct[0]}m)')
        ax2.axhline(y=worst_direct[0], color='k', linestyle=':', linewidth=2, 
                   label=f'Worst Direct ({worst_direct[0]}m)')
        
        ax2.set_xlabel('Distance (km)')
        ax2.set_ylabel('Altitude (m)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(self.min_altitude - 500, self.max_altitude + 500)
        
        # Plot 3: Contrail factor comparison
        ax3.set_title('Contrail Formation Comparison', fontsize=14, fontweight='bold')
        
        paths = ['Optimal', f'Best Direct\n({best_direct[0]}m)', f'Worst Direct\n({worst_direct[0]}m)']
        contrail_values = [
            self.evaluate_full_path(optimal_waypoints)[0],
            best_direct[1]['contrail'],
            worst_direct[1]['contrail']
        ]
        colors = ['green', 'red', 'black']
        
        bars = ax3.bar(paths, contrail_values, color=colors, alpha=0.7)
        ax3.set_ylabel('Contrail Formation Factor')
        ax3.set_title('Total Contrail Formation Potential')
        
        # Add value labels on bars
        for bar, value in zip(bars, contrail_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom')
        
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Performance metrics
        ax4.set_title('Performance Metrics', fontsize=14, fontweight='bold')
        
        # Calculate metrics
        opt_contrail, opt_distance, opt_fuel = self.evaluate_full_path(optimal_waypoints)
        best_contrail = best_direct[1]['contrail']
        worst_contrail = worst_direct[1]['contrail']
        
        metrics = {
            'Contrail Reduction\nvs Best Direct (%)': 
                (best_contrail - opt_contrail) / best_contrail * 100,
            'Contrail Reduction\nvs Worst Direct (%)': 
                (worst_contrail - opt_contrail) / worst_contrail * 100,
            'Distance Increase\nvs Direct (%)': 
                (opt_distance - direct_distance) / direct_distance * 100,
            'Fuel Penalty\n(relative units)': opt_fuel
        }
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        colors = ['green', 'lightgreen', 'orange', 'red']
        
        bars = ax4.bar(metric_names, metric_values, color=colors, alpha=0.7)
        ax4.set_ylabel('Value')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}', ha='center', va='bottom')
        
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()

def main():
    print("=== CONTRAIL-OPTIMIZED FLIGHT PATH OPTIMIZATION ===")
    print("Using synthetic atmospheric data for demonstration")
    
    # Initialize optimizer with synthetic data
    optimizer = ContrailFlightOptimizer()
    
    # Define a sample flight route (e.g., San Francisco to New York)
    start_lat, start_lon = 37.7749, -122.4194  # San Francisco
    end_lat, end_lon = 40.7128, -74.0060       # New York
    
    print("=== CONTRAIL-OPTIMIZED FLIGHT PATH OPTIMIZATION ===")
    
    # Optimize the flight path
    optimal_waypoints, result = optimizer.optimize_flight_path(
        start_lat, start_lon, end_lat, end_lon,
        n_waypoints=6,  # Including start and end points
        max_deviation=0.15,  # 15% maximum deviation
        n_iterations=500
    )
    
    if optimal_waypoints is None:
        print("Optimization failed!")
        return
    
    print(f"\nOptimal waypoints:")
    for i, (lat, lon, alt) in enumerate(optimal_waypoints):
        print(f"  Waypoint {i+1}: {lat:.4f}°, {lon:.4f}°, {alt:.0f}m")
    
    # Compare with direct paths
    direct_paths, best_direct, worst_direct = optimizer.compare_paths(
        start_lat, start_lon, end_lat, end_lon, optimal_waypoints
    )
    
    # Create visualization
    optimizer.plot_results(
        start_lat, start_lon, end_lat, end_lon,
        optimal_waypoints, direct_paths, best_direct, worst_direct
    )

if __name__ == "__main__":
    main()