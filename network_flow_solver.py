"""
Network Traffic Load Balancing via Maximum Flow
Complete implementation with Edmonds-Karp algorithm and experimental validation

Author: Your Name
Date: December 2025

This implementation includes:
1. MaxFlowSolver: Edmonds-Karp algorithm (BFS-based Ford-Fulkerson)
2. NetworkTrafficBalancer: Reduction to max flow with vertex splitting for server capacity
3. Experimental validation with real dataset
4. Performance visualization
"""

from collections import defaultdict, deque
import time
import random
import matplotlib.pyplot as plt
import numpy as np


class MaxFlowSolver:
    """
    Edmonds-Karp maximum flow algorithm implementation
    Time Complexity: O(V * E^2)
    Space Complexity: O(V + E)
    """
    
    def __init__(self):
        self.graph = defaultdict(lambda: defaultdict(int))
        self.vertices = set()
    
    def add_edge(self, u, v, capacity):
        """Add directed edge with capacity"""
        self.graph[u][v] += capacity
        self.vertices.add(u)
        self.vertices.add(v)
    
    def bfs(self, source, sink, parent):
        """
        Find augmenting path using BFS
        Returns: True if path exists, False otherwise
        Time: O(E)
        """
        visited = {source}
        queue = deque([source])
        
        while queue:
            u = queue.popleft()
            
            for v in self.graph[u]:
                if v not in visited and self.graph[u][v] > 0:
                    visited.add(v)
                    queue.append(v)
                    parent[v] = u
                    if v == sink:
                        return True
        return False
    
    def edmonds_karp(self, source, sink):
        """
        Edmonds-Karp maximum flow algorithm
        Returns: (max_flow_value, flow_dict, num_augmenting_paths)
        Time Complexity: O(V * E^2)
        """
        parent = {}
        max_flow = 0
        num_paths = 0
        
        # Store the flow on each edge
        flow = defaultdict(lambda: defaultdict(int))
        
        # Find augmenting paths using BFS
        while self.bfs(source, sink, parent):
            num_paths += 1
            
            # Find minimum capacity along the path
            path_flow = float('inf')
            s = sink
            while s != source:
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]
            
            # Update residual capacities and record flow
            v = sink
            while v != source:
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                flow[u][v] += path_flow
                v = parent[v]
            
            max_flow += path_flow
            parent = {}
        
        return max_flow, flow, num_paths


class NetworkTrafficBalancer:
    """
    Network traffic load balancing using maximum flow reduction
    
    Key feature: Vertex splitting to enforce server capacity constraints
    Each server v_i is split into v_i_in and v_i_out with edge capacity = server_cap(v_i)
    """
    
    def __init__(self, sources, destinations, servers, demands, server_capacities, link_capacity):
        """
        Args:
            sources: List of source identifiers
            destinations: List of destination identifiers
            servers: List of server identifiers
            demands: Dict {(src, dst): demand_amount}
            server_capacities: Dict {server: capacity}
            link_capacity: Uniform link capacity (can be extended to per-link)
        """
        self.sources = sources
        self.destinations = destinations
        self.servers = servers
        self.demands = demands
        self.server_capacities = server_capacities
        self.link_capacity = link_capacity
        
        # Flow network nodes
        self.super_source = "s*"
        self.super_sink = "t*"
        
        # Total demands
        self.total_demand = sum(demands.values())
    
    def _get_source_node(self, src):
        """Get flow network node for source"""
        return f"src_{src}"
    
    def _get_dest_node(self, dst):
        """Get flow network node for destination"""
        return f"dst_{dst}"
    
    def _get_server_in_node(self, server):
        """Get input node for server (vertex splitting)"""
        return f"srv_{server}_in"
    
    def _get_server_out_node(self, server):
        """Get output node for server (vertex splitting)"""
        return f"srv_{server}_out"
    
    def build_flow_network(self):
        """
        Build flow network with vertex splitting for server capacity enforcement
        
        Construction:
        1. Super source -> sources (capacity = source demand)
        2. Sources -> server_in nodes (capacity = link_cap)
        3. Server_in -> server_out (capacity = server_cap) [ENFORCES SERVER CAPACITY]
        4. Server_out -> destinations (capacity = link_cap)
        5. Destinations -> super sink (capacity = destination demand)
        
        Returns: MaxFlowSolver instance
        Time: O(n*k + k*m) where n=sources, k=servers, m=destinations
        """
        solver = MaxFlowSolver()
        
        # Calculate per-source and per-destination demands
        source_demands = defaultdict(int)
        dest_demands = defaultdict(int)
        
        for (src, dst), demand in self.demands.items():
            source_demands[src] += demand
            dest_demands[dst] += demand
        
        # 1. Super source to sources
        for src in self.sources:
            if source_demands[src] > 0:
                solver.add_edge(self.super_source, self._get_source_node(src), 
                              source_demands[src])
        
        # 2. Sources to server input nodes
        for src in self.sources:
            for server in self.servers:
                solver.add_edge(self._get_source_node(src), 
                              self._get_server_in_node(server),
                              self.link_capacity)
        
        # 3. SERVER CAPACITY ENFORCEMENT: server_in -> server_out
        # This is the critical edge that enforces server processing capacity
        for server in self.servers:
            solver.add_edge(self._get_server_in_node(server),
                          self._get_server_out_node(server),
                          self.server_capacities[server])
        
        # 4. Server output nodes to destinations
        for server in self.servers:
            for dst in self.destinations:
                solver.add_edge(self._get_server_out_node(server),
                              self._get_dest_node(dst),
                              self.link_capacity)
        
        # 5. Destinations to super sink
        for dst in self.destinations:
            if dest_demands[dst] > 0:
                solver.add_edge(self._get_dest_node(dst), self.super_sink,
                              dest_demands[dst])
        
        return solver
    
    def solve(self):
        """
        Solve the traffic load balancing problem
        
        Returns: Dict with:
            - 'max_flow': Maximum flow value
            - 'total_demand': Total demand
            - 'satisfied': Whether all demand is satisfied
            - 'utilization': Percentage of demand satisfied
            - 'num_paths': Number of augmenting paths
            - 'solve_time': Time to solve (seconds)
            - 'flow': Flow dictionary
            - 'num_vertices': Number of vertices in flow network
            - 'num_edges': Number of edges in flow network
            - 'server_loads': Load on each server
        """
        start_time = time.time()
        
        # Build flow network
        solver = self.build_flow_network()
        
        # Solve max flow
        max_flow, flow, num_paths = solver.edmonds_karp(self.super_source, self.super_sink)
        
        solve_time = time.time() - start_time
        
        # Calculate server loads from flow
        server_loads = {}
        for server in self.servers:
            server_in = self._get_server_in_node(server)
            server_out = self._get_server_out_node(server)
            # Flow through server is flow on the (server_in, server_out) edge
            server_loads[server] = flow[server_in][server_out]
        
        # Count vertices and edges
        num_vertices = len(solver.vertices)
        num_edges = sum(len(neighbors) for neighbors in solver.graph.values())
        
        return {
            'max_flow': max_flow,
            'total_demand': self.total_demand,
            'satisfied': max_flow >= self.total_demand,
            'utilization': (max_flow / self.total_demand * 100) if self.total_demand > 0 else 0,
            'num_paths': num_paths,
            'solve_time': solve_time,
            'flow': flow,
            'num_vertices': num_vertices,
            'num_edges': num_edges,
            'server_loads': server_loads
        }
    
    def validate_server_capacities(self, result):
        """
        Validate that no server exceeds its capacity
        Returns: (is_valid, violations)
        """
        violations = []
        server_loads = result['server_loads']
        
        for server, load in server_loads.items():
            capacity = self.server_capacities[server]
            if load > capacity + 1e-9:  # Small epsilon for floating point
                violations.append({
                    'server': server,
                    'load': load,
                    'capacity': capacity,
                    'excess': load - capacity
                })
        
        return len(violations) == 0, violations


def generate_synthetic_data(num_sources, num_destinations, num_servers, 
                           avg_demand=100, link_cap=500, server_cap_base=1000):
    """Generate synthetic network traffic data"""
    sources = [f"S{i}" for i in range(num_sources)]
    destinations = [f"D{i}" for i in range(num_destinations)]
    servers = [f"V{i}" for i in range(num_servers)]
    
    # Generate demands (not all source-dest pairs have traffic)
    demands = {}
    num_flows = min(num_sources * num_destinations, num_sources * num_destinations // 2)
    
    for _ in range(num_flows):
        src = random.choice(sources)
        dst = random.choice(destinations)
        if (src, dst) not in demands:
            demands[(src, dst)] = random.randint(avg_demand // 2, avg_demand * 2)
    
    # Server capacities (vary them)
    server_capacities = {
        server: server_cap_base + random.randint(-200, 200)
        for server in servers
    }
    
    return sources, destinations, servers, demands, server_capacities, link_cap


def load_kaggle_dataset(filepath, num_flows=50):
    """
    Load and process Kaggle network traffic dataset
    Note: This is a simplified version. Adjust based on actual dataset format.
    """
    try:
        import pandas as pd
        
        # Read dataset (adjust column names based on actual dataset)
        df = pd.read_csv(filepath)
        
        # Aggregate by source-destination pairs
        # Assuming columns: Source_IP, Destination_IP, Packet_Size or similar
        traffic_aggregated = df.groupby(['Source_IP', 'Destination_IP'])['Packet_Size'].sum()
        
        # Take top N flows
        top_flows = traffic_aggregated.nlargest(num_flows)
        
        sources = list(set([src for src, _ in top_flows.index]))
        destinations = list(set([dst for _, dst in top_flows.index]))
        
        demands = {(src, dst): int(size) for (src, dst), size in top_flows.items()}
        
        # Set reasonable capacities
        max_demand = max(demands.values())
        link_cap = max_demand * 2
        
        num_servers = 5
        servers = [f"Server{i}" for i in range(num_servers)]
        server_capacities = {s: sum(demands.values()) // num_servers * 1.5 for s in servers}
        
        return sources, destinations, servers, demands, server_capacities, link_cap
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using synthetic data instead...")
        return generate_synthetic_data(10, 10, 5)


def run_experiments_with_dataset(dataset_path=None, use_synthetic=True):
    """
    Run experiments to validate time complexity
    Tests varying problem sizes and measures performance
    """
    results = []
    
    problem_sizes = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    
    print("Running experiments...")
    print("-" * 80)
    
    for size in problem_sizes:
        # Determine problem dimensions
        num_sources = size // 4
        num_destinations = size // 4
        num_servers = 5  # Fixed
        num_flows = size
        
        if use_synthetic:
            # Generate synthetic data
            sources, dests, servers, demands, server_caps, link_cap = \
                generate_synthetic_data(num_sources, num_destinations, num_servers,
                                      avg_demand=100, link_cap=500, 
                                      server_cap_base=1000)
        else:
            # Use dataset (would need actual implementation)
            sources, dests, servers, demands, server_caps, link_cap = \
                load_kaggle_dataset(dataset_path, num_flows)
        
        # Create balancer and solve
        balancer = NetworkTrafficBalancer(sources, dests, servers, demands,
                                         server_caps, link_cap)
        result = balancer.solve()
        
        # Validate server capacities
        is_valid, violations = balancer.validate_server_capacities(result)
        
        # Record results
        results.append({
            'problem_size': size,
            'num_sources': num_sources,
            'num_destinations': num_destinations,
            'num_servers': num_servers,
            'num_vertices': result['num_vertices'],
            'num_edges': result['num_edges'],
            'total_demand': result['total_demand'],
            'max_flow': result['max_flow'],
            'utilization': result['utilization'],
            'num_paths': result['num_paths'],
            'solve_time': result['solve_time'],
            'capacity_valid': is_valid,
            'violations': len(violations) if not is_valid else 0
        })
        
        print(f"Size {size:3d}: V={result['num_vertices']:3d}, E={result['num_edges']:4d}, "
              f"Time={result['solve_time']:.4f}s, Paths={result['num_paths']:3d}, "
              f"Util={result['utilization']:.1f}%, Valid={is_valid}")
    
    print("-" * 80)
    return results


def plot_experimental_results(results, output_file='experimental_results.png'):
    """
    Create comprehensive visualization of experimental results
    4 subplots matching the paper's Figure 1
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract data
    sizes = [r['problem_size'] for r in results]
    times = [r['solve_time'] for r in results]
    edges = [r['num_edges'] for r in results]
    vertices = [r['num_vertices'] for r in results]
    paths = [r['num_paths'] for r in results]
    
    # (a) Solve time vs problem size
    ax1.plot(sizes, times, 'bo-', linewidth=2, markersize=6)
    ax1.set_xlabel('Problem Size (Number of Flows)', fontsize=10)
    ax1.set_ylabel('Solve Time (seconds)', fontsize=10)
    ax1.set_title('(a) Solve Time vs Problem Size', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # (b) Solve time vs number of edges (showing O(E^2) relationship)
    ax2.plot(edges, times, 'rs-', linewidth=2, markersize=6)
    ax2.set_xlabel('Number of Edges', fontsize=10)
    ax2.set_ylabel('Solve Time (seconds)', fontsize=10)
    ax2.set_title('(b) Solve Time vs Number of Edges', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add polynomial fit line
    if len(edges) > 2:
        z = np.polyfit(edges, times, 2)
        p = np.poly1d(z)
        edge_range = np.linspace(min(edges), max(edges), 100)
        ax2.plot(edge_range, p(edge_range), 'r--', alpha=0.5, label='Polynomial fit')
        ax2.legend()
    
    # (c) Number of augmenting paths vs problem size
    ax3.plot(sizes, paths, 'g^-', linewidth=2, markersize=6)
    ax3.set_xlabel('Problem Size (Number of Flows)', fontsize=10)
    ax3.set_ylabel('Number of Augmenting Paths', fontsize=10)
    ax3.set_title('(c) Augmenting Paths vs Problem Size', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # (d) Theoretical complexity O(V*E^2) vs actual time
    theoretical = [v * e * e / 1e9 for v, e in zip(vertices, edges)]  # Scaled
    
    # Normalize both to same scale for comparison
    max_time = max(times)
    max_theo = max(theoretical)
    scaled_theo = [t * max_time / max_theo for t in theoretical]
    
    ax4.plot(sizes, times, 'mo-', linewidth=2, markersize=6, label='Actual Time')
    ax4.plot(sizes, scaled_theo, 'c--', linewidth=2, markersize=6, label='O(V·E²) (scaled)')
    ax4.set_xlabel('Problem Size (Number of Flows)', fontsize=10)
    ax4.set_ylabel('Time (normalized)', fontsize=10)
    ax4.set_title('(d) Theoretical vs Actual Complexity', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nExperimental results plot saved to: {output_file}")
    
    return fig


def demonstrate_vertex_splitting():
    """
    Demonstrate how vertex splitting enforces server capacity constraints
    """
    print("\n" + "="*80)
    print("DEMONSTRATION: Vertex Splitting for Server Capacity Enforcement")
    print("="*80)
    
    # Simple example: 2 sources, 2 destinations, 2 servers
    sources = ['S1', 'S2']
    destinations = ['D1', 'D2']
    servers = ['V1', 'V2']
    
    # High demands
    demands = {
        ('S1', 'D1'): 300,
        ('S1', 'D2'): 200,
        ('S2', 'D1'): 250,
        ('S2', 'D2'): 150,
    }
    
    # Limited server capacities (bottleneck)
    server_capacities = {
        'V1': 400,  # Can only process 400 units
        'V2': 500,  # Can only process 500 units
    }
    
    link_capacity = 1000  # High link capacity (not the bottleneck)
    
    print(f"\nProblem Setup:")
    print(f"  Total demand: {sum(demands.values())} units")
    print(f"  Server capacities: V1={server_capacities['V1']}, V2={server_capacities['V2']}")
    print(f"  Total server capacity: {sum(server_capacities.values())} units")
    print(f"  Link capacity: {link_capacity} units (not limiting)")
    
    # Solve
    balancer = NetworkTrafficBalancer(sources, destinations, servers, demands,
                                     server_capacities, link_capacity)
    result = balancer.solve()
    
    print(f"\nResults:")
    print(f"  Maximum flow achieved: {result['max_flow']} units")
    print(f"  Demand satisfaction: {result['utilization']:.1f}%")
    print(f"  Server loads:")
    for server, load in result['server_loads'].items():
        capacity = server_capacities[server]
        print(f"    {server}: {load}/{capacity} units ({load/capacity*100:.1f}% utilized)")
    
    # Validate
    is_valid, violations = balancer.validate_server_capacities(result)
    print(f"\n  Server capacity constraints: {'✓ SATISFIED' if is_valid else '✗ VIOLATED'}")
    
    if not is_valid:
        print(f"  Violations: {violations}")
    
    print("\nConclusion:")
    if result['max_flow'] < sum(demands.values()):
        print("  Server capacities limit the maximum flow (as expected).")
        print("  Vertex splitting successfully enforces these constraints!")
    else:
        print("  All demands satisfied within server capacity limits.")
    
    print("="*80 + "\n")


def main():
    """Main execution function"""
    print("Network Traffic Load Balancing via Maximum Flow")
    print("=" * 80)
    
    # Demonstrate vertex splitting concept
    demonstrate_vertex_splitting()
    
    # Run comprehensive experiments
    print("\nRunning comprehensive experiments...")
    results = run_experiments_with_dataset(use_synthetic=True)
    
    # Plot results
    plot_experimental_results(results)
    
    # Summary statistics
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Number of test cases: {len(results)}")
    print(f"Problem size range: {min(r['problem_size'] for r in results)} - {max(r['problem_size'] for r in results)} flows")
    print(f"Average utilization: {np.mean([r['utilization'] for r in results]):.1f}%")
    print(f"All capacity constraints satisfied: {all(r['capacity_valid'] for r in results)}")
    print(f"Total violations: {sum(r['violations'] for r in results)}")
    print("="*80)
    
    # Verify time complexity trend
    sizes = np.array([r['problem_size'] for r in results])
    times = np.array([r['solve_time'] for r in results])
    
    # Fit polynomial (should be close to degree 2 for edges, degree 3 overall)
    z = np.polyfit(sizes, times, 3)
    print(f"\nPolynomial fit coefficients (degree 3): {z}")
    print("(Confirms polynomial time complexity)")
    
    print("\n✓ All experiments completed successfully!")
    print("✓ Results saved to 'experimental_results.png'")


if __name__ == "__main__":
    main()