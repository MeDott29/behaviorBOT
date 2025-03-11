import networkx as nx
import matplotlib.pyplot as plt
import re
import random
import numpy as np
from collections import defaultdict, Counter

class AgenticGraphReasoning:
    """
    Implementation of the Agentic Deep Graph Reasoning framework based on the paper.
    This class handles the iterative knowledge graph construction and expansion.
    """
    
    def __init__(self, initial_prompt=None):
        """Initialize the knowledge graph and set the initial prompt."""
        # Create a directed graph for knowledge representation
        self.graph = nx.DiGraph()
        self.initial_prompt = initial_prompt
        self.iteration = 0
        self.history = []
        
        # Track centrality metrics over time
        self.centrality_metrics = {
            'betweenness': [],
            'closeness': [],
            'eigenvector': [],
            'degree': []
        }
        
        # Graph statistics over time
        self.stats = {
            'node_count': [],
            'edge_count': [],
            'avg_degree': [],
            'max_degree': [],
            'lcc_size': [],  # Largest Connected Component
            'avg_clustering': [],
            'modularity': [],
            'avg_path_length': [],
            'diameter': []
        }
    
    def extract_reasoning_graph(self, reasoning_text):
        """
        Extract nodes and relationships from reasoning text.
        
        This is a simplified version - in the real implementation from the paper, 
        this would use more sophisticated NLP techniques.
        """
        # Simple pattern matching for concept extraction
        # Format expected: "ConceptA -- RELATION --> ConceptB"
        pattern = r'([^-]+)--\s*([A-Z-]+)\s*-->\s*([^-]+)'
        matches = re.findall(pattern, reasoning_text)
        
        local_graph = nx.DiGraph()
        
        for match in matches:
            source = match[0].strip()
            relation = match[1].strip()
            target = match[2].strip()
            
            # Add nodes and edge to the local graph
            local_graph.add_node(source)
            local_graph.add_node(target)
            local_graph.add_edge(source, target, relation=relation)
        
        return local_graph
    
    def generate_reasoning(self, prompt):
        """
        Generate reasoning tokens from a prompt.
        
        In the actual implementation, this would call a language model.
        Here we'll simulate it with a simple function that generates random concepts.
        """
        # This is a placeholder - in real implementation would call LLM
        concepts = [
            "Materials Science", "Carbon Nanotubes", "Self-healing Materials",
            "Machine Learning", "Smart Infrastructure", "Sustainability",
            "AI Techniques", "Predictive Modeling", "Impact-Resistant Materials",
            "Adaptive Systems", "Resilience", "Environmental Footprint",
            "Knowledge Discovery", "Graph Theory", "Nanotechnology",
            "Composite Materials", "Sensor Networks", "Data Analysis"
        ]
        
        relations = ["IS-A", "INFLUENCES", "RELATES-TO", "PART-OF", "USES"]
        
        # Generate random graph (this simulates the LLM output)
        num_edges = random.randint(3, 7)
        reasoning_text = ""
        
        for _ in range(num_edges):
            source = random.choice(concepts)
            target = random.choice([c for c in concepts if c != source])
            relation = random.choice(relations)
            
            reasoning_text += f"{source} -- {relation} --> {target}\n"
        
        # Ensure some connection to existing concepts if the graph is not empty
        if self.iteration > 0 and self.graph.number_of_nodes() > 0:
            existing_node = random.choice(list(self.graph.nodes()))
            new_node = random.choice(concepts)
            relation = random.choice(relations)
            
            reasoning_text += f"{existing_node} -- {relation} --> {new_node}\n"
        
        return reasoning_text
    
    def merge_graph(self, local_graph):
        """Merge the local graph into the global knowledge graph."""
        # Add all nodes and edges from local graph to global graph
        for node in local_graph.nodes():
            if node not in self.graph:
                self.graph.add_node(node)
        
        for source, target, data in local_graph.edges(data=True):
            if not self.graph.has_edge(source, target):
                self.graph.add_edge(source, target, relation=data['relation'])
    
    def generate_follow_up_question(self, local_graph):
        """
        Generate a follow-up question based on recently extracted entities.
        
        In the actual implementation, this would call a language model.
        Here we'll use a template approach.
        """
        recent_nodes = list(local_graph.nodes())
        
        if not recent_nodes:
            return "Explore more concepts related to the initial topic."
        
        selected_node = random.choice(recent_nodes)
        
        templates = [
            f"How does {selected_node} relate to other emerging technologies?",
            f"What are the practical applications of {selected_node}?",
            f"Discuss the future implications of advances in {selected_node}.",
            f"How can {selected_node} be integrated with other concepts?",
            f"What are the key research challenges in {selected_node}?"
        ]
        
        return random.choice(templates)
    
    def compute_graph_metrics(self):
        """Compute and store various graph metrics for analysis."""
        G = self.graph
        
        # Basic stats
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        
        self.stats['node_count'].append(n_nodes)
        self.stats['edge_count'].append(n_edges)
        
        if n_nodes > 0:
            # Degree statistics
            degrees = [d for _, d in G.degree()]
            avg_degree = sum(degrees) / n_nodes
            max_degree = max(degrees) if degrees else 0
            
            self.stats['avg_degree'].append(avg_degree)
            self.stats['max_degree'].append(max_degree)
            
            # Largest connected component
            if G.is_directed():
                largest_cc = max(nx.weakly_connected_components(G), key=len)
            else:
                largest_cc = max(nx.connected_components(G), key=len)
            
            lcc_size = len(largest_cc)
            self.stats['lcc_size'].append(lcc_size)
            
            # Clustering coefficient
            try:
                avg_clustering = nx.average_clustering(G.to_undirected())
                self.stats['avg_clustering'].append(avg_clustering)
            except:
                self.stats['avg_clustering'].append(0)
            
            # For more complex metrics, we need to ensure the graph is large enough
            if n_nodes > 5 and lcc_size > 5:
                # Create a subgraph of the largest connected component
                lcc_graph = G.subgraph(largest_cc).copy()
                
                # Centrality metrics
                try:
                    betweenness = nx.betweenness_centrality(lcc_graph)
                    closeness = nx.closeness_centrality(lcc_graph)
                    eigenvector = nx.eigenvector_centrality(lcc_graph, max_iter=1000)
                    
                    self.centrality_metrics['betweenness'].append(list(betweenness.values()))
                    self.centrality_metrics['closeness'].append(list(closeness.values()))
                    self.centrality_metrics['eigenvector'].append(list(eigenvector.values()))
                    self.centrality_metrics['degree'].append([d for _, d in lcc_graph.degree()])
                except:
                    # If calculation fails, append empty lists
                    for metric in self.centrality_metrics:
                        self.centrality_metrics[metric].append([])
                
                # Path length and diameter (can be computationally expensive)
                try:
                    undirected_lcc = lcc_graph.to_undirected()
                    avg_path = nx.average_shortest_path_length(undirected_lcc)
                    diameter = nx.diameter(undirected_lcc)
                    
                    self.stats['avg_path_length'].append(avg_path)
                    self.stats['diameter'].append(diameter)
                except:
                    self.stats['avg_path_length'].append(0)
                    self.stats['diameter'].append(0)
                
                # Community detection (modularity)
                try:
                    # A simple approach to community detection
                    communities = nx.community.greedy_modularity_communities(undirected_lcc)
                    modularity = nx.community.modularity(undirected_lcc, communities)
                    self.stats['modularity'].append(modularity)
                except:
                    self.stats['modularity'].append(0)
            else:
                # For small graphs, append placeholder values
                for metric in self.centrality_metrics:
                    self.centrality_metrics[metric].append([])
                
                self.stats['avg_path_length'].append(0)
                self.stats['diameter'].append(0)
                self.stats['modularity'].append(0)
    
    def iterate(self, iterations=1):
        """Run the iterative graph expansion for a specified number of iterations."""
        prompt = self.initial_prompt
        
        for _ in range(iterations):
            # Step 1: Generate reasoning based on prompt
            reasoning_text = self.generate_reasoning(prompt)
            
            # Step 2: Extract local graph from reasoning
            local_graph = self.extract_reasoning_graph(reasoning_text)
            
            # Step 3: Merge with global graph
            self.merge_graph(local_graph)
            
            # Step 4: Generate follow-up question for next iteration
            prompt = self.generate_follow_up_question(local_graph)
            
            # Step 5: Compute and store graph metrics
            self.compute_graph_metrics()
            
            # Store history
            self.history.append({
                'iteration': self.iteration,
                'reasoning': reasoning_text,
                'local_graph': local_graph.copy(),
                'prompt': prompt
            })
            
            self.iteration += 1
    
    def visualize_graph(self, figsize=(12, 10), save_path=None):
        """Visualize the current state of the knowledge graph."""
        plt.figure(figsize=figsize)
        
        # Use a layout that works well for knowledge graphs
        pos = nx.spring_layout(self.graph, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_size=500, alpha=0.8)
        
        # Draw edges with relation labels
        edge_labels = {(u, v): d['relation'] for u, v, d in self.graph.edges(data=True)}
        nx.draw_networkx_edges(self.graph, pos, width=1.5, alpha=0.7)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=8)
        
        # Draw node labels
        nx.draw_networkx_labels(self.graph, pos, font_size=10)
        
        plt.title(f"Knowledge Graph after {self.iteration} iterations")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
    
    def plot_graph_metrics(self, figsize=(15, 10), save_path=None):
        """Plot the evolution of various graph metrics over iterations."""
        iterations = list(range(self.iteration))
        
        if not iterations:
            print("No iterations to plot metrics for.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Plot node and edge counts
        axes[0, 0].plot(iterations, self.stats['node_count'], 'b-', label='Nodes')
        axes[0, 0].plot(iterations, self.stats['edge_count'], 'r-', label='Edges')
        axes[0, 0].set_title('Graph Size')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].legend()
        
        # Plot degree statistics
        axes[0, 1].plot(iterations, self.stats['avg_degree'], 'g-', label='Average Degree')
        axes[0, 1].plot(iterations, self.stats['max_degree'], 'm-', label='Max Degree')
        axes[0, 1].set_title('Degree Statistics')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Degree')
        axes[0, 1].legend()
        
        # Plot clustering coefficient
        axes[0, 2].plot(iterations, self.stats['avg_clustering'], 'c-')
        axes[0, 2].set_title('Average Clustering Coefficient')
        axes[0, 2].set_xlabel('Iteration')
        axes[0, 2].set_ylabel('Coefficient')
        
        # Plot path length and diameter
        valid_iterations = [i for i, v in enumerate(self.stats['avg_path_length']) if v > 0]
        if valid_iterations:
            valid_path_lengths = [self.stats['avg_path_length'][i] for i in valid_iterations]
            valid_diameters = [self.stats['diameter'][i] for i in valid_iterations]
            
            axes[1, 0].plot(valid_iterations, valid_path_lengths, 'y-', label='Avg Path Length')
            axes[1, 0].plot(valid_iterations, valid_diameters, 'k-', label='Diameter')
            axes[1, 0].set_title('Path Statistics')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Length')
            axes[1, 0].legend()
        
        # Plot modularity
        valid_iterations = [i for i, v in enumerate(self.stats['modularity']) if v > 0]
        if valid_iterations:
            valid_modularity = [self.stats['modularity'][i] for i in valid_iterations]
            
            axes[1, 1].plot(valid_iterations, valid_modularity, 'r-')
            axes[1, 1].set_title('Modularity')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Modularity')
        
        # Plot LCC size
        axes[1, 2].plot(iterations, self.stats['lcc_size'], 'b-')
        axes[1, 2].set_title('Largest Connected Component Size')
        axes[1, 2].set_xlabel('Iteration')
        axes[1, 2].set_ylabel('Size')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        plt.show()
    
    def identify_hubs_and_bridges(self, top_n=5):
        """Identify hub nodes and bridge nodes in the graph."""
        if self.graph.number_of_nodes() < 3:
            return {"hubs": [], "bridges": []}
        
        # Calculate centrality metrics
        betweenness = nx.betweenness_centrality(self.graph)
        degree = dict(self.graph.degree())
        
        # Sort nodes by centrality measures
        sorted_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
        sorted_degree = sorted(degree.items(), key=lambda x: x[1], reverse=True)
        
        # Get top hub nodes (by degree centrality)
        hubs = [node for node, _ in sorted_degree[:top_n]]
        
        # Get top bridge nodes (by betweenness centrality)
        bridges = [node for node, _ in sorted_betweenness[:top_n]]
        
        return {
            "hubs": hubs,
            "bridges": bridges
        }
    
    def get_scale_free_properties(self):
        """Calculate properties to determine if the graph exhibits scale-free characteristics."""
        if self.graph.number_of_nodes() < 10:
            return {"is_scale_free": False, "exponent": None, "r_squared": None}
        
        # Get degree distribution
        degree_sequence = sorted([d for n, d in self.graph.degree()], reverse=True)
        degree_count = Counter(degree_sequence)
        
        x = np.array(list(degree_count.keys()))
        y = np.array(list(degree_count.values()))
        
        # Skip if not enough unique degree values
        if len(x) < 3:
            return {"is_scale_free": False, "exponent": None, "r_squared": None}
        
        # Log transform for power law fit
        log_x = np.log10(x)
        log_y = np.log10(y)
        
        # Linear regression on log-log data
        try:
            coeffs = np.polyfit(log_x, log_y, 1)
            exponent = -coeffs[0]  # Negative slope is the power law exponent
            
            # Calculate R-squared
            p = np.poly1d(coeffs)
            y_pred = p(log_x)
            y_mean = np.mean(log_y)
            
            ss_total = np.sum((log_y - y_mean)**2)
            ss_residual = np.sum((log_y - y_pred)**2)
            
            r_squared = 1 - (ss_residual / ss_total)
            
            # Scale-free networks typically have exponents between 2 and 3
            # and high R-squared values
            is_scale_free = (2 <= exponent <= 3) and (r_squared > 0.8)
            
            return {
                "is_scale_free": is_scale_free,
                "exponent": exponent,
                "r_squared": r_squared
            }
        except:
            return {"is_scale_free": False, "exponent": None, "r_squared": None}

# Usage example
def run_demonstration(initial_prompt="Impact-resistant materials and their applications", iterations=50):
    """Run a demonstration of the agentic graph reasoning system."""
    agr = AgenticGraphReasoning(initial_prompt=initial_prompt)
    agr.iterate(iterations=iterations)
    
    print(f"Graph after {iterations} iterations:")
    print(f"Nodes: {agr.graph.number_of_nodes()}")
    print(f"Edges: {agr.graph.number_of_edges()}")
    
    # Identify hubs and bridges
    key_nodes = agr.identify_hubs_and_bridges(top_n=5)
    print("\nHub nodes (highest degree):", key_nodes["hubs"])
    print("Bridge nodes (highest betweenness):", key_nodes["bridges"])
    
    # Check if the graph is scale-free
    scale_free_props = agr.get_scale_free_properties()
    if scale_free_props["is_scale_free"]:
        print(f"\nThe graph exhibits scale-free properties with exponent: {scale_free_props['exponent']:.2f}")
        print(f"R-squared: {scale_free_props['r_squared']:.2f}")
    else:
        print("\nThe graph does not yet exhibit clear scale-free properties.")
    
    # Visualize the graph
    agr.visualize_graph()
    
    # Plot metrics evolution
    agr.plot_graph_metrics()
    
    return agr

if __name__ == "__main__":
    # Run the demonstration
    agr = run_demonstration(iterations=50)
